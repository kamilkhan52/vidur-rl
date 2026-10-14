"""Learning curves of the C0 gain model (diagnostic, not a gate; 2026-10-14, after C0b failed item 1).

The gain model is fitted once on logged rollouts (no episodes), so "training longer" means either more data or more
boosting rounds. Both are measured with C0's outer leave-one-range-out split on the C0b data:

* data curve: each fold's model is trained on a fraction f of its training decisions (one fixed random order of all
  decisions, so the subsets are nested), ``c0.MODEL_PARAMS`` otherwise;
* boosting curve: each fold's model is trained on all its training decisions with ``max_iter`` = ``MAX_ITER`` and
  evaluated after ``ITERS`` rounds (``staged_predict``), on the held-out range and on a sample of its training
  decisions (in-sample).

Per point, on the held-out decisions pooled over folds: explained variance of the gain over all alternatives; the
mean within-range Spearman correlation of the best alternative's predicted and recorded gain (item 2's statistic);
and the value of acting at margin 0 (override when the best predicted gain is positive): mean recorded gain per
decision and per override, with 95 % CIs from a bootstrap over request ranges.

Usage: python -m evolve.rl_plan.c0_curves --out evolve/rl_plan/out/c0 --json evolve/rl_plan/out/c0_curves.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evolve.rl_plan import c0

FRACTIONS = (0.0625, 0.125, 0.25, 0.5, 1.0)
ITERS = (10, 25, 50, 100, 200, 300, 500, 1000)
MAX_ITER = max(ITERS)
BOOT = 2000
N_INSAMPLE = 3000


def _fit(X, y, max_iter):
    from sklearn.ensemble import HistGradientBoostingRegressor
    lo, hi = np.percentile(y, c0.CLIP_Q)
    return HistGradientBoostingRegressor(**(c0.MODEL_PARAMS | {"max_iter": max_iter})).fit(X, np.clip(y, lo, hi))


def _fold_stats(P: np.ndarray, G: np.ndarray, base: np.ndarray) -> dict:
    """P, G: decisions x R predictions and gains; the base pick's column is excluded."""
    n, R = P.shape
    mask = np.ones((n, R), bool)
    mask[np.arange(n), base] = False
    P = np.where(mask, P, -np.inf)
    best = P.argmax(axis=1)
    pb, gb = P[np.arange(n), best], G[np.arange(n), best]
    ov = pb > 0
    return {"alt_pred": P[mask], "alt_gain": G[mask], "rho": c0._spearman0(pb, gb),
            "n": n, "n_ov": int(ov.sum()), "gain_ov": float(gb[ov].sum())}


def _summ(folds: list, seed: int = 0) -> dict:
    ap = np.concatenate([f["alt_pred"] for f in folds])
    ag = np.concatenate([f["alt_gain"] for f in folds])
    rho = np.array([f["rho"] for f in folds])
    n = np.array([f["n"] for f in folds])
    nov = np.array([f["n_ov"] for f in folds])
    gov = np.array([f["gain_ov"] for f in folds])
    rng = np.random.default_rng(seed)
    bd, bo, br = [], [], []
    for _ in range(BOOT):
        k = rng.integers(0, len(folds), len(folds))
        bd.append(gov[k].sum() / n[k].sum())
        bo.append(gov[k].sum() / nov[k].sum() if nov[k].sum() else np.nan)
        br.append(rho[k].mean())
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"explained_variance": 1.0 - float(np.var(ag - ap) / np.var(ag)),
            "spearman_within_range_mean": float(rho.mean()), "spearman_ci95": ci(br),
            "gain_per_decision": float(gov.sum() / n.sum()), "gain_per_decision_ci95": ci(bd),
            "override_rate": float(nov.sum() / n.sum()),
            "gain_per_override": float(gov.sum() / nov.sum()) if nov.sum() else None, "gain_per_override_ci95": ci(bo)}


def curves(ds: dict, seed: int = 0) -> dict:
    N, R = ds["G"].shape
    order = np.random.default_rng(seed).permutation(N)
    rank = np.empty(N, int)
    rank[order] = np.arange(N)
    allidx = np.arange(N)
    data = {f: [] for f in FRACTIONS}
    boost = {i: [] for i in ITERS}
    boost_in = {i: [] for i in ITERS}
    for g in sorted(set(ds["range"])):
        tr, te = allidx[ds["range"] != g], allidx[ds["range"] == g]
        Xte = ds["X"][te].reshape(-1, ds["X"].shape[2])
        for f in FRACTIONS:
            sub = tr[rank[tr] < f * N]
            m = c0.fit_model(*c0._alt_rows(ds, sub)) if f < 1.0 else None
            if m is not None:
                data[f].append(_fold_stats(m.predict(Xte).reshape(len(te), R), ds["G"][te], ds["base"][te]))
        m = _fit(*c0._alt_rows(ds, tr), MAX_ITER)
        ins = tr[np.random.default_rng(seed).choice(len(tr), min(N_INSAMPLE, len(tr)), replace=False)]
        Xin = ds["X"][ins].reshape(-1, ds["X"].shape[2])
        for it, (pte, pin) in enumerate(zip(m.staged_predict(Xte), m.staged_predict(Xin)), start=1):
            if it in boost:
                st = _fold_stats(pte.reshape(len(te), R), ds["G"][te], ds["base"][te])
                boost[it].append(st)
                boost_in[it].append(_fold_stats(pin.reshape(len(ins), R), ds["G"][ins], ds["base"][ins]))
                if it == 300:
                    data[1.0].append(st)  # c0.MODEL_PARAMS has max_iter 300: the full-data point of the data curve
        print(f"fold {g} done", flush=True)
    return {"decisions": N, "ranges": len(set(ds["range"])),
            "data_curve": {str(f): {"train_decisions_per_fold": int(round(f * N * (1 - 1 / len(set(ds["range"]))))),
                                    **_summ(data[f])} for f in FRACTIONS},
            "boosting_curve": {str(i): {"held_out": _summ(boost[i]), "in_sample": _summ(boost_in[i])} for i in ITERS}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="c0b")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    assert c0.MODEL_PARAMS["max_iter"] == 300
    dirs = {sc.tag: c0.window_dir(Path(a.out), sc) for sc in c0.windows(a.stage)}
    ds = c0.dataset(c0.check_windows(dirs, c0.EVERY[a.stage]))
    res = curves(ds)
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
