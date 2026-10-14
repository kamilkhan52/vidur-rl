"""Exploratory, not a gate (2026-10-14, after C0 failed items 1-2): is the gain over a shorter horizon, or a single
term of the C7 score, more learnable than the full-window C7 score C0 used?

For each target, the same gain model as C0 (``c0.MODEL_PARAMS``, target clipped as in ``c0.fit_model``) is fitted
leave-one-range-out; reported on the held-out decisions: explained variance over all alternatives, the mean
within-range Spearman correlation of the best alternative's predicted and recorded gain, the mean recorded gain of
the best alternative where its prediction is positive (and how often), and the gain's spread (sd, IQR).

Reads the C0 rows (j % 6 == 0) directly and skips an unparsable last line, so it can run while C0b appends to the
logs.  Usage: python -m evolve.rl_plan.c0_explore --out evolve/rl_plan/out/c0 [--json result.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evolve.rl_plan import c0
from evolve.rl_plan import c0_features as cf

TARGETS = ("full", "a120", "a60", "a30", "full_sum", "a30_sum", "full_hinge", "postcut")


def _terms(cand: dict, target: str) -> dict:
    if target in ("full", "full_sum", "full_hinge"):
        return cand
    return cand["sub"][target.split("_")[0]]


def _score(t: dict, w: float, s_tbt: float, target: str) -> float:
    if target.endswith("_sum"):
        return t["sum"]
    if target.endswith("_hinge"):
        return w * (t["h_e2e"] + t["h_ttft"] + t["h_tpot"] + s_tbt * t["h_tbt"])
    return c0.score(t, w, s_tbt)


def load(out: Path, every: int = 6) -> dict:
    X, G, rng, base = [], {k: [] for k in TARGETS}, [], []
    for sc in c0.windows("c0"):
        d = c0.window_dir(out, sc)
        s = json.loads((d / "summary.json").read_text())
        rows = []
        for line in (d / "log.jsonl").read_text().splitlines()[1:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["j"] % every == 0:
                rows.append(r)
        if len(rows) != s["complete"][str(every)]["points"]:
            raise SystemExit(f"{sc.tag}: {len(rows)} rows, completion record says {s['complete'][str(every)]['points']}")
        for r in sorted(rows, key=lambda r: r["k"]):
            b, R = r["pe"], len(r["reps"])
            X.append(cf.features(r["req"], r["reps"], b, r["hist"]))
            for k in TARGETS:
                sc_ = [_score(_terms(r["cands"][str(c)], k), s["w"], s["s_tbt"], k) for c in range(R)]
                G[k].append([sc_[b] - sc_[c] for c in range(R)])
            rng.append(c0.range_id(sc.tag))
            base.append(b)
    return {"X": np.asarray(X, float), "G": {k: np.asarray(v, float) for k, v in G.items()}, "range": np.asarray(rng),
            "base": np.asarray(base, int)}


def evaluate(data: dict, target: str) -> dict:
    ds = {"X": data["X"], "G": data["G"][target], "range": data["range"], "base": data["base"]}
    allidx = np.arange(len(ds["range"]))
    rho, ap, ag, bp, bg = [], [], [], [], []
    for g in sorted(set(ds["range"])):
        tr, te = allidx[ds["range"] != g], allidx[ds["range"] == g]
        m = c0.fit_model(*c0._alt_rows(ds, tr))
        pb = c0.predict_best(m, ds, te)
        rho.append(c0._spearman0(pb["pred"], pb["gain"]))
        xa, ya = c0._alt_rows(ds, te)
        ap.append(m.predict(xa))
        ag.append(ya)
        bp.append(pb["pred"])
        bg.append(pb["gain"])
    ap, ag, bp, bg = map(np.concatenate, (ap, ag, bp, bg))
    _, galt = c0._alt_rows(ds, allidx)
    pos = bp > 0
    return {"explained_variance_left_out": 1.0 - float(np.var(ag - ap) / np.var(ag)),
            "spearman_all_alts_pooled": c0._spearman0(ap, ag),
            "spearman_best_within_range_mean": float(np.mean(rho)), "spearman_best_per_range": rho,
            "best_pred_positive_frac": float(pos.mean()),
            "mean_gain_where_best_pred_positive": float(bg[pos].mean()) if pos.any() else None,
            "gain_sd": float(galt.std()), "gain_iqr": [float(np.percentile(galt, 25)), float(np.percentile(galt, 75))],
            "frac_alt_gain_positive": float((galt > 0).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--targets", default=",".join(TARGETS))
    a = ap.parse_args(argv)
    data = load(Path(a.out))
    res = {"decisions": int(len(data["range"])), "targets": {}}
    for t in a.targets.split(","):
        res["targets"][t] = evaluate(data, t)
        print(t, json.dumps({k: v for k, v in res["targets"][t].items() if k != "spearman_best_per_range"}), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
