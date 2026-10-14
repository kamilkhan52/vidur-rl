"""R0 (2026-10-14): how reliable are C0's per-decision labels? A diagnostic for the on-policy learner (option 3 after
step C), not a gate.

C0 scored each candidate replica with one rollout on the recorded future. Its gain model explained none of the gain's
variance. R0 measures how much of a label is reproducible: at a sample of C0's logged decisions, every candidate is
rolled out again under two other futures (every arrival after the decision jittered by U(-5 ms, 5 ms), the noise
controls' rule, seeds 1 and 2; the same jitter for all 8 candidates of a replicate, so the gains compare candidates
under one future). The logged rollout is replicate 0.

``--mode analyze`` reports per target (C7 score over the full window, the 30/60/120 s affected subsets, single terms):
the correlation of the alternatives' gains between replicates, pooled and within decision (what choosing a replica
needs); the gain in replicate b of the alternative that is best in replicate a (cross-fitted hindsight gain); the
argmax agreement; and from the within-decision correlation r the number of futures K a label must average to reach a
reliability of 0.5 or 0.8 (Spearman-Brown: K = q (1 - r) / (r (1 - q))).

Usage (Alpine tree with C0's logs):
    python -m evolve.rl_plan.r0_probe --mode run --window Tool:0:20 --every 48 --workers 8 --out evolve/rl_plan/out/r0
    python -m evolve.rl_plan.r0_probe --mode analyze --out evolve/rl_plan/out/r0
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from evolve.rl_plan import c0
from evolve.rl_plan import headroom as hr
from evolve.rl_plan.c0_explore import _score, _terms
from evolve import evaluate as ev

C0_OUT = Path("evolve/rl_plan/out/c0")
SEEDS = (1, 2)
DELTA = hr.NOISE_DELTA
TARGETS = ("full", "a120", "a60", "a30", "full_sum", "a30_sum", "full_hinge")


def child(h, choice, taus, cut, seed, delta=None):
    hr.jitter_future(h, DELTA if delta is None else delta, seed)
    return c0.c0_child(h, choice, taus, cut)


def run_window(sc, out: Path, every: int, workers: int, timeout: float, c0_dir: Path = C0_OUT,
               delta=None) -> dict:
    d = c0.window_dir(c0_dir, sc)
    summ = json.loads((d / "summary.json").read_text())
    lines = (d / "log.jsonl").read_text().splitlines()
    header = json.loads(lines[0])["header"]
    if header["decision_sha"] != c0.c0_sha() or header["code_version"] != ev.code_version():
        raise SystemExit(f"{sc.tag}: C0 log from other code")
    rows = {r["k"]: r for r in map(json.loads, lines[1:]) if r["j"] % every == 0}
    path = out / f"{sc.tag}.jsonl"
    done = {json.loads(x)["k"] for x in path.read_text().splitlines()} if path.exists() else set()
    taus, cut = header["taus"], header["cut"]
    c0.install_capture()
    t0, n_new = time.time(), 0
    with tempfile.TemporaryDirectory() as tmp, open(path, "a") as f:
        h = c0.build_simulator(hr.argv_for(sc, tmp, policy=c0.BASE))
        while h.next_decision() and len(done) < len(rows):
            n = h.decisions
            k = n + 1
            if k in rows and k not in done:
                r = rows[k]
                t = h.sim._scheduler._request_queue[0].arrived_at
                if t != r["t"]:
                    raise RuntimeError(f"{sc.tag}: decision {k} at t={t}, logged at {r['t']}")
                reps = {"0": r["cands"]}
                for s in SEEDS:
                    res = hr._run_children([lambda c=c, s=s: child(h, c, taus, cut, s, delta) for c in range(sc.replicas)],
                                           workers, timeout)
                    if any(x["policy_choice"] != r["pe"] or x["executed"] != x["choice"] for x in res):
                        raise RuntimeError(f"{sc.tag}: rollouts at decision {k} disagree with the log")
                    reps[str(s)] = {str(x["choice"]): x["t"] for x in res}
                f.write(json.dumps({"k": k, "j": r["j"], "pe": r["pe"], "w": summ["w"], "s_tbt": summ["s_tbt"],
                                    "reps": reps}) + "\n")
                f.flush()
                done.add(k)
                n_new += 1
            h.run(lambda x: x.decisions > n)
    return {"window": sc.tag, "decisions": len(done), "new": n_new, "wall_s": time.time() - t0}


def _gains(row: dict, rep: str, target: str) -> np.ndarray:
    R = len(row["reps"][rep])
    s = [_score(_terms(row["reps"][rep][str(c)], target), row["w"], row["s_tbt"], target) for c in range(R)]
    return np.array([s[row["pe"]] - s[c] for c in range(R)])


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def _k_needed(r: float, q: float):
    return None if not r > 0 else float(q * (1 - r) / (r * (1 - q)))


def analyze(out: Path) -> dict:
    rows = [json.loads(x) for p in sorted(out.glob("*.jsonl")) for x in p.read_text().splitlines()]
    rows = [r for r in rows if all(s in r["reps"] for s in ("0", "1", "2"))]
    res = {"decisions": len(rows), "windows": len(list(out.glob("*.jsonl"))), "delta_s": DELTA, "targets": {}}
    for tg in TARGETS:
        G = {s: [] for s in ("0", "1", "2")}
        for r in rows:
            for s in G:
                g = _gains(r, s, tg)
                G[s].append(np.delete(g, r["pe"]))  # alternatives only
        G = {s: np.array(v) for s, v in G.items()}  # decisions x (R-1)
        out_t = {"gain_sd": float(G["1"].std())}
        for a, b in (("1", "2"), ("0", "1"), ("0", "2")):
            A, B = G[a], G[b]
            dA, dB = A - A.mean(axis=1, keepdims=True), B - B.mean(axis=1, keepdims=True)
            best = A.argmax(axis=1)
            gb = B[np.arange(len(B)), best]
            agree = float(np.mean((A.max(axis=1) > 0) == (B.max(axis=1) > 0)))
            same = float(np.mean(np.where(A.max(axis=1) > 0, best + 1, 0) == np.where(B.max(axis=1) > 0,
                                                                                  B.argmax(axis=1) + 1, 0)))
            out_t[f"{a}v{b}"] = {"corr_pooled": _corr(A.ravel(), B.ravel()),
                                 "corr_within_decision": _corr(dA.ravel(), dB.ravel()),
                                 "hindsight_gain_same_future": float(A.max(axis=1).clip(min=0).mean()),
                                 "hindsight_gain_cross_fitted": float(np.where(A.max(axis=1) > 0, gb, 0).mean()),
                                 "hindsight_gain_cross_fitted_se": float(np.where(A.max(axis=1) > 0, gb, 0).std()
                                                                         / np.sqrt(len(gb))),
                                 "override_agreement": agree, "same_choice_incl_base": same}
        r12 = out_t["1v2"]["corr_within_decision"]
        out_t["futures_needed"] = {"reliability_0.5": _k_needed(r12, 0.5), "reliability_0.8": _k_needed(r12, 0.8)}
        res["targets"][tg] = out_t
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=("run", "analyze"))
    ap.add_argument("--window")
    ap.add_argument("--every", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--child-timeout", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hr.set_base_policy(c0.BASE)
    if a.mode == "run":
        if a.every % c0.EVERY["c0"]:
            raise SystemExit("--every must be a multiple of C0's spacing")
        t, s_, p = a.window.split(":")
        sc = ev.Scenario(t, 8, 0.75, int(s_), c0.LEN, int(p), model="l3_8b")
        print(json.dumps(run_window(sc, out, a.every, a.workers, a.child_timeout)))
    else:
        res = analyze(out)
        (out / "analysis.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
