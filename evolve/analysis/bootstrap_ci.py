"""
Confidence intervals for P99 differences between a policy and the per-scenario best
heuristic, by moving-block bootstrap over arrival time.

Requests in one run are not independent (queues couple neighbours in time), so a plain
bootstrap over requests would understate the uncertainty. We cut each run into blocks of
BLOCK_S seconds of arrival time and resample whole blocks with replacement; the SAME block
indices are used for the policy run and every baseline run of that scenario (paired: the
runs replay the same arrivals), so the CI is for the ratio under common random numbers.

For each scenario and metric: point ratio P99(policy)/P99(best heuristic), where "best" is
chosen on the full run as in the fitness, and a 95 % percentile interval. Across the
held-out set: the geometric-mean ratio with an interval from resampling all scenarios jointly.

    python -m evolve.analysis.bootstrap_ci --policy evolve/finalists/claude_r1_76be082eb0f5.py \
        --stage heldout --out-dir evolve/analysis/out/ci_claude_r1 --workers 16
"""

import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evolve import evaluate as ev
from evolve.analysis.mechanism import simulate

BLOCK_S = 60.0
N_BOOT = 2000
METRICS = ("e2e", "ttft", "tpot")


def per_request(path):
    df = pd.read_csv(path, usecols=["request_arrived_at", "request_e2e_time", "prefill_e2e_time",
                                    "request_num_decode_tokens"])
    e2e, ttft, n = df["request_e2e_time"].to_numpy(), df["prefill_e2e_time"].to_numpy(), \
        df["request_num_decode_tokens"].to_numpy()
    tpot = np.where(n > 1, (e2e - ttft) / np.maximum(n - 1, 1), np.nan)
    block = np.floor((df["request_arrived_at"].to_numpy() - df["request_arrived_at"].min()) / BLOCK_S).astype(int)
    return {"e2e": e2e, "ttft": ttft, "tpot": tpot, "block": block}


def p99(values, idx):
    v = values[idx]
    v = v[~np.isnan(v)]
    return np.percentile(v, 99) if len(v) else np.nan


def scenario_ci(runs: dict, rng, n_boot=N_BOOT) -> dict:
    """runs: scheduler -> per_request dict; key 'evolved' is the policy."""
    pol = runs["evolved"]
    nblocks = int(max(r["block"].max() for r in runs.values())) + 1
    members = {s: [np.flatnonzero(r["block"] == b) for b in range(nblocks)] for s, r in runs.items()}
    out = {}
    base_names = [s for s in runs if s != "evolved"]
    for m in METRICS:
        full_best = min(base_names, key=lambda s: p99(runs[s][m], np.arange(len(runs[s][m]))))
        point = p99(pol[m], np.arange(len(pol[m]))) / p99(runs[full_best][m], np.arange(len(runs[full_best][m])))
        out[m] = {"ratio": float(point), "best_heuristic": full_best, "boot": []}
    for _ in range(n_boot):
        pick = rng.integers(0, nblocks, nblocks)
        idx = {s: np.concatenate([members[s][b] for b in pick]) for s in runs}
        for m in METRICS:
            b = out[m]["best_heuristic"]
            out[m]["boot"].append(float(p99(pol[m], idx["evolved"]) / p99(runs[b][m], idx[b])))
    for m in METRICS:
        bt = np.array(out[m]["boot"])
        out[m]["ci95"] = [float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))]
        out[m]["p_not_better"] = float(np.mean(bt >= 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--stage", default="heldout", choices=sorted(ev.STAGES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    scen = ev.STAGES[a.stage]
    jobs = [(b, sc, None) for sc in scen for b in ev.DEFAULT_BASELINES] + [("evolved", sc, a.policy) for sc in scen]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        paths = list(ex.map(lambda j: simulate(j[0], j[1], j[2], out / "runs"), jobs))
    by = {}
    for (s, sc, _), p in zip(jobs, paths):
        by.setdefault(sc.tag, {})[s] = per_request(p)
    rng = np.random.default_rng(a.seed)
    res = {tag: scenario_ci(runs, rng) for tag, runs in by.items()}
    # joint geo-mean interval: same bootstrap draw index across scenarios is not meaningful
    # (independent runs), so resample each scenario's bootstrap replicates independently.
    summary = {}
    for m in METRICS:
        logs = np.array([[math.log(x) for x in res[t][m]["boot"]] for t in res])  # [S, B]
        point = math.exp(np.mean([math.log(res[t][m]["ratio"]) for t in res]))
        geo = np.exp(logs.mean(axis=0))
        summary[m] = {"geo_ratio": point, "ci95": [float(np.percentile(geo, 2.5)), float(np.percentile(geo, 97.5))],
                      "scenarios_ci_below_1": sum(res[t][m]["ci95"][1] < 1 for t in res),
                      "scenarios_ci_above_1": sum(res[t][m]["ci95"][0] > 1 for t in res), "n": len(res)}
    for t in res:
        for m in METRICS:
            del res[t][m]["boot"]
    (out / "ci.json").write_text(json.dumps({"block_s": BLOCK_S, "n_boot": N_BOOT, "summary": summary,
                                             "scenarios": res}, indent=1))
    print(json.dumps(summary, indent=1))
    for t, r in res.items():
        print(f"{t:28s} " + "  ".join(f"{m} {r[m]['ratio']:.3f} [{r[m]['ci95'][0]:.3f},{r[m]['ci95'][1]:.3f}]"
                                        for m in METRICS))


if __name__ == "__main__":
    main()
