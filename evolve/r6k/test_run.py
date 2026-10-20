"""r6k_test (PREREG amendment 3): the simulation's prediction for the hardware main comparison (M5), run once.

Runs the 9 hardware routers on `r6k_test` (ToolHW/ConvHW first 1000 requests, 4 replicas, draws 1-6, each candidate
high load) and scores each load with M5's success rule (vidur-realsys prereg 2026-10-19, 1cac8d4):
- primary: for each baseline b, the geometric mean over the 12 cells (trace x draw) of mean E2E(4873) / mean E2E(b),
  with a two-sided 95 % t-interval on the log ratios (11 df); pass iff its upper bound < 1; all baselines must pass;
- RL: geometric mean of mean E2E(es) / mean E2E(4873), pass iff the 95 % upper bound < 1.03 (the simulation has no
  hardware run-to-run noise, so max(3 %, 2 sigma_rep) is 3 %).
Reported, not judged: the other metrics as ratios to 4873, per trace and per cell.

  python -m evolve.r6k.test_run --workers 16 --out evolve/r6k/out/r6k_test.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from evolve import evaluate as ev

POL = ev.REPO / "evolve/policies"
ROUTERS = {  # name -> scheduler, as named in the M5 prereg
    "4873": f"policy:{ev.REPO / 'evolve/finalists/v4/v4_487363424749.py'}",
    "es": f"policy:{ev.REPO / 'evolve/finalists/rl_es/es_gen8div_b0_40.py'}",
    "mooncake0": f"policy:{POL / 'systems/mooncake_min_ttft.py'}",
    "mooncake6": f"policy:{POL / 'tuned_r6k/mooncake_min_ttft_tuned_best.py'}",
    "dynamo6": f"policy:{POL / 'tuned_r6k/dynamo_kv_cost_tuned_best.py'}",
    "dynamo": f"policy:{POL / 'systems/dynamo_kv_cost.py'}",
    "lor": "lor",
    "sticky_lor": "sticky_lor",
    "lop_uncached": "lop_uncached",
}
BASELINES = [k for k in ROUTERS if k not in ("4873", "es")]
REPORTED = ("mean_e2e", "p50_e2e", "mean_ttft", "p50_tpot", "p99_e2e", "p99_ttft", "p99_tbt")
RL_MARGIN = 0.03


def t_ratio(num: np.ndarray, den: np.ndarray) -> dict:
    d = np.log(num / den)
    n, m = len(d), float(d.mean())
    h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"geo": round(math.exp(m), 4), "lo": round(math.exp(m - h), 4), "hi": round(math.exp(m + h), 4),
            "lower_in": int((d < 0).sum()), "n": n}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    scen = list(ev.STAGES["r6k_test"])
    res = ev.ensure_baselines(scen, list(ROUTERS.values()), workers=a.workers)
    cache = ev.load_baseline_cache()

    def row(name, sc):
        r = cache.get(ev.baseline_key(ROUTERS[name], sc)) or res.get((ROUTERS[name], sc))
        if not r or not r.get("ok", True):
            raise KeyError(f"no result for {name} on {sc.tag}")
        return r

    out = {"stage": "r6k_test", "code_version": ev.code_version(), "rule": __doc__.split("Reported")[0].strip(),
           "routers": {k: v for k, v in ROUTERS.items()}, "loads": {}}
    for s in ev.R6K_TEST_SCALES:
        cells = [sc for sc in scen if sc.scale == s]
        val = {k: {m: np.array([float(row(k, sc)[m]) for sc in cells]) for m in REPORTED} for k in ROUTERS}
        prim = {b: t_ratio(val["4873"]["mean_e2e"], val[b]["mean_e2e"]) for b in BASELINES}
        rl = t_ratio(val["es"]["mean_e2e"], val["4873"]["mean_e2e"])
        rep = {k: {m: t_ratio(val[k][m], val["4873"][m]) for m in REPORTED} for k in ROUTERS if k != "4873"}
        per_trace = {}
        for tr in ("ToolHW", "ConvHW"):
            ix = np.array([sc.trace == tr for sc in cells])
            per_trace[tr] = {k: {m: t_ratio(val[k][m][ix], val["4873"][m][ix]) for m in REPORTED}
                             for k in ROUTERS if k != "4873"}
        out["loads"][str(s)] = {
            "cells": [sc.tag for sc in cells],
            "primary_4873_over_baseline": prim,
            "primary_pass": all(r["hi"] < 1 for r in prim.values()),
            "rl_es_over_4873": rl, "rl_pass": rl["hi"] < 1 + RL_MARGIN,
            "reported_vs_4873": rep, "per_trace_vs_4873": per_trace,
            "per_cell": {sc.tag: {k: {m: round(float(row(k, sc)[m]), 4) for m in REPORTED} for k in ROUTERS}
                         for sc in cells},
        }
        print(f"scale {s}: 4873 / baseline, mean E2E, geo [95 % t] (cells where 4873 is lower)")
        for b, r in prim.items():
            print(f"  {b:13s} {r['geo']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] ({r['lower_in']}/{r['n']})")
        print(f"  primary pass: {out['loads'][str(s)]['primary_pass']};  es / 4873 {rl['geo']:.4f} "
              f"[{rl['lo']:.4f}, {rl['hi']:.4f}], RL pass: {rl['hi'] < 1 + RL_MARGIN}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
