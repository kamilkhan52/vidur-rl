"""Where do the ES policies gain over 4873: replica count or load? (plan §11 A2, 2026-10-15)

The 16R test configurations replay the traces at 2x the recorded arrival rate (scale 0.5), so replica count and
arrival rate are confounded; per replica, 16R at 0.5 carries the load of 8R at 1.0. This map evaluates the frozen
policies and 4873 on:
- a fixed per-replica-load series: 8R x 1.0, 12R x 0.667, 16R x 0.5, 24R x 0.333;
- the knee-equivalent pair: 8R x 0.75 and 16R x 0.375;
- load sweeps: 8R x 0.5 (heavier) and 16R x 0.75 and 16R x 1.0 (lighter).

Draws p44-p46 have never been used (training p22-39, learning curves p40-43, heldout3 p1-5, test3b p9-13). Per
configuration: geometric-mean ratio over draws for every metric and for tails4, means4 and gen8, with the per-draw
values. Diagnostic only, never used to pick a policy.

    python -m evolve.rl_plan.regime_map --workers 16 --out evolve/rl_plan/out/regime_map.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan.es_validate import COMBINED, FINAL_DIR, METS

CONFIGS = [(8, 1.0), (12, 0.667), (16, 0.5), (24, 0.333), (8, 0.75), (16, 0.375), (8, 0.5), (16, 0.75), (16, 1.0)]
TRACES = ("Tool", "Conv", "Synth")
DRAWS = (44, 45, 46)
POLICIES = ("es_lin9_20", "es_mlp9_10")


def scenarios(configs=CONFIGS, traces=TRACES, draws=DRAWS):
    return [ev.Scenario(t, R, s, -1, 0, d, model="l3_8b", jitter_s=ev.SYNTH_JITTER_S if t == "Synth" else -1.0)
            for t in traces for (R, s) in configs for d in draws]


def summarise(scen, res, pol, ref) -> dict:
    out = {}
    for t in TRACES:
        for (R, s) in CONFIGS:
            sel = [sc for sc in scen if sc.trace == t and sc.replicas == R and sc.scale == s]
            row = {}
            for m in METS:
                d = [np.log(res[(pol, sc)][m] / res[(ref, sc)][m]) for sc in sel]
                row[m] = {"geo": float(np.exp(np.mean(d))), "draws": [float(np.exp(x)) for x in d]}
            for name, ms in COMBINED.items():
                d = [np.mean([np.log(res[(pol, sc)][m] / res[(ref, sc)][m]) for m in ms]) for sc in sel]
                row[name] = {"geo": float(np.exp(np.mean(d))), "draws": [float(np.exp(x)) for x in d]}
            row["per_replica_load"] = round(8.0 / (R * s), 3)  # in units of 8R x 1.0
            out[f"{t}:{R}:{s}"] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=10800.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), help="frozen policy names in finalists/rl_es")
    a = ap.parse_args(argv)
    scen = scenarios(draws=tuple(a.draws))
    ref = f"policy:{Path(c0.BASE).resolve()}"
    pols = {p: f"policy:{(FINAL_DIR / f'{p}.py').resolve()}" for p in a.policies}  # p may be "gate_sweep/<name>"
    res = ev.ensure_baselines(scen, [ref] + list(pols.values()), workers=a.workers, timeout=a.timeout)
    out = {"code_version": ev.code_version(), "draws": list(a.draws),
           "policies": {p: summarise(scen, res, name, ref) for p, name in pols.items()},
           "ref_abs": {sc.tag: {m: res[(ref, sc)][m] for m in METS} for sc in scen}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for p in a.policies:
        print("==", p)
        for k, row in out["policies"][p].items():
            print(f"{k:16s} load/rep {row['per_replica_load']:5.2f}  tails4 {row['tails4']['geo']:.3f}  "
                  f"means4 {row['means4']['geo']:.3f}  gen8 {row['gen8']['geo']:.3f}  p50_ttft {row['p50_ttft']['geo']:.3f}")


if __name__ == "__main__":
    main()
