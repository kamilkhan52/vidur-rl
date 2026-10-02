"""
Failure atlas: where do fixed-rule routers break? Runs the five heuristics and reference
policies (e.g. the evolved routers) on stress workloads (evaluate.TRANSFORMS: load ramps,
bursts, two-workload mixes) and on the unstressed traces, and reports for each scenario and
metric which policy is best and how far every policy is from it.

A policy "breaks" on a scenario when it is far from the best available policy there; a rule
that is tuned for the traces it was designed on should break first on stress workloads.

    python -m evolve.analysis.stress_atlas --policy evolve/finalists/claude_r1_76be082eb0f5.py \
        --out evolve/analysis/out/stress_atlas.json --workers 16
"""

import argparse
import json
from pathlib import Path

from evolve import evaluate as ev

MIX_PARTNER = {"Tool": "mix-Synth", "Conv": "mix-Synth", "Synth": "mix-Tool"}


def atlas_scenarios(traces=("Tool", "Conv", "Synth"), scales=(1.0, 0.75), replicas=8, perturbation=1):
    out = []
    for t in traces:
        for s in scales:
            for x in ("", "ramp", "burst", MIX_PARTNER[t]):
                out.append(ev.Scenario(t, replicas, s, -1, 0, perturbation, transform=x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=[], help="reference policy file (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    scen = atlas_scenarios()
    names = list(ev.DEFAULT_BASELINES) + [f"policy:{p}" for p in a.policy]
    base = ev.ensure_baselines(scen, names, workers=a.workers)
    label = {n: (Path(n[7:]).stem if n.startswith("policy:") else n) for n in names}
    rows = []
    for sc in scen:
        r = {"scenario": sc.tag, "transform": sc.transform or "none", "trace": sc.trace, "scale": sc.scale}
        for m in ev.METRICS:
            vals = {label[n]: base[(n, sc)][f"p99_{m}"] for n in names}
            best = min(vals, key=vals.get)
            heur_best = min(vals[label[n]] for n in ev.DEFAULT_BASELINES)
            r[m] = {"best": best, "ratio_to_best": {k: v / vals[best] for k, v in vals.items()},
                    "ratio_to_best_heuristic": {k: v / heur_best for k, v in vals.items()}}
        r["mean_e2e"] = {label[n]: base[(n, sc)]["mean_e2e"] for n in names}
        rows.append(r)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    refs = [label[n] for n in names if n.startswith("policy:")]
    for r in rows:
        line = f"{r['scenario']:38s}"
        for m in ev.METRICS:
            q = r[m]
            line += f" | {m} best={q['best'][:12]:12s} " + " ".join(
                f"{k[:10]}={q['ratio_to_best_heuristic'][k]:.2f}" for k in refs)
        print(line, flush=True)


if __name__ == "__main__":
    main()
