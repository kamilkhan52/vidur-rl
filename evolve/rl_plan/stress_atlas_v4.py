"""Where does 4873 leave room? The v4t bar and 4873 on stress workloads (load ramps, bursts, two-workload mixes;
``evaluate.TRANSFORMS``) and the plain traces, Llama-3-8B, 8 replicas, loads 1.0 and 0.75, training draw p22 only (a
diagnostic for choosing RL training regimes; validation and test draws stay unseen).

Per scenario and P99 metric: 4873's ratio to the best bar router, and that router.

    python -m evolve.rl_plan.stress_atlas_v4 --workers 16 --out evolve/rl_plan/out/stress_atlas_v4.json
"""
import argparse
import json
from pathlib import Path

from evolve import evaluate as ev
from evolve.rl_plan import c0

MIX = {"Tool": "mix-Synth", "Conv": "mix-Synth", "Synth": "mix-Tool"}
METS = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e")


def scenarios(draw: int = 22):
    out = []
    for t in ("Tool", "Conv", "Synth"):
        for s in (1.0, 0.75):
            for x in ("", "ramp", "burst", MIX[t]):
                out.append(ev.Scenario(t, 8, s, -1, 0, draw, model="l3_8b", transform=x,
                                       jitter_s=ev.SYNTH_JITTER_S if t == "Synth" else -1.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    scen = scenarios()
    pol = f"policy:{Path(c0.BASE).resolve()}"
    names = list(ev.BARS["v4t"]) + [pol]
    res = ev.ensure_baselines(scen, names, workers=a.workers)
    rows = []
    for sc in scen:
        r = res[(pol, sc)]
        row = {"scenario": sc.tag}
        for m in METS:
            best = min((res[(b, sc)][m], b) for b in ev.BARS["v4t"])
            row[m] = {"4873": r[m], "ratio_to_best": r[m] / best[0], "best": Path(best[1]).stem}
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(f"{row['scenario']:45s} " + " ".join(f"{m[4:]} {row[m]['ratio_to_best']:.3f}" for m in METS[:4]))


if __name__ == "__main__":
    main()
