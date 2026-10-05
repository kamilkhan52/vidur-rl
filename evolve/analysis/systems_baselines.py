"""
Re-implemented production routing rules (evolve/policies/systems/: Mooncake min-TTFT, SGLang
cache-aware, AIBrix prefix+load, Dynamo KV cost) as extra baselines. Reports, per scenario and
metric, each rule's P99 relative to the best of the five original heuristics, and the evolved
finalist's P99 relative to the best of all nine (heuristics + system rules).

    python -m evolve.analysis.systems_baselines --out evolve/analysis/out/systems_baselines.json --workers 6
"""

import argparse
import json
import math
from pathlib import Path

from evolve import evaluate as ev

REPO = Path(__file__).resolve().parents[2]
SYSTEMS = sorted(str(p) for p in (REPO / "evolve" / "policies" / "systems").glob("*.py"))
FINALIST = str(REPO / "evolve" / "finalists" / "claude_r1_76be082eb0f5.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    scen = ev.STAGE1 + ev.HELDOUT
    sysn = [f"policy:{p}" for p in SYSTEMS]
    names = list(ev.DEFAULT_BASELINES) + sysn + [f"policy:{FINALIST}"]
    base = ev.ensure_baselines(scen, names, workers=a.workers)
    label = {n: (Path(n[7:]).stem if n.startswith("policy:") else n) for n in names}
    rows = []
    for sc in scen:
        r = {"scenario": sc.tag, "split": "search" if sc in ev.STAGE1 else "heldout"}
        for m in ev.METRICS:
            h = min(base[(b, sc)][f"p99_{m}"] for b in ev.DEFAULT_BASELINES)
            allb = min(base[(b, sc)][f"p99_{m}"] for b in list(ev.DEFAULT_BASELINES) + sysn)
            r[m] = {"vs_best_heuristic": {label[n]: base[(n, sc)][f"p99_{m}"] / h for n in names},
                    "finalist_vs_best_of_9": base[(f"policy:{FINALIST}", sc)][f"p99_{m}"] / allb,
                    "best_of_9": min((label[n] for n in list(ev.DEFAULT_BASELINES) + sysn),
                                     key=lambda k: base[([n for n in names if label[n] == k][0], sc)][f"p99_{m}"])}
        rows.append(r)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    geo = lambda v: math.exp(sum(map(math.log, v)) / len(v))
    ho = [r for r in rows if r["split"] == "heldout"]
    print("held-out (20) geo P99 ratio to best of 5 heuristics:")
    for n in names:
        print(f"  {label[n]:28s} " + "  ".join(f"{m} {geo([r[m]['vs_best_heuristic'][label[n]] for r in ho]):.3f}" for m in ev.METRICS))
    print("finalist vs best of 9 (heuristics + system rules), held-out: " + "  ".join(
        f"{m} geo {geo([r[m]['finalist_vs_best_of_9'] for r in ho]):.3f} better "
        f"{sum(r[m]['finalist_vs_best_of_9'] < 1 for r in ho)}/{len(ho)}" for m in ev.METRICS))


if __name__ == "__main__":
    main()
