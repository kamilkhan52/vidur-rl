"""
Held-out evaluation of several frozen finalists on ONE machine (shared baselines).

    python -m evolve.analysis.heldout_compare --out evolve/analysis/heldout_compare.json \
        --workers 16 evolve/finalists/claude_r1_76be082eb0f5.py evolve/finalists/deepseek_r1_c5b7cd41164e.py
"""

import argparse
import json
import math
from pathlib import Path

from evolve import evaluate as ev


def ratios(res, metric):
    r = [d["p99"] / d["best_heuristic_p99"] for d in res["details"] if d["metric"] == metric]
    return {"better": sum(x < 1 for x in r), "n": len(r), "geo": math.exp(sum(map(math.log, r)) / len(r)),
            "min": min(r), "max": max(r)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stage", default="heldout", choices=sorted(ev.STAGES))
    a = ap.parse_args()
    scen = ev.STAGES[a.stage]
    ev.ensure_baselines(scen, workers=a.workers)
    out = {"code_version": ev.code_version(), "stage": a.stage, "policies": {}}
    for p in a.policies:
        res = ev.evaluate_policy(p, scen, workers=a.workers)
        out["policies"][Path(p).stem] = {"result": res, **({m: ratios(res, m) for m in ev.METRICS}
                                                         if res.get("valid") else {})}
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(Path(p).stem, round(res["combined_score"], 4),
              {m: out["policies"][Path(p).stem].get(m) for m in ev.METRICS}, flush=True)


if __name__ == "__main__":
    main()
