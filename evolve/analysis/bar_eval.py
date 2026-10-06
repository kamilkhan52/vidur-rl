"""
Score policies against the production-aware bar: per scenario and metric, the best of the five heuristics plus
our Mooncake and Dynamo re-implementations (AIBrix/SGLang are dominated and omitted). Reports the held-out
geo-mean ratio overall and by load regime (knee = 8R scale 0.75; scale 1.0; 16R), win counts, and fitness.

    python -m evolve.analysis.bar_eval --policy evolve/probes/rule_inflight.py --policy evolve/finalists/claude_r1_76be082eb0f5.py
"""

import argparse
import json
import math
from pathlib import Path

from evolve import evaluate as ev

REPO = Path(__file__).resolve().parents[2]
SYS = [f"policy:{REPO / 'evolve/policies/systems' / n}.py" for n in ("mooncake_min_ttft", "dynamo_kv_cost")]
BAR = list(ev.DEFAULT_BASELINES) + SYS


def regime(tag: str) -> str:
    return "16R" if "16R" in tag else ("knee" if "s0.75" in tag else "s1.0")


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--stage", default="heldout", choices=sorted(ev.STAGES))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scen = ev.STAGES[a.stage]
    pols = [f"policy:{Path(p).resolve()}" for p in a.policy]
    base = ev.ensure_baselines(scen, BAR + pols, workers=a.workers)
    report = {}
    for p in pols:
        rows = {}
        for sc in scen:
            rows[sc.tag] = {m: base[(p, sc)][f"p99_{m}"] / min(base[(b, sc)][f"p99_{m}"] for b in BAR)
                            for m in ev.METRICS}
        out = {"overall": {}, "by_regime": {}}
        for m in ev.METRICS:
            xs = list(rows.values())
            out["overall"][m] = {"geo": round(geo([r[m] for r in xs]), 4), "better": sum(r[m] < 1 for r in xs), "n": len(xs)}
        for g in ("knee", "s1.0", "16R"):
            xs = [r for t, r in rows.items() if regime(t) == g]
            if xs:
                out["by_regime"][g] = {m: {"geo": round(geo([r[m] for r in xs]), 4), "better": sum(r[m] < 1 for r in xs),
                                           "n": len(xs)} for m in ev.METRICS}
        fit = ev.score({sc: base[(p, sc)] for sc in scen}, base, BAR)
        out["fitness_vs_bar"] = round(fit["combined_score"], 4)
        out["per_scenario"] = {t: {m: round(v, 4) for m, v in r.items()} for t, r in rows.items()}
        report[Path(p[7:]).name] = out
        print(f"\n{Path(p[7:]).name}  fitness vs bar {out['fitness_vs_bar']:+.4f}")
        print("  overall " + "  ".join(f"{m} {v['geo']:.3f} ({v['better']}/{v['n']})" for m, v in out["overall"].items()))
        for g, d in out["by_regime"].items():
            print(f"  {g:5s}   " + "  ".join(f"{m} {v['geo']:.3f} ({v['better']}/{v['n']})" for m, v in d.items()))
    if a.out:
        Path(a.out).write_text(json.dumps({"bar": BAR, "stage": a.stage, "code_version": ev.code_version(), "policies": report}, indent=1))


if __name__ == "__main__":
    main()
