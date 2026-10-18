"""r6k_val comparisons against 4873 (evolve/r6k/PREREG.md §5) and the 4873-r6k adoption rule.

Runs (or reads from the cache) every policy and BAR_V4 on `r6k_val`, then reports each policy / 4873 on the
policy_table metrics with the configuration-cluster bootstrap. 4873-r6k is adopted iff its mean E2E ratio's 95 %
upper bound is < 1 and its P99 E2E ratio's 95 % lower bound is <= 1.

  python -m evolve.r6k.val_compare --workers 30 --out evolve/r6k/out/val_compare.json
"""
import argparse
import json
from pathlib import Path

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import clusters
from evolve.analysis.policy_table import METS, compare, short, values

BASE = ev.REPO / "evolve/finalists/v4/v4_487363424749.py"
CANDIDATE = ev.REPO / "evolve/policies/tuned_r6k/v4_487363424749_best.py"
OTHERS = [ev.REPO / "evolve/policies/tuned_r6k/mooncake_min_ttft_tuned_best.py",
          ev.REPO / "evolve/policies/tuned_r6k/dynamo_kv_cost_tuned_best.py",
          ev.REPO / "evolve/policies/tuned/mooncake_min_ttft_tuned_best.py",
          ev.REPO / "evolve/policies/tuned/dynamo_kv_cost_tuned_best.py"]


def label(p: str) -> str:
    return short(p) + (" (r6k)" if "/tuned_r6k/" in p else " (A100)" if "/tuned/" in p else "")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="r6k_val")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    scen = list(ev.STAGES[a.stage])
    bar = list(ev.BAR_V4)
    base, cand = f"policy:{BASE}", f"policy:{CANDIDATE}"
    pols = [base, cand] + [f"policy:{p}" for p in OTHERS]
    ev.ensure_baselines(scen, bar + pols, workers=a.workers)
    vals = values(bar + pols, scen, ev.load_baseline_cache())
    groups = clusters([sc.tag for sc in scen])
    tab = {label(p): compare(vals[p], vals[base], groups, a.boot) for p in pols[1:] + bar}
    c = tab[label(cand)]
    adopt = c["mean_e2e"]["hi"] < 1 and c["p99_e2e"]["lo"] <= 1
    out = {"stage": a.stage, "code_version": ev.code_version(), "n_scenarios": len(scen),
           "n_configurations": len(groups), "reference": short(base), "candidate": str(CANDIDATE.relative_to(ev.REPO)),
           "rule": "adopt iff mean_e2e hi < 1 and p99_e2e lo <= 1", "adopt_4873_r6k": adopt, "vs_4873": tab}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{a.stage}: {len(scen)} scenarios, {len(groups)} configurations; ratio to 4873, geo [95% CI] (wins)")
    for name, r in tab.items():
        print(name[:40].ljust(41) + "  ".join(f"{m} {r[m]['geo']:.3f} [{r[m]['lo']:.3f},{r[m]['hi']:.3f}] ({r[m]['wins']})"
                                             for m in ("mean_e2e", "p99_e2e", "mean_ttft", "p50_e2e")))
    print(f"4873-r6k adopted: {adopt}")
    return out


if __name__ == "__main__":
    main()
