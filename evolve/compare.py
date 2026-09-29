"""
Compare evolution runs (e.g. DeepSeek via OpenRouter vs Claude via subscription) at equal
proposal budgets.

    python -m evolve.compare evolve/runs/claude_r1 /path/to/deepseek_r1 [--at 10 20 40]

For each run and budget N (the first N LLM proposals, in archive order): best stage-1 fitness
reached, share of proposals that were valid / rejected by the guard / crashed / returned no
code, proposer spend (USD; notional API-equivalent for the subscription back end) and median
seconds per proposal. Seeds (the heuristic policies) are excluded from the counts but set the
starting point: "best seed" is the fitness of the best heuristic policy in the same archive.
Fitness comparisons across machines differ by ~1e-6 relative; re-score finalists on one
machine (`python -m evolve.evaluate --stage heldout`) before drawing conclusions.
"""

import argparse
import json
from pathlib import Path


def load(run_dir):
    rows = [json.loads(l) for l in (Path(run_dir) / "archive.jsonl").read_text().splitlines() if l.strip()]
    seeds = [r for r in rows if (r.get("proposer") or {}).get("seed")]
    props = [r for r in rows if not (r.get("proposer") or {}).get("seed")]
    return seeds, props


def summarize(seeds, props, n):
    p = props[:n]
    if not p:
        return None
    best_seed = max((r["fitness"] for r in seeds if r.get("valid")), default=float("nan"))
    best = max([r["fitness"] for r in p if r.get("valid")] + [best_seed])
    stage = lambda s: sum(1 for r in p if r.get("stage_reached") == s and not r.get("valid"))
    costs = [(r.get("proposer") or {}).get("cost_usd") or 0.0 for r in p]
    def secs_of(r):
        m = r.get("proposer") or {}
        return m.get("proposal_s") or (m.get("duration_ms") or 0) / 1000
    secs = sorted(secs_of(r) for r in p)
    return {
        "proposals": len(p),
        "best_seed_fitness": round(best_seed, 4),
        "best_fitness": round(best, 4),
        "improvement_over_best_seed": round(best - best_seed, 4),
        "valid": sum(1 for r in p if r.get("valid")),
        "guard_rejected": stage("0a"),
        "crashed": stage("0b"),
        "no_code": stage("proposal"),
        "spend_usd": round(sum(costs), 3),
        "median_proposal_s": round(secs[len(secs) // 2], 1) if any(secs) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--at", type=int, nargs="*", default=None)
    a = ap.parse_args()
    loaded = {r: load(r) for r in a.runs}
    budgets = a.at or sorted({min(len(p) for _, p in loaded.values())} | {10, 20, 40})
    for n in budgets:
        print(f"\n=== first {n} proposals")
        for run, (seeds, props) in loaded.items():
            s = summarize(seeds, props, n)
            print(f"  {Path(run).name:20s} " + (json.dumps(s) if s and s["proposals"] == n else f"only {len(props)} proposals so far"))


if __name__ == "__main__":
    main()
