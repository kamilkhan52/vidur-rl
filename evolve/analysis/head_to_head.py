"""Head-to-head: a policy's P99s against each bar member separately, with a cluster bootstrap over configurations.

bar_eval JSONs store per-scenario ratios to the bar (the per-metric best baseline). The policy's own P99 is
ratio x bar, and its ratio to baseline b is ratio x bar / P99_b, with every baseline read from the cache under the
current code_version and trace hashes.

  python -m evolve.analysis.head_to_head --json evolve/analysis/out/test2_finalists.json --stage test2 \
      --policy tail4_ni.py
"""
import argparse
import json

import numpy as np

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import METRICS, boot, clusters


def short(b: str) -> str:
    return b.rsplit("/", 1)[-1].removesuffix(".py") if b.startswith("policy:") else b


def head_to_head(per_scenario: dict, scenarios, cache: dict, bar=ev.BAR_V2) -> dict:
    by_tag = {sc.tag: sc for sc in scenarios}
    tags = sorted(per_scenario)
    missing = [t for t in tags if t not in by_tag]
    if missing:
        raise KeyError(f"scenarios not in the stage: {missing[:3]}")
    out = {}
    for b in bar:
        res = {}
        for m in METRICS:
            d = []
            for t in tags:
                sc = by_tag[t]
                best = min(cache[ev.baseline_key(x, sc)][f"p99_{m}"] for x in bar)
                d.append(np.log(per_scenario[t][m] * best / cache[ev.baseline_key(b, sc)][f"p99_{m}"]))
            d = np.array(d)
            g, lo, hi = boot(d, clusters(tags))
            res[m] = {"geo": g, "lo": lo, "hi": hi, "wins": int((d < 0).sum()), "n": len(d)}
        out[short(b)] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(ev.STAGES))
    ap.add_argument("--policy", action="append", required=True)
    a = ap.parse_args()
    pol = {}
    for f in a.json:
        pol.update(json.load(open(f))["policies"])
    cache = ev.load_baseline_cache()
    scs = ev.STAGES[a.stage]
    print(f"{a.stage} ({len(scs)} scenarios): policy P99 / baseline P99, geo mean [95% cluster-bootstrap CI] "
          f"and wins out of {len(scs)}")
    for p in a.policy:
        print(f"\n{p.removesuffix('.py')}")
        for b, res in head_to_head(pol[p]["per_scenario"], scs, cache).items():
            print(f"  vs {b:18s} " + " | ".join(
                f"{m} {r['geo']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] {r['wins']}" for m, r in res.items()))


if __name__ == "__main__":
    main()
