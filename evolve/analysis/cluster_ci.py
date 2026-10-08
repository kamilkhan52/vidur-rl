"""Paired policy comparisons and ratios-to-bar with a cluster bootstrap over scenario configurations.

Scenarios that differ only in their arrival draw (the -p<k> suffix) share a configuration and are not
independent, so the bootstrap resamples configurations, then all their draws. Reads bar_eval JSON outputs.

  python -m evolve.analysis.cluster_ci --json evolve/analysis/out/test2_finalists.json \
      --vs-bar tail4_ni.py --pair tail4_ni.py:claude_r1_76be082eb0f5.py
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np

METRICS = ("e2e", "ttft", "tpot", "tbt")


def clusters(tags):
    groups = defaultdict(list)
    for i, t in enumerate(tags):
        groups[re.sub(r"-p\d+", "", t)].append(i)
    return list(groups.values())


def boot(d, groups, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        means[b] = d[idx].mean()
    return np.exp(d.mean()), np.exp(np.percentile(means, 2.5)), np.exp(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True)
    ap.add_argument("--vs-bar", action="append", default=[], help="policy: geo-mean ratio to the bar with CI")
    ap.add_argument("--pair", action="append", default=[], help="a:b -> ratio a/b per metric with CI")
    a = ap.parse_args()
    pol = {}
    for f in a.json:
        pol.update(json.load(open(f))["policies"])

    def logs(name, m):
        s = pol[name]["per_scenario"]
        tags = sorted(s)
        return tags, np.array([np.log(s[t][m]) for t in tags])

    for name in a.vs_bar:
        parts = []
        for m in METRICS:
            tags, d = logs(name, m)
            g, lo, hi = boot(d, clusters(tags))
            parts.append(f"{m} {g:.3f} [{lo:.3f},{hi:.3f}] {int((d < 0).sum())}/{len(d)}")
        print(f"{name} vs bar ({len(clusters(tags))} configs): " + " | ".join(parts))
    for spec in a.pair:
        x, y = spec.split(":")
        parts = []
        for m in METRICS:
            tags, dx = logs(x, m)
            tags_y, dy = logs(y, m)
            assert tags == tags_y
            d = dx - dy
            g, lo, hi = boot(d, clusters(tags))
            parts.append(f"{m} {g:.3f} [{lo:.3f},{hi:.3f}] {int((d < 0).sum())}/{len(d)}")
        print(f"{x} / {y}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
