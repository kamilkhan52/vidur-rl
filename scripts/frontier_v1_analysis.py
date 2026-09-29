#!/usr/bin/env python3
"""
Frontier v1 analysis (job 33083463): is there tail leverage left for a learned router?

Input : run root (…/frontier_v1/runs) laid out p<P>/<SCHED>/7b-<NR>-<Trace>/scale<X>/<ts>/run1/<vidur ts>/
Output: per-run CSV and a text report with, per scenario and metric:
  - each scheduler's P99 (mean over perturbations, and min–max),
  - the tail-best and mean-best scheduler and whether they differ (mean-vs-tail conflict),
  - leverage SLI = (worst − best) / worst on the P99,
  - how often the tail-best scheduler wins across the paired perturbations (stability),
and a completion check (every run must finish every admitted request).

Usage: python3 scripts/frontier_v1_analysis.py --root <runs dir> --out <dir>
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_metrics import SUMMARY_FIELDS, summarize_run  # noqa: E402

# Requests per trace after the 7b KV-capacity filter (122,752 tokens).
EXPECTED = {"Synth": 3982, "Conv": 12020, "Tool": 23599}
METRICS = ("e2e", "ttft", "tpot")


def discover(root: Path):
    for f in sorted(root.rglob("request_metrics.csv")):
        parts = f.parts
        try:
            pi = next(i for i, t in enumerate(parts) if re.fullmatch(r"p\d+", t))
            ci = next(i for i, t in enumerate(parts) if re.fullmatch(r"\d+b-\d+R-[A-Za-z]+", t))
        except StopIteration:
            continue
        model, repl, trace = parts[ci].split("-")
        scale = parts[ci + 1][len("scale"):]
        yield {"perturbation": int(parts[pi][1:]), "scheduler": parts[ci - 1], "model": model,
               "replicas": repl, "trace": trace, "scale": scale, "path": str(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for r in discover(Path(a.root)):
        r.update(summarize_run(r["path"]))
        runs.append(r)
    keys = ["perturbation", "scheduler", "model", "replicas", "trace", "scale", "path"]
    with open(out / "frontier_v1_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + SUMMARY_FIELDS)
        w.writeheader()
        for r in runs:
            w.writerow({k: r[k] for k in keys + SUMMARY_FIELDS})

    L = []
    incomplete = [r for r in runs if r["n_requests"] != EXPECTED[r["trace"]]]
    L.append(f"runs found: {len(runs)}")
    L.append(f"completion check: {len(runs) - len(incomplete)} complete, {len(incomplete)} incomplete")
    for r in incomplete[:20]:
        L.append(f"   INCOMPLETE {r['scheduler']} {r['replicas']}-{r['trace']}-{r['scale']} p{r['perturbation']}: "
                 f"{r['n_requests']}/{EXPECTED[r['trace']]}")

    # scenario -> scheduler -> perturbation -> run
    S = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        S[(r["replicas"], r["trace"], r["scale"])][r["scheduler"]][r["perturbation"]] = r

    summary = {m: {"conflict": 0, "sli": [], "best": defaultdict(int), "stable": 0, "n": 0} for m in METRICS}
    for sk in sorted(S, key=lambda k: (k[1], k[0], k[2])):
        scheds = S[sk]
        perts = sorted(set.intersection(*[set(v) for v in scheds.values()]))
        if not perts:
            continue
        L.append(f"\n=== {'-'.join(sk)}  (perturbations {perts})")
        mean_e2e = {s: sum(scheds[s][p]["mean_e2e"] for p in perts) / len(perts) for s in scheds}
        mean_best = min(mean_e2e, key=mean_e2e.get)
        for m in METRICS:
            p99 = {s: [scheds[s][p][f"p99_{m}"] for p in perts] for s in scheds}
            avg = {s: sum(v) / len(v) for s, v in p99.items()}
            best, worst = min(avg, key=avg.get), max(avg, key=avg.get)
            wins = sum(1 for i in range(len(perts)) if min(p99, key=lambda s: p99[s][i]) == best)
            sli = (avg[worst] - avg[best]) / avg[worst] if avg[worst] else 0.0
            flag = "  <-- mean-best differs" if best != mean_best else ""
            L.append(f"  P99 {m:4s}: " + "  ".join(f"{s}={avg[s]:.3g}[{min(p99[s]):.3g},{max(p99[s]):.3g}]"
                                                  for s in sorted(avg, key=avg.get))
                     + f"  | SLI {sli:.2f}; best wins {wins}/{len(perts)} perturbations{flag}")
            st = summary[m]
            st["n"] += 1
            st["sli"].append(sli)
            st["best"][best] += 1
            st["conflict"] += best != mean_best
            st["stable"] += wins == len(perts)
        L.append(f"  mean E2E best: {mean_best} ({mean_e2e[mean_best]:.3g})")

    L.append("\n=== SUMMARY over scenarios")
    for m in METRICS:
        st = summary[m]
        if not st["n"]:
            continue
        sli = sorted(st["sli"])
        L.append(f"  {m.upper():4s}: tail-best counts {dict(st['best'])}; tail-best != mean-best in "
                 f"{st['conflict']}/{st['n']}; tail-best wins every perturbation in {st['stable']}/{st['n']}; "
                 f"median SLI {sli[len(sli) // 2]:.2f}")
    text = "\n".join(L) + "\n"
    (out / "frontier_v1_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
