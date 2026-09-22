#!/usr/bin/env python3
"""EXP09 tail-latency leverage re-cut.

For each latency metric present in the per-run CSV (e2e, ttft, tpot), compares
scheduler leverage on the MEAN vs the P99, and reports whether the best/worst
scheduler ranking *changes* between mean and tail (the mean-vs-tail tradeoff
that motivates a tail-aware RL policy).

Auto-detects metrics: needs columns mean_<m> and p99_<m>. Works on the merged
enhanced CSV (e2e only) and on the fuller tail CSV (e2e/ttft/tpot) once raw
per-request data has been re-cut on Alpine.

Usage: python3 exp09_tail_leverage.py --csv <per_run_csv> [--out <dir>]
"""
import argparse, csv, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_metrics import is_missing, percentile  # noqa: E402
from collections import defaultdict, Counter

KEY = ("scheduler", "model", "replicas", "trace", "scale")
FIXED = ("model", "replicas", "trace", "scale")


pctl = percentile


def sli(items, metric):
    # Before 2026-09-22 this filtered with `not in (None, "", float("nan"))`, which
    # never excludes NaN (a fresh NaN compares unequal to every NaN).
    vals = [(s, m[metric]) for s, m in items if not is_missing(m.get(metric))]
    worst = max(vals, key=lambda x: x[1]); best = min(vals, key=lambda x: x[1])
    return (worst[1] - best[1]) / worst[1] if worst[1] else 0.0, worst[0], best[0]


def completion_mismatch(items, tol=0.0):
    """Relative spread of completed-request counts across schedulers in one scenario."""
    ns = [m["n_requests"] for _, m in items if not is_missing(m.get("n_requests"))]
    if not ns or max(ns) == 0:
        return 0.0
    return (max(ns) - min(ns)) / max(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out")
    ap.add_argument("--max-completion-gap", type=float, default=0.0,
                    help="flag scenarios whose completed-request counts differ by more than this fraction")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    metrics = [m for m in ("e2e", "ttft", "tpot")
               if f"mean_{m}" in rows[0] and f"p99_{m}" in rows[0]]

    # scheduler-scenario means (average repeats)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = tuple(r[x] for x in KEY)
        try:
            agg[k]["n_requests"].append(float(r["n_requests"]))
        except (ValueError, KeyError):
            pass
        for m in metrics:
            for stat in (f"mean_{m}", f"p99_{m}"):
                try:
                    agg[k][stat].append(float(r[stat]))
                except (ValueError, KeyError):
                    pass
    ss = {k: {s: sum(v) / len(v) for s, v in d.items() if v} for k, d in agg.items()}
    fixed = defaultdict(list)
    for k, v in ss.items():
        fixed[tuple(k[KEY.index(x)] for x in FIXED)].append((k[0], v))

    out_lines = []
    # Tail statistics over completed requests only are not comparable when the
    # schedulers completed different request sets (e.g. the pre-2026-09-21 runs,
    # where oversize requests wedged replicas). Flag such scenarios loudly.
    bad = {fk: completion_mismatch(items) for fk, items in fixed.items()}
    bad = {fk: v for fk, v in bad.items() if v > args.max_completion_gap}
    if bad:
        out_lines.append(f"WARNING: {len(bad)}/{len(fixed)} scenarios have completed-request counts that "
                         f"differ across schedulers by more than {args.max_completion_gap:.1%}; their "
                         f"tail comparisons are biased:")
        for fk, v in sorted(bad.items(), key=lambda x: -x[1]):
            out_lines.append(f"    {'-'.join(fk):20s} completion gap {v:.1%}")
    for m in metrics:
        sli_mean, sli_p99, reversal = [], [], 0
        bestm, bestp, worstm, worstp = Counter(), Counter(), Counter(), Counter()
        per = []
        for fk, items in fixed.items():
            sm, wm, bm = sli(items, f"mean_{m}")
            sp, wp, bp = sli(items, f"p99_{m}")
            sli_mean.append(sm); sli_p99.append(sp)
            bestm[bm] += 1; bestp[bp] += 1; worstm[wm] += 1; worstp[wp] += 1
            if bm != bp:
                reversal += 1
            per.append((fk, sm, sp, wm, bm, wp, bp))
        n = len(per)
        L = out_lines.append
        L(f"\n================ METRIC: {m.upper()} ({n} fixed scenarios) ================")
        L(f"  median SLI_mean={pctl(sli_mean,.5):.3f}  median SLI_p99={pctl(sli_p99,.5):.3f}")
        L(f"  p90    SLI_mean={pctl(sli_mean,.9):.3f}  p90    SLI_p99={pctl(sli_p99,.9):.3f}")
        L(f"  scenarios SLI_p99>SLI_mean: {sum(1 for a,b in zip(sli_mean,sli_p99) if b>a)}/{n}")
        L(f"  best-scheduler CHANGES mean->p99 in {reversal}/{n} scenarios")
        L(f"  BEST  | mean:{dict(bestm)}  p99:{dict(bestp)}")
        L(f"  WORST | mean:{dict(worstm)} p99:{dict(worstp)}")
        for fk, sm, sp, wm, bm, wp, bp in sorted(per, key=lambda r: -r[2]):
            flag = "  <-rank flip" if bm != bp else ""
            L(f"    {'-'.join(fk):20s} SLI_mean={sm:.3f} SLI_p99={sp:.3f}"
              f"  mean[w={wm},b={bm}] p99[w={wp},b={bp}]{flag}")
    txt = "\n".join(out_lines) + "\n"
    print(txt)
    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        open(os.path.join(args.out, "exp09_tail_leverage.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
