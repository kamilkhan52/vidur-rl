#!/usr/bin/env python3
"""Tail summary of RL inference-eval runs: one row per <eval_root>/<variant>/*/request_metrics.csv.

Usage: python3 scripts/rl_inference_eval_tail.py [eval_root]
Default root is the 2026-03-28 eval (purged from scratch by 2026-09). Metrics from tail_metrics.py.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_metrics import summarize_run  # noqa: E402

DEFAULT = "/scratch/alpine/kamil@colostate.edu/vidur-rl/evals/postfix_inference_eval_eps005_rerun_20260328_182411"


def main(base):
    print(f"{'variant':22s} {'n':>6} {'med_e2e':>8} {'p99_e2e':>9} {'p99_ttft':>9} {'p99_tpot':>9}")
    for d in sorted(glob.glob(os.path.join(base, "*/"))):
        rm = glob.glob(os.path.join(d, "*", "request_metrics.csv"))
        if not rm:
            continue
        s = summarize_run(rm[0])
        print(f"{os.path.basename(d.rstrip('/')):22s} {s['n_requests']:>6} {s['p50_e2e']:>8.2f} "
              f"{s['p99_e2e']:>9.1f} {s['p99_ttft']:>9.1f} {s['p99_tpot']:>9.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
