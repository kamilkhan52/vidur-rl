#!/usr/bin/env python3
"""Per-run tail metrics for a sweep laid out as .../<SCHED>/<model>-<NR>-<Trace>/scale<X>/...

Usage: python3 scripts/exp09_tail_extract.py <file listing run dirs> > runs.csv

Each listed directory must contain request_metrics.csv. Metric definitions live
in scripts/tail_metrics.py (TPOT fixed on 2026-09-22: divides by n_decode - 1).
Runs that fail to parse are reported on stderr by path, not silently dropped.
Works for any model size (the old EXP09 version only matched 7b paths).
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tail_metrics import SUMMARY_FIELDS, summarize_run  # noqa: E402

KEY_FIELDS = ["scheduler", "model", "replicas", "trace", "scale", "run_path"]


def parse_run_path(p):
    parts = p.rstrip("/").split("/")
    ci = next(i for i, t in enumerate(parts) if re.match(r"^\d+b-\d+R-[A-Za-z]+$", t))
    model, repl, trace = parts[ci].split("-")
    sched = parts[ci - 1]
    scale = next(t for t in parts if t.startswith("scale"))[5:]
    return sched, model, repl, trace, scale


def main(listing):
    w = csv.writer(sys.stdout)
    w.writerow(KEY_FIELDS + SUMMARY_FIELDS)
    failed = []
    for p in (l.strip() for l in open(listing) if l.strip()):
        try:
            key = parse_run_path(p)
            s = summarize_run(os.path.join(p, "request_metrics.csv"))
            if not s["n_requests"]:
                raise ValueError("empty request_metrics.csv")
            w.writerow(list(key) + [p] + [s[f] for f in SUMMARY_FIELDS])
        except Exception as e:  # report, do not hide
            failed.append((p, repr(e)))
    for p, e in failed:
        sys.stderr.write(f"FAILED {p}: {e}\n")
    sys.stderr.write(f"failed={len(failed)}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
