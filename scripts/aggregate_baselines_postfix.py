#!/usr/bin/env python3
"""
Aggregate post-fix baseline sweep results into a summary CSV.

Reads raw baseline output directories and produces a CSV with
per-scheduler, per-config, per-scale statistics including both E2E and TTFT.

Usage:
    python scripts/aggregate_baselines_postfix.py \
        --input-dir /scratch/alpine/.../baselines_postfix/priority2/ \
        --output priority2_postfix_all.csv
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_PATTERN = re.compile(r"^(?P<model>\d+b)-(?P<replicas>\d+R)-(?P<trace>\w+)$")
SCALE_PATTERN = re.compile(r"^scale(?P<scale>[\d.]+)$")
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
INNER_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+$")


def find_latest_run_csv(run_dir):
    """Find the request_metrics.csv from the latest run inside a timestamp dir."""
    # Each timestamp dir contains run1/, run2/, ... with timestamped subdirs
    best_run = None
    best_run_num = -1
    for entry in run_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("run"):
            continue
        try:
            run_num = int(entry.name.replace("run", ""))
        except ValueError:
            continue
        if run_num > best_run_num:
            best_run_num = run_num
            best_run = entry

    if best_run is None:
        return None

    # Inside runN/, find latest timestamped subdir
    latest_ts_dir = None
    for entry in best_run.iterdir():
        if entry.is_dir() and INNER_TIMESTAMP_PATTERN.match(entry.name):
            if latest_ts_dir is None or entry.name > latest_ts_dir.name:
                latest_ts_dir = entry

    if latest_ts_dir is None:
        return None

    csv_path = latest_ts_dir / "request_metrics.csv"
    return csv_path if csv_path.exists() else None


def compute_stats(metrics_path):
    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"  WARNING: Could not read {metrics_path}: {e}", file=sys.stderr)
        return None
    if "request_e2e_time" not in df.columns:
        return None
    e2e = df["request_e2e_time"].dropna()
    if len(e2e) == 0:
        return None
    stats = {
        "count": len(e2e),
        "mean_e2e": float(np.mean(e2e)),
        "median_e2e": float(np.median(e2e)),
        "p95_e2e": float(np.percentile(e2e, 95)),
        "p99_e2e": float(np.percentile(e2e, 99)),
    }
    if "prefill_e2e_time" in df.columns:
        ttft = df["prefill_e2e_time"].dropna()
        if len(ttft) > 0:
            stats["mean_ttft"] = float(np.mean(ttft))
            stats["median_ttft"] = float(np.median(ttft))
            stats["p95_ttft"] = float(np.percentile(ttft, 95))
            stats["p99_ttft"] = float(np.percentile(ttft, 99))
    return stats


def scan_baselines(input_dir):
    results = []
    for sched_dir in sorted(input_dir.iterdir()):
        if not sched_dir.is_dir():
            continue
        scheduler = sched_dir.name

        for config_dir in sorted(sched_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            m = CONFIG_PATTERN.match(config_dir.name)
            if not m:
                continue
            model, replicas, trace = m.group("model"), m.group("replicas"), m.group("trace")

            for scale_dir in sorted(config_dir.iterdir()):
                if not scale_dir.is_dir():
                    continue
                sm = SCALE_PATTERN.match(scale_dir.name)
                if not sm:
                    continue
                scale = float(sm.group("scale"))

                # Find latest timestamp dir
                ts_dirs = sorted(
                    [d for d in scale_dir.iterdir() if d.is_dir() and TIMESTAMP_PATTERN.match(d.name)],
                    key=lambda d: d.name
                )
                if not ts_dirs:
                    continue
                latest_ts = ts_dirs[-1]

                csv_path = find_latest_run_csv(latest_ts)
                if csv_path is None:
                    continue

                stats = compute_stats(csv_path)
                if stats is None:
                    continue

                results.append({
                    "scheduler": scheduler,
                    "model": model,
                    "replicas": replicas,
                    "trace": trace,
                    "scale": scale,
                    "run_id": latest_ts.name,
                    **stats,
                    "file_path": str(csv_path),
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate post-fix baseline results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="priority2_postfix_all.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: {input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning baselines in: {input_dir}")
    results = scan_baselines(input_dir)
    if not results:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(results)
    df = df.sort_values(["scheduler", "model", "replicas", "trace", "scale"])
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

    # Summary
    for sched in sorted(df["scheduler"].unique()):
        sdf = df[df["scheduler"] == sched]
        print(f"  {sched}: {len(sdf)} runs")


if __name__ == "__main__":
    main()
