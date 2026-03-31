#!/usr/bin/env python3
"""
Aggregate graduation sweep results into a summary CSV.

Reads raw graduation output directories from Alpine and produces a CSV with
per-config, per-scale statistics. This is the first script in the graduation
analysis pipeline -- the other scripts (plot_graduation_progress.py and
plot_graduation_comparisons.py) read from the CSV this script produces.

Usage:
    python scripts/aggregate_graduation.py --input-dir /path/to/graduation/ --output graduation_summary.csv
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCALE_PROGRESSION = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]

CONFIG_DIR_PATTERN = re.compile(
    r"^grad_"
    r"(?P<model>\d+b)-(?P<replicas>\d+R)-(?P<trace>\w+)"
    r"_h(?P<hidden>[\d_]+)"
    r"_r(?P<reward>.+)$"
)

SCALE_DIR_PATTERN = re.compile(
    r"^scale(?P<scale>[\d.]+)(?:_(?P<timestamp>\d{8}_\d{6}))?$"
)

ITER_DIR_PATTERN = re.compile(r"^iter(?P<iter>\d+)$")

INNER_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d+$")


def parse_config_dir(dirname):
    m = CONFIG_DIR_PATTERN.match(dirname)
    if not m:
        return None
    return {
        "config": dirname,
        "model": m.group("model"),
        "replicas": m.group("replicas"),
        "trace": m.group("trace"),
        "hidden": m.group("hidden"),
        "reward": m.group("reward"),
    }


def parse_scale_dir(dirname):
    m = SCALE_DIR_PATTERN.match(dirname)
    if not m:
        return None
    return (float(m.group("scale")), m.group("timestamp"))


def find_latest_run_dir(config_path, scale):
    candidates = []
    for entry in config_path.iterdir():
        if not entry.is_dir():
            continue
        parsed = parse_scale_dir(entry.name)
        if parsed is None:
            continue
        dir_scale, timestamp = parsed
        if abs(dir_scale - scale) > 0.001:
            continue
        ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S") if timestamp else datetime(2000, 1, 1)
        candidates.append((ts, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def find_last_iteration(run_dir):
    """Find the iteration with the highest number, then the latest timestamp within it."""
    best_iter = -1
    best_path = None
    
    # First pass: find highest iteration number
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        m = ITER_DIR_PATTERN.match(entry.name)
        if not m:
            continue
        iter_num = int(m.group("iter"))
        if iter_num > best_iter:
            best_iter = iter_num
    
    if best_iter < 0:
        return None
    
    # Second pass: find latest timestamp within the highest iteration
    iter_dir = run_dir / f"iter{best_iter}"
    if not iter_dir.exists():
        return None
    
    latest_ts = None
    latest_path = None
    
    # Check for direct CSV first
    direct = iter_dir / "request_metrics.csv"
    if direct.exists():
        return (best_iter, direct)
    
    # Find latest timestamped subdirectory
    for entry in iter_dir.iterdir():
        if not entry.is_dir():
            continue
        if INNER_TIMESTAMP_PATTERN.match(entry.name):
            ts_str = entry.name  # e.g., "2026-02-14_22-02-55-491120"
            # Parse: 2026-02-14_22-02-55-491120 -> datetime
            ts = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S-%f")
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_path = entry / "request_metrics.csv"
    
    if latest_path and latest_path.exists():
        return (best_iter, latest_path)
    
    return None


def compute_stats(metrics_path):
    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"  WARNING: Could not read {metrics_path}: {e}", file=sys.stderr)
        return None
    if "request_e2e_time" not in df.columns:
        print(f"  WARNING: 'request_e2e_time' column not found in {metrics_path}", file=sys.stderr)
        return None
    e2e = df["request_e2e_time"].dropna()
    if len(e2e) == 0:
        return None
    stats = {
        "mean_e2e": float(np.mean(e2e)),
        "p99_e2e": float(np.percentile(e2e, 99)),
        "median_e2e": float(np.median(e2e)),
        "min_e2e": float(np.min(e2e)),
        "max_e2e": float(np.max(e2e)),
        "count": len(e2e),
    }
    if "prefill_e2e_time" in df.columns:
        ttft = df["prefill_e2e_time"].dropna()
        if len(ttft) > 0:
            stats["mean_ttft"] = float(np.mean(ttft))
            stats["median_ttft"] = float(np.median(ttft))
            stats["p99_ttft"] = float(np.percentile(ttft, 99))
    return stats


def scan_graduation_dir(input_dir):
    results = []
    config_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("grad_")]
    )
    if not config_dirs:
        print(f"WARNING: No graduation config directories found in {input_dir}", file=sys.stderr)
        return results

    print(f"Found {len(config_dirs)} config directories")

    for config_dir in config_dirs:
        parsed = parse_config_dir(config_dir.name)
        if parsed is None:
            print(f"  Skipping unrecognized dir: {config_dir.name}", file=sys.stderr)
            continue

        available_scales = set()
        for entry in config_dir.iterdir():
            if not entry.is_dir():
                continue
            p = parse_scale_dir(entry.name)
            if p is not None:
                available_scales.add(p[0])

        if not available_scales:
            continue

        min_scale = min(available_scales)
        total_iterations = 0

        for scale in sorted(available_scales, reverse=True):
            run_dir = find_latest_run_dir(config_dir, scale)
            if run_dir is None:
                continue
            iter_result = find_last_iteration(run_dir)
            if iter_result is None:
                continue
            # New format: (iter_num, metrics_path) - already resolved to CSV
            iter_num, metrics_path = iter_result
            total_iterations += iter_num

            if metrics_path is None or not metrics_path.exists():
                print(f"  No request_metrics.csv for {config_dir.name} scale{scale} iter{iter_num}", file=sys.stderr)
                continue
            stats = compute_stats(metrics_path)
            if stats is None:
                continue

            row = {
                **parsed,
                "scale": scale,
                "iter_num": iter_num,
                "num_iterations": total_iterations,
                "min_scale_reached": min_scale,
                "graduated": min_scale < 48.0,
                **stats,
            }
            results.append(row)

        print(f"  {config_dir.name}: {len(available_scales)} scales, min_scale={min_scale}, graduated={'yes' if min_scale < 48.0 else 'no'}")

    return results


def print_summary_table(results):
    if not results:
        print("\nNo results to summarize.")
        return
    df = pd.DataFrame(results)
    summary = (
        df.sort_values("scale")
        .groupby("config")
        .first()
        .reset_index()
        .sort_values(["model", "trace", "replicas", "reward"])
    )
    print("\n" + "=" * 100)
    print("GRADUATION PROGRESS SUMMARY")
    print("=" * 100)
    print(f"{'Config':<50} {'Min Scale':>10} {'Mean E2E':>10} {'P99 E2E':>10} {'Count':>7} {'Grad?':>6}")
    print("-" * 100)
    for _, row in summary.iterrows():
        print(
            f"{row['config']:<50} {row['min_scale_reached']:>10.1f} "
            f"{row['mean_e2e']:>10.3f} {row['p99_e2e']:>10.3f} "
            f"{int(row['count']):>7d} {'YES' if row['graduated'] else 'no':>6}"
        )
    print("-" * 100)
    graduated_count = int(summary["graduated"].sum())
    print(f"Total configs: {len(summary)} | Graduated: {graduated_count} | Stuck at 48: {len(summary) - graduated_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate graduation sweep results into a summary CSV.",
        epilog="Example: python scripts/aggregate_graduation.py --input-dir /scratch/alpine/.../graduation/ --output graduation_summary.csv",
    )
    parser.add_argument("--input-dir", required=True, help="Root directory containing graduation output")
    parser.add_argument("--output", default="graduation_summary.csv", help="Output CSV file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning graduation directory: {input_dir}")
    results = scan_graduation_dir(input_dir)
    if not results:
        print("ERROR: No valid results found.", file=sys.stderr)
        sys.exit(1)

    output_columns = [
        "config", "model", "replicas", "trace", "hidden", "reward",
        "scale", "iter_num", "num_iterations",
        "mean_e2e", "p99_e2e", "median_e2e", "min_e2e", "max_e2e", "count",
        "mean_ttft", "median_ttft", "p99_ttft",
        "min_scale_reached", "graduated",
    ]
    df = pd.DataFrame(results)
    df = df[output_columns].sort_values(["config", "scale"], ascending=[True, False])
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
