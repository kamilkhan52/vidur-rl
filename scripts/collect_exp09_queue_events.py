#!/usr/bin/env python3
"""
Collect and merge EXP09 queue diagnostics CSV files from many run directories.

Typical input layout:
  <runs_root>/**/metrics/queue_diag_events.csv

Output:
  A single merged CSV suitable for analyze_exp09_queue_diagnostics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def find_event_files(root: Path, pattern: str) -> List[Path]:
    return sorted([p for p in root.rglob(pattern) if p.is_file()])


def merge_event_files(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"WARN: skipping unreadable file: {p} ({e})")
            continue

        # Keep provenance columns for debugging and post-hoc filtering
        df["source_file"] = str(p)
        df["source_run_dir"] = str(p.parent.parent)  # .../<run>/metrics/queue_diag_events.csv
        if "run_id" not in df.columns or df["run_id"].isna().all():
            df["run_id"] = p.parent.parent.name

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    # Stable row order for reproducibility
    sort_cols = [c for c in ["scheduler", "model", "replicas", "trace", "scale", "run_id", "event_index"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Root directory containing EXP09 run outputs")
    ap.add_argument(
        "--pattern",
        default="queue_diag_events.csv",
        help="Filename pattern used with recursive glob (default: queue_diag_events.csv)",
    )
    ap.add_argument("--output-csv", required=True, help="Path to merged output CSV")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    paths = find_event_files(input_dir, args.pattern)
    print(f"Found {len(paths)} files matching '{args.pattern}' under {input_dir}")

    merged = merge_event_files(paths)
    if merged.empty:
        raise SystemExit("No readable queue diagnostics CSV files found.")

    merged.to_csv(output_csv, index=False)

    print(f"Wrote merged CSV: {output_csv}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
