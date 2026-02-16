#!/usr/bin/env python3
"""
Aggregate training effort (total iterations) per scale per config.

Creates a heatmap showing how many total iterations each track has trained
at each scale level. Useful for understanding training distribution.

Usage:
    python scripts/aggregate_training_effort.py --input-dir /path/to/graduation/ --output training_effort.csv
"""

import argparse
import re
import sys
from pathlib import Path

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
    return {
        "scale": float(m.group("scale")),
        "timestamp": m.group("timestamp"),
    }


def count_iterations(scale_dir_path):
    """Count total iteration directories in a scale directory."""
    if not scale_dir_path.is_dir():
        return 0
    
    count = 0
    for item in scale_dir_path.iterdir():
        if ITER_DIR_PATTERN.match(item.name) and item.is_dir():
            count += 1
    return count


def aggregate_training_effort(input_dir, output_file):
    input_path = Path(input_dir)
    results = []
    
    # Get all config directories
    config_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    for config_dir in config_dirs:
        config_info = parse_config_dir(config_dir.name)
        if not config_info:
            print(f"Skipping non-config dir: {config_dir.name}", file=sys.stderr)
            continue
        
        # Track iterations per scale
        scale_iterations = {s: 0 for s in SCALE_PROGRESSION}
        
        # Get all scale directories
        scale_dirs = sorted([d for d in config_dir.iterdir() if d.is_dir()])
        
        for scale_dir in scale_dirs:
            scale_info = parse_scale_dir(scale_dir.name)
            if not scale_info:
                continue
            
            scale = scale_info["scale"]
            # Only count scales in our progression
            if scale not in SCALE_PROGRESSION:
                continue
            
            # Count iterations in this scale directory
            iterations = count_iterations(scale_dir)
            scale_iterations[scale] += iterations
        
        # Build result row
        row = {
            "config": config_dir.name,
            "model": config_info["model"],
            "replicas": config_info["replicas"],
            "trace": config_info["trace"],
            "hidden": config_info["hidden"],
            "reward": config_info["reward"],
        }
        
        # Add iteration counts for each scale
        for scale in SCALE_PROGRESSION:
            row[f"scale_{scale}"] = scale_iterations[scale]
        
        # Add total
        row["total_iterations"] = sum(scale_iterations.values())
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by config for consistent ordering
    df = df.sort_values("config").reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved training effort data to {output_file}")
    print(f"Total configs: {len(df)}")
    print(f"\nSample (first 5 rows):")
    print(df.head())
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total iterations by model:")
    print(df.groupby("model")["total_iterations"].sum())
    print(f"\nTotal iterations by trace:")
    print(df.groupby("trace")["total_iterations"].sum())
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Aggregate training effort per scale per config")
    parser.add_argument("--input-dir", required=True, help="Path to graduation output directory")
    parser.add_argument("--output", default="training_effort.csv", help="Output CSV file")
    args = parser.parse_args()
    
    aggregate_training_effort(args.input_dir, args.output)


if __name__ == "__main__":
    main()
