#!/usr/bin/env python3
"""Analyze training progress across iterations for a single config."""

import os
import sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python analyze_grad_progress.py <config_dir>")
    sys.exit(1)

config_dir = Path(sys.argv[1])
print(f"Analyzing: {config_dir}")

# Find all scale directories
scale_dirs = sorted([d for d in config_dir.iterdir() if d.name.startswith('scale')])

results = []

for scale_dir in scale_dirs:
    scale_name = scale_dir.name
    # Extract scale value
    scale_val = scale_name.split('_')[0].replace('scale', '')
    
    # Find all iteration directories
    iter_dirs = sorted([d for d in scale_dir.iterdir() if d.name.startswith('iter')], 
                       key=lambda x: int(x.name.replace('iter', '')))
    
    for iter_dir in iter_dirs:
        iter_num = int(iter_dir.name.replace('iter', ''))
        
        # Find timestamp directories
        ts_dirs = sorted([d for d in iter_dir.iterdir() if d.name.startswith('2026')])
        
        for ts_dir in ts_dirs:
            metrics_file = ts_dir / 'request_metrics.csv'
            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    mean_e2e = df['request_e2e_time'].mean()
                    p99_e2e = df['request_e2e_time'].quantile(0.99)
                    count = len(df)
                    results.append({
                        'scale': scale_val,
                        'iteration': iter_num,
                        'mean_e2e': mean_e2e,
                        'p99_e2e': p99_e2e,
                        'count': count,
                        'timestamp': ts_dir.name
                    })
                    break  # Only use first valid timestamp
                except Exception as e:
                    print(f"  Error reading {metrics_file}: {e}")

# Print results
print(f"\n{'Scale':>8} {'Iter':>5} {'Mean E2E':>12} {'P99 E2E':>12} {'Count':>8}")
print("-" * 50)

for r in results:
    print(f"{r['scale']:>8} {r['iteration']:>5} {r['mean_e2e']:>12.4f} {r['p99_e2e']:>12.4f} {r['count']:>8}")

# Save to CSV
import csv
output_csv = config_dir.name + "_training_progress.csv"
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['scale', 'iteration', 'mean_e2e', 'p99_e2e', 'count', 'timestamp'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to: {output_csv}")
