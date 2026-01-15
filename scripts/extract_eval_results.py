#!/usr/bin/env python3
"""Extract results from completed evaluation."""

import pandas as pd
from pathlib import Path

results_dir = Path("results/eval_continuous_latest/simulations")

results = []

for scheduler_dir in results_dir.glob("*"):
    if not scheduler_dir.is_dir():
        continue
    
    scheduler_name = scheduler_dir.name.split('_')[0]
    
    # Find latest timestamped directory
    timestamped_dirs = sorted([d for d in scheduler_dir.glob("20*") if d.is_dir()])
    if not timestamped_dirs:
        continue
    
    latest_dir = timestamped_dirs[-1]
    metrics_file = latest_dir / "request_metrics.csv"
    
    if not metrics_file.exists():
        continue
    
    try:
        df = pd.read_csv(metrics_file)
        
        result = {
            'scheduler': scheduler_name,
            'mean_e2e': df['request_e2e_time'].mean(),
            'median_e2e': df['request_e2e_time'].median(),
            'p95_e2e': df['request_e2e_time'].quantile(0.95),
            'num_requests': len(df),
        }
        results.append(result)
        
    except Exception as e:
        print(f"Failed to extract metrics from {scheduler_name}: {e}")

# Display results
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"{'Scheduler':<20} {'Mean E2E':<12} {'Median E2E':<12} {'P95 E2E':<12}")
print("-"*70)

for r in sorted(results, key=lambda x: x['mean_e2e']):
    print(f"{r['scheduler']:<20} {r['mean_e2e']:>10.3f}s {r['median_e2e']:>10.3f}s {r['p95_e2e']:>10.3f}s")

print("="*70)
