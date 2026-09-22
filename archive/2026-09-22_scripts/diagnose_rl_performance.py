#!/usr/bin/env python3
"""
Diagnose RL model performance issues.

Checks:
1. Model file structure and weights
2. Action distribution during inference
3. State representation quality
4. Comparison with baseline policies
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path


def load_model(model_path: str):
    """Load and inspect RL model checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    print(f"\nQ-Network structure:")
    q_net_key = 'q_network_state_dict' if 'q_network_state_dict' in checkpoint else 'q_network'
    for name, param in checkpoint[q_net_key].items():
        print(f"  {name}: {param.shape}")
    
    # Check if weights are all zeros or random
    first_layer_key = list(checkpoint[q_net_key].keys())[0]
    first_layer = checkpoint[q_net_key][first_layer_key]
    print(f"\nFirst layer ({first_layer_key}) weight statistics:")
    print(f"  Mean: {first_layer.mean():.6f}")
    print(f"  Std: {first_layer.std():.6f}")
    print(f"  Min: {first_layer.min():.6f}")
    print(f"  Max: {first_layer.max():.6f}")
    
    # Check if all weights are near initialization (might indicate no training)
    if abs(first_layer.mean()) < 0.001 and abs(first_layer.std() - 0.05) < 0.01:
        print(f"  ⚠️  WARNING: Weights look like random initialization!")
    else:
        print(f"  ✅ Weights appear to be trained")
    
    return checkpoint


def analyze_metrics(metrics_path: str):
    """Analyze request metrics to understand routing behavior."""
    print(f"\nAnalyzing metrics from: {metrics_path}")
    df = pd.read_csv(metrics_path)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Total requests: {len(df)}")
    print(f"  Mean E2E: {df['request_e2e_time'].mean():.3f}s")
    print(f"  Median E2E: {df['request_e2e_time'].median():.3f}s")
    print(f"  P99 E2E: {df['request_e2e_time'].quantile(0.99):.3f}s")
    
    print(f"\n📈 Load Distribution:")
    replica_counts = df['replica'].value_counts().sort_index()
    for replica, count in replica_counts.items():
        pct = 100.0 * count / len(df)
        print(f"  Replica {replica}: {count:5d} ({pct:5.1f}%)")
    
    # Calculate load imbalance
    max_load = replica_counts.max()
    min_load = replica_counts.min()
    imbalance = (max_load - min_load) / len(df)
    print(f"\n  Load imbalance: {imbalance:.4f}")
    
    if imbalance > 0.3:
        print(f"  ❌ SEVERE IMBALANCE - Model is not load balancing!")
    elif imbalance > 0.1:
        print(f"  ⚠️  Moderate imbalance - Room for improvement")
    else:
        print(f"  ✅ Good load balance")
    
    # Check if one replica is dominating
    max_pct = 100.0 * max_load / len(df)
    if max_pct > 60:
        print(f"  ❌ One replica handling {max_pct:.1f}% of load!")
        print(f"  → Model likely learned to always pick same action")


def compare_schedulers(results_dir: str):
    """Compare RL with other schedulers."""
    results_dir = Path(results_dir)
    
    print(f"\n🔍 Comparing schedulers in: {results_dir}")
    
    schedulers = {}
    for sim_dir in results_dir.glob("simulations/*/2026-*"):
        scheduler_name = sim_dir.parent.name.split('_')[0]
        metrics_file = sim_dir / "request_metrics.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            schedulers[scheduler_name] = {
                'mean_e2e': df['request_e2e_time'].mean(),
                'median_e2e': df['request_e2e_time'].median(),
                'p99_e2e': df['request_e2e_time'].quantile(0.99),
            }
    
    if not schedulers:
        print("  No scheduler results found")
        return
    
    print(f"\n📊 Scheduler Comparison:")
    print(f"{'Scheduler':<20} {'Mean E2E':>12} {'Median E2E':>12} {'P99 E2E':>12}")
    print("="*60)
    
    for name, metrics in sorted(schedulers.items(), key=lambda x: x[1]['mean_e2e']):
        print(f"{name:<20} {metrics['mean_e2e']:>10.3f}s {metrics['median_e2e']:>10.3f}s {metrics['p99_e2e']:>10.3f}s")
    
    # Check if RL is worst
    rl_mean = schedulers.get('rl', {}).get('mean_e2e', 0)
    if rl_mean > 0:
        worst_mean = max(s['mean_e2e'] for s in schedulers.values())
        if rl_mean == worst_mean:
            print(f"\n❌ RL scheduler is WORST performer!")
            best_name = min(schedulers.items(), key=lambda x: x[1]['mean_e2e'])[0]
            best_mean = schedulers[best_name]['mean_e2e']
            ratio = rl_mean / best_mean
            print(f"   {ratio:.1f}x worse than {best_name}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose RL model performance")
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to RL model checkpoint'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        help='Path to request_metrics.csv from RL run'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Path to evaluation results directory'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  RL Model Performance Diagnostics")
    print("="*80)
    
    if args.model:
        load_model(args.model)
    
    if args.metrics:
        analyze_metrics(args.metrics)
    
    if args.results_dir:
        compare_schedulers(args.results_dir)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
