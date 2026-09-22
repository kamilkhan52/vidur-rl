#!/usr/bin/env python3
"""
Generate individual heatmaps for each trace from Priority 1 baseline data.
Reads from Alpine scratch directory structure.
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Scheduler display order (uppercase - matching Alpine directories)
SCHEDULERS = [
    'RANDOM',
    'ROUND_ROBIN',
    'LOR',
    'LOP',
    'LOP_BINARY',
    'LOP_BATCH_END',
    'LOP_UNCACHED',
    'STICKY_RR',
    'STICKY_LOR',
    'RANKED_STICKY_LOP',
    'TOLERANT_STICKY',
]

SCALES = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5]

def find_latest_run(scale_dir):
    """Find the latest timestamped run directory."""
    run_dirs = []
    if not scale_dir.exists():
        return None
        
    for item in scale_dir.iterdir():
        if not item.is_dir():
            continue
        # Direct timestamp directories (YYYYMMDD_HHMMSS format)
        if item.name[0].isdigit() and len(item.name) >= 15:
            run_dirs.append(item)
    
    if not run_dirs:
        return None
    
    # Return latest by timestamp
    return sorted(run_dirs, key=lambda p: p.name)[-1]

def collect_results(base_dir, trace_name):
    """Collect E2E results from baseline sweep outputs."""
    results = {}
    base_path = Path(base_dir)
    
    print(f"\nScanning {base_path} for {trace_name} results...")
    
    for scheduler in SCHEDULERS:
        sched_dir = base_path / scheduler
        if not sched_dir.exists():
            print(f"  ⚠️  {scheduler} directory not found")
            continue
        
        config_pattern = f"7b-16R-{trace_name}"
        config_dir = sched_dir / config_pattern
        
        if not config_dir.exists():
            print(f"  ⚠️  {scheduler}/{config_pattern} not found")
            continue
        
        for scale in SCALES:
            scale_dir = config_dir / f"scale{scale}"
            if not scale_dir.exists():
                continue
            
            latest_run = find_latest_run(scale_dir)
            if not latest_run:
                continue
            
            metrics_file = latest_run / "request_metrics.csv"
            if not metrics_file.exists():
                continue
            
            try:
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    e2e_times = [float(row['request_e2e_time']) for row in reader]
                    
                if e2e_times:
                    mean_e2e = np.mean(e2e_times)
                    results[(scheduler, scale)] = mean_e2e
                    print(f"  ✓ {scheduler:20s} Scale {scale:4.1f}: {mean_e2e:7.2f}s")
            except Exception as e:
                print(f"  ✗ {scheduler} Scale {scale}: {e}")
    
    return results

def plot_heatmap(results, trace_name, output_path):
    """Generate heatmap: Scheduler (rows) × Scale (columns)."""
    
    # Build matrix
    matrix = []
    for scheduler in SCHEDULERS:
        row = []
        for scale in SCALES:
            value = results.get((scheduler, scale), np.nan)
            row.append(value)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap with green (good) to red (bad)
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Green for low (good), red for high (bad)
        xticklabels=[f'{s:.1f}' for s in SCALES],
        yticklabels=SCHEDULERS,
        cbar_kws={'label': 'Mean E2E Time (s)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
    )
    
    # Invert x-axis (higher QPS on right)
    ax.invert_xaxis()
    
    ax.set_xlabel('Scale (inverted: right = higher QPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scheduler', fontsize=12, fontweight='bold')
    ax.set_title(f'EXP02 Priority 1: {trace_name} Trace\n(7b-16R, Green = Better)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()

def print_summary(results, trace_name):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Summary: {trace_name} Trace")
    print(f"{'='*60}")
    
    # Compute averages per scheduler
    averages = {}
    for scheduler in SCHEDULERS:
        values = [results.get((scheduler, scale), np.nan) for scale in SCALES]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            averages[scheduler] = np.mean(valid_values)
    
    if not averages:
        print("No data available!")
        return
    
    # Sort and display
    sorted_schedulers = sorted(averages.items(), key=lambda x: x[1])
    
    print("\nTop 5 Schedulers (by average E2E):")
    for rank, (scheduler, avg_e2e) in enumerate(sorted_schedulers[:5], 1):
        print(f"  {rank}. {scheduler:25s} {avg_e2e:.2f}s")
    
    print(f"\nWorst 3 Schedulers:")
    for rank, (scheduler, avg_e2e) in enumerate(sorted_schedulers[-3:], 1):
        print(f"  {rank}. {scheduler:25s} {avg_e2e:.2f}s")

def main():
    parser = argparse.ArgumentParser(description='Generate individual trace heatmaps for Priority 1')
    parser.add_argument('--base-dir', 
                        default='/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines',
                        help='Base directory for baseline results')
    parser.add_argument('--output-dir', 
                        default='/scratch/alpine/kamil@colostate.edu/vidur-rl/plots/EXP02_Priority1',
                        help='Output directory for plots')
    parser.add_argument('--traces', nargs='+', default=['Synth', 'Tool', 'Conv'],
                        help='Traces to process')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Traces: {', '.join(args.traces)}")
    
    all_plots = []
    
    for trace in args.traces:
        print(f"\n{'='*60}")
        print(f"Processing {trace} Trace")
        print(f"{'='*60}")
        
        results = collect_results(args.base_dir, trace)
        
        if not results:
            print(f"❌ No results found for {trace} trace!")
            continue
        
        print(f"\n✓ Collected {len(results)} data points")
        
        # Generate heatmap
        output_path = output_dir / f'heatmap_{trace.lower()}.png'
        plot_heatmap(results, trace, output_path)
        all_plots.append(output_path)
        
        # Print summary
        print_summary(results, trace)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(all_plots)} heatmaps:")
    for p in all_plots:
        print(f"   {p}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
