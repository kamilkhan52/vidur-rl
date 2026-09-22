#!/usr/bin/env python3
"""
Generate heatmaps for Priority 1 baseline comparison across all 3 traces.
Shows Scheduler × Scale heatmap with green (better) → red (worse) color scheme.
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Scheduler display order (grouped by category)
SCHEDULER_ORDER = [
    # Simple
    'random',
    'round_robin',
    # Lookup (cache-aware)
    'lor',
    'lop',
    'lop_binary',
    'lop_batch_end',
    'lop_uncached',
    # Sticky (session affinity)
    'sticky_round_robin',
    'sticky_lor',
    'ranked_sticky_lop_uncached',
    'tolerant_sticky_lop_uncached',
]

SCHEDULER_LABELS = {
    'random': 'RANDOM',
    'round_robin': 'ROUND_ROBIN',
    'lor': 'LOR',
    'lop': 'LOP',
    'lop_binary': 'LOP_BINARY',
    'lop_batch_end': 'LOP_BATCH_END',
    'lop_uncached': 'LOP_UNCACHED',
    'sticky_round_robin': 'STICKY_RR',
    'sticky_lor': 'STICKY_LOR',
    'ranked_sticky_lop_uncached': 'RANKED_STICKY_LOP',
    'tolerant_sticky_lop_uncached': 'TOLERANT_STICKY',
}

SCALES = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5]

def find_latest_run(scale_dir):
    """Find the latest timestamped run directory."""
    run_dirs = []
    for run in scale_dir.iterdir():
        if not run.is_dir():
            continue
        # Look for YYYY-MM-DD_HH-MM-SS-NNNNNN directories inside runN/
        for timestamp_dir in run.iterdir():
            if timestamp_dir.is_dir() and timestamp_dir.name.startswith('2026'):
                run_dirs.append(timestamp_dir)
    
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) and return latest
    return sorted(run_dirs, key=lambda p: p.name)[-1]

def collect_results(base_dir, trace_name):
    """Collect E2E results from baseline sweep outputs."""
    results = {}
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"ERROR: Base directory not found: {base_dir}")
        return results
    
    for sched_dir in base_path.iterdir():
        if not sched_dir.is_dir():
            continue
        
        scheduler = sched_dir.name
        
        # Look for 7b-16R-{trace_name} config
        config_pattern = f"7b-16R-{trace_name}"
        config_dir = sched_dir / config_pattern
        
        if not config_dir.exists():
            continue
        
        for scale in SCALES:
            scale_dir = config_dir / f"scale{scale}"
            if not scale_dir.exists():
                continue
            
            # Find latest run
            latest_run = find_latest_run(scale_dir)
            if not latest_run:
                continue
            
            metrics_file = latest_run / "request_metrics.csv"
            if not metrics_file.exists():
                continue
            
            # Parse metrics
            try:
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    e2e_times = [float(row['request_e2e_time']) for row in reader]
                    
                if e2e_times:
                    mean_e2e = np.mean(e2e_times)
                    results[(scheduler, scale)] = mean_e2e
                    print(f"  {scheduler:30s} Scale {scale:4.1f}: {mean_e2e:7.2f}s")
            except Exception as e:
                print(f"  WARNING: Failed to parse {metrics_file}: {e}")
    
    return results

def plot_heatmap(results, trace_name, output_dir):
    """Generate heatmap: Scheduler (rows) × Scale (columns)."""
    
    # Build matrix
    matrix = []
    row_labels = []
    
    for scheduler in SCHEDULER_ORDER:
        row = []
        for scale in SCALES:
            value = results.get((scheduler, scale), np.nan)
            row.append(value)
        matrix.append(row)
        row_labels.append(SCHEDULER_LABELS.get(scheduler, scheduler.upper()))
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use green (good) to red (bad) colormap
    # Lower E2E is better, so we want lower values to be green
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Reversed: green for low, red for high
        xticklabels=[f'{s:.1f}' for s in SCALES],
        yticklabels=row_labels,
        cbar_kws={'label': 'Mean E2E Time (s)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
    )
    
    # Invert x-axis (higher QPS on right)
    ax.invert_xaxis()
    
    ax.set_xlabel('Scale (inverted: right = higher QPS)', fontsize=12)
    ax.set_ylabel('Scheduler', fontsize=12)
    ax.set_title(f'Baseline Scheduler Comparison: {trace_name} Trace\n(7b-16R, Green = Better)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f'baseline_{trace_name.lower()}_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    plt.close()
    
    return output_path

def print_summary(results, trace_name):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Summary: {trace_name} Trace")
    print(f"{'='*60}")
    
    # Compute averages per scheduler
    averages = {}
    for scheduler in SCHEDULER_ORDER:
        values = [results.get((scheduler, scale), np.nan) for scale in SCALES]
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            averages[scheduler] = np.mean(valid_values)
    
    # Sort and display top 5
    sorted_schedulers = sorted(averages.items(), key=lambda x: x[1])
    
    print("\nTop 5 Schedulers (by average E2E):")
    for rank, (scheduler, avg_e2e) in enumerate(sorted_schedulers[:5], 1):
        label = SCHEDULER_LABELS.get(scheduler, scheduler.upper())
        print(f"  {rank}. {label:25s} {avg_e2e:.2f}s")
    
    print(f"\nWorst 3 Schedulers:")
    for rank, (scheduler, avg_e2e) in enumerate(sorted_schedulers[-3:], 1):
        label = SCHEDULER_LABELS.get(scheduler, scheduler.upper())
        print(f"  {rank}. {label:25s} {avg_e2e:.2f}s")

def main():
    parser = argparse.ArgumentParser(description='Generate Priority 1 baseline heatmaps')
    parser.add_argument('--base-dir', default='/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines',
                        help='Base directory for baseline results')
    parser.add_argument('--output-dir', default='plots/EXP02_Baselines',
                        help='Output directory for plots')
    parser.add_argument('--traces', nargs='+', default=['Synth', 'Tool', 'Conv'],
                        help='Traces to plot (default: all)')
    
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
            print(f"WARNING: No results found for {trace} trace!")
            continue
        
        print(f"\nCollected {len(results)} data points")
        
        # Generate heatmap
        plot_path = plot_heatmap(results, trace, output_dir)
        all_plots.append(plot_path)
        
        # Print summary
        print_summary(results, trace)
    
    print(f"\n{'='*60}")
    print(f"✅ Generated {len(all_plots)} heatmaps:")
    for p in all_plots:
        print(f"   - {p}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
