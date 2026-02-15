#!/usr/bin/env python3
"""
Generate individual trace heatmaps from aggregated CSV data.
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def load_csv(csv_path):
    """Load aggregated results from CSV."""
    results = {}
    
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) != 6:
                continue
            scheduler, model, replicas, trace, scale, e2e = row
            
            # Only 7b-16R configs
            if model != '7b' or replicas != '16R':
                continue
            
            key = (scheduler, trace, float(scale))
            results[key] = float(e2e)
    
    return results

def plot_heatmap(results, trace_name, output_path):
    """Generate heatmap for one trace."""
    
    # Build matrix
    matrix = []
    for scheduler in SCHEDULERS:
        row = []
        for scale in SCALES:
            value = results.get((scheduler, trace_name, scale), np.nan)
            row.append(value)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Find best (minimum) value in each column
    best_per_scale = {}
    for col_idx, scale in enumerate(SCALES):
        col_values = matrix[:, col_idx]
        if not np.all(np.isnan(col_values)):
            min_idx = np.nanargmin(col_values)
            best_per_scale[scale] = (min_idx, SCHEDULERS[min_idx], col_values[min_idx])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Green = good (low), red = bad (high)
        xticklabels=[f'{s:.1f}' for s in SCALES],
        yticklabels=SCHEDULERS,
        cbar_kws={'label': 'Mean E2E Time (s)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
        annot_kws={'fontsize': 9},
    )
    
    # Add bold rectangles around best values
    for col_idx, scale in enumerate(SCALES):
        if scale in best_per_scale:
            row_idx, scheduler, value = best_per_scale[scale]
            # Draw thick black rectangle around the cell
            rect = plt.Rectangle((col_idx, row_idx), 1, 1, 
                                fill=False, edgecolor='black', linewidth=3)
            ax.add_patch(rect)
    
    # NOTE: Do NOT call ax.invert_xaxis() here!
    # seaborn heatmap already shows columns left-to-right as they appear in data.
    # We want: left=48.0 (low QPS), right=1.5 (high QPS)
    # SCALES list is [48.0, 24.0, 12.0, 6.0, 3.0, 1.5] so this is already correct!
    
    ax.set_xlabel('Scale (left = low QPS, right = high QPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scheduler', fontsize=12, fontweight='bold')
    ax.set_title(f'EXP02 Priority 1: {trace_name} Trace Heatmap\n(7b-16R, Green = Better, Black Box = Best)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    return best_per_scale

def print_summary(results, trace_name, best_per_scale):
    """Print detailed summary for one trace."""
    print(f"\n{'='*80}")
    print(f"{trace_name} Trace - Detailed Analysis")
    print(f"{'='*80}")
    
    # Overall averages
    averages = {}
    for scheduler in SCHEDULERS:
        values = [results.get((scheduler, trace_name, scale), np.nan) for scale in SCALES]
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            averages[scheduler] = np.mean(valid)
    
    if not averages:
        print("No data!")
        return
    
    sorted_sched = sorted(averages.items(), key=lambda x: x[1])
    
    print("\n📊 OVERALL RANKINGS (by average E2E across all scales):")
    print(f"{'Rank':<6} {'Scheduler':<30} {'Avg E2E':<12} {'Quality'}")
    print("-" * 80)
    for rank, (sched, avg) in enumerate(sorted_sched, 1):
        quality = "🥇 Best" if rank == 1 else "🥈 2nd" if rank == 2 else "🥉 3rd" if rank == 3 else "⚠️  Worst" if rank > len(sorted_sched) - 3 else ""
        print(f"{rank:<6} {sched:<30} {avg:>8.2f}s    {quality}")
    
    print(f"\n🎯 BEST SCHEDULER AT EACH SCALE:")
    print(f"{'Scale':<8} {'QPS Range':<15} {'Best Scheduler':<30} {'E2E Time'}")
    print("-" * 80)
    for scale in SCALES:
        if scale in best_per_scale:
            _, scheduler, value = best_per_scale[scale]
            qps_range = f"~{1.0/scale:.3f}" if scale >= 1.0 else f"~{1.0/scale:.2f}"
            print(f"{scale:<8.1f} {qps_range:<15} {scheduler:<30} {value:>8.2f}s")
    
    # Check if there's a dominant winner
    winner_counts = {}
    for scale, (_, scheduler, _) in best_per_scale.items():
        winner_counts[scheduler] = winner_counts.get(scheduler, 0) + 1
    
    most_wins_sched = max(winner_counts.items(), key=lambda x: x[1])
    most_wins_count = most_wins_sched[1]
    total_scales = len(SCALES)
    
    print(f"\n🏆 DOMINANCE ANALYSIS:")
    if most_wins_count == total_scales:
        print(f"   {most_wins_sched[0]} is the CLEAR WINNER - best at ALL {total_scales} scales!")
    elif most_wins_count >= total_scales * 0.7:
        print(f"   {most_wins_sched[0]} is DOMINANT - best at {most_wins_count}/{total_scales} scales ({most_wins_count/total_scales*100:.0f}%)")
    else:
        print(f"   NO CLEAR WINNER - best scheduler varies by scale:")
        for sched, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
            print(f"      {sched}: {count}/{total_scales} scales")
    
    # Performance range analysis
    print(f"\n📈 PERFORMANCE RANGE ANALYSIS:")
    all_values = [v for v in averages.values() if not np.isnan(v)]
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        spread = max_val - min_val
        print(f"   Best:  {min_val:.2f}s ({sorted_sched[0][0]})")
        print(f"   Worst: {max_val:.2f}s ({sorted_sched[-1][0]})")
        print(f"   Spread: {spread:.2f}s ({spread/min_val*100:.1f}% relative to best)")
        
        if spread < 1.0:
            print(f"   → TIGHT RACE: <1s difference between best and worst")
        elif spread < 5.0:
            print(f"   → MODERATE DIFFERENCES: scheduler choice matters but not critical")
        else:
            print(f"   → LARGE GAPS: scheduler choice is CRITICAL for this trace!")
    
    # Scale progression (compare low QPS vs high QPS)
    print(f"\n🔄 LOW QPS → HIGH QPS PROGRESSION:")
    print(f"   Comparing Scale 48.0 (low load) vs Scale 1.5 (high load):")
    for sched in sorted_sched[:3]:  # Top 3
        scheduler = sched[0]
        val_low = results.get((scheduler, trace_name, 48.0), np.nan)
        val_high = results.get((scheduler, trace_name, 1.5), np.nan)
        if not np.isnan(val_low) and not np.isnan(val_high):
            change = val_high - val_low
            pct_change = (change / val_low) * 100
            direction = "↗️ DEGRADES" if change > 0 else "↘️ IMPROVES"
            print(f"   {scheduler:<30} {val_low:>6.2f}s → {val_high:>6.2f}s  {direction} ({pct_change:+.1f}%)")
    
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='exp02_all_traces.csv',
                        help='Input CSV file')
    parser.add_argument('--output-dir', default='plots',
                        help='Output directory')
    parser.add_argument('--traces', nargs='+', default=['Synth', 'Tool', 'Conv'])
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {csv_path}")
    results = load_csv(csv_path)
    print(f"Loaded {len(results)} data points\n")
    
    for trace in args.traces:
        print(f"\n{'='*80}")
        print(f"Generating {trace} Trace Heatmap")
        print(f"{'='*80}")
        
        output_path = output_dir / f'heatmap_{trace.lower()}.png'
        best_per_scale = plot_heatmap(results, trace, output_path)
        print_summary(results, trace, best_per_scale)
    
    print(f"\n{'='*80}")
    print(f"✅ Done! Generated {len(args.traces)} heatmaps in {output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
