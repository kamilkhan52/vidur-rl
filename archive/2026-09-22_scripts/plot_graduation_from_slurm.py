#!/usr/bin/env python3
"""
Plot graduation training progress from SLURM output logs.
Extracts metrics like E2E latency and scale progression across iterations.
"""

import re
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import subprocess

def get_remote_logs(remote_path, remote_host="alpine"):
    """Fetch remote log content via SSH."""
    cmd = f"ssh -o ControlMaster=no {remote_host} 'cat {remote_path}/*.out'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def parse_slurm_log_content(content):
    """Parse slurm log content for metrics."""
    iterations = []
    e2e_times = []
    scales = []
    
    # Pattern: "Scale X.X Iteration Results: Mean E2E = Y.YYs"
    pattern = re.compile(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s', re.IGNORECASE)
    
    # Also catch: "Mean E2E = X.Xs" in context
    alt_pattern = re.compile(r'Mean E2E[:\s]+=?\s*([\d.]+)s?', re.IGNORECASE)
    
    lines = content.split('\n')
    
    for line in lines:
        match = pattern.search(line)
        if match:
            scale = float(match.group(1))
            e2e = float(match.group(2))
            scales.append(scale)
            e2e_times.append(e2e)
    
    return scales, e2e_times


def plot_graduation_progress(remote_log_dir, output_path=None):
    """Plot graduation progress from remote slurm logs."""
    
    # Fetch all log content
    print(f"Fetching logs from: {remote_log_dir}")
    cmd = f"ssh -o ControlMaster=no alpine 'cat {remote_log_dir}/vidur_grad_23669902_*.out'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"Error fetching logs: {result.stderr}")
        return None
    
    all_content = result.stdout
    print(f"Fetched {len(all_content):,} bytes of log data")
    
    # Parse content - find scale/E2E pairs
    pattern = re.compile(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s', re.IGNORECASE)
    matches = pattern.findall(all_content)
    
    if not matches:
        print("No scale/E2E data found in logs")
        # Try alternate pattern
        alt_pattern = re.compile(r'Mean E2E[:\s]+=?\s*([\d.]+)s?', re.IGNORECASE)
        matches = alt_pattern.findall(all_content)
        print(f"Found {len(matches)} E2E values with alternate pattern")
    
    # Collect unique scale/E2E combinations
    data_points = set()
    for match in matches:
        try:
            if len(match) == 2:
                scale, e2e = float(match[0]), float(match[1])
            else:
                scale, e2e = 48.0, float(match[0])  # Default scale
            data_points.add((scale, e2e))
        except:
            pass
    
    data_points = sorted(list(data_points))
    scales = [p[0] for p in data_points]
    e2es = [p[1] for p in data_points]
    
    print(f"Parsed {len(data_points)} unique scale/E2E data points")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: E2E vs Scale (all data)
    ax1 = axes[0, 0]
    if scales:
        ax1.scatter(scales, e2es, alpha=0.6, s=50, c='steelblue', edgecolors='darkblue')
        ax1.set_xlabel('Request Scale')
        ax1.set_ylabel('Mean E2E Latency (s)')
        ax1.set_title('E2E Latency vs Scale (All Tasks)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    
    # Add tension zone
    ax1.axvspan(0.5, 4, alpha=0.1, color='red')
    ax1.text(2, max(e2es)*0.3 if e2es else 100, 'Tension\nZone', ha='center', fontsize=10, color='red')
    
    # Plot 2: Scale distribution
    ax2 = axes[0, 1]
    unique_scales = sorted(set(scales))
    scale_counts = [scales.count(s) for s in unique_scales]
    
    if unique_scales:
        ax2.bar(range(len(unique_scales)), scale_counts, color='coral', edgecolor='darkred')
        ax2.set_xticks(range(len(unique_scales)))
        ax2.set_xticklabels([f'{s:.1f}' for s in unique_scales], rotation=45)
        ax2.set_xlabel('Scale')
        ax2.set_ylabel('Number of Iterations')
        ax2.set_title('Iteration Distribution by Scale')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: E2E distribution
    ax3 = axes[1, 0]
    if e2es:
        ax3.hist(e2es, bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
        ax3.set_xlabel('E2E Latency (s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('E2E Latency Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xscale('log')
    
    # Plot 4: Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "=== Graduation Training Summary ===\n" + "="*32 + "\n\n"
    
    if scales and e2es:
        summary_text += f"Total Data Points: {len(data_points)}\n"
        summary_text += f"Unique Scales: {len(unique_scales)}\n"
        summary_text += f"\nScale Range: {min(scales):.1f} → {max(scales):.1f}\n"
        summary_text += f"E2E Range: {min(e2es):.3f}s → {max(e2es):.1f}s\n"
        
        # Stats by scale band
        summary_text += "\n*** Performance by Scale:\n"
        scale_bands = [(48, "48.0"), (24, "24.0"), (12, "12.0"), (6, "6.0"), (3, "3.0"), (1.5, "1.5")]
        
        for band, label in scale_bands:
            band_es = [e for s, e in data_points if s == band]
            if band_es:
                avg_e2e = sum(band_es) / len(band_es)
                summary_text += f"  Scale {label:>5}: avg E2E = {avg_e2e:>8.3f}s (n={len(band_es)})\n"
        
        # Low-scale performance (tension zone)
        low_es = [e for s, e in data_points if s <= 6]
        if low_es:
            avg_low = sum(low_es) / len(low_es)
            summary_text += "\n*** Tension Zone (<=6.0):\n"
            summary_text += f"  Avg E2E: {avg_low:.3f}s (n={len(low_es)})\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved to: {output_path}")
    
    return fig


if __name__ == "__main__":
    import sys
    
    remote_log_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/kamil@colostate.edu/slurm_logs"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/home/kamil/.openclaw/workspace/graduation_progress.png"
    
    fig = plot_graduation_progress(remote_log_dir, output_path)
    if fig:
        plt.show()
