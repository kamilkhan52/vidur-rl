#!/usr/bin/env python3
"""
Plot training effort heatmap for graduation sweep.

Shows total iterations trained at each scale for each config track.
Useful for understanding training distribution across the curriculum.

Usage:
    python scripts/plot_training_effort.py --input training_effort.csv --output training_effort_heatmap.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCALE_PROGRESSION = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]
SCALE_LABELS = [str(s) for s in SCALE_PROGRESSION]


def plot_training_effort_heatmap(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if df.empty:
        print("Error: Empty CSV", file=sys.stderr)
        return
    
    # Prepare data for heatmap
    # Rows: configs, Columns: scales
    scale_cols = [f"scale_{s}" for s in SCALE_PROGRESSION]
    
    # Sort configs: 7b first, then by trace
    df["sort_key"] = df["model"] + "-" + df["trace"]
    df = df.sort_values("sort_key").reset_index(drop=True)
    
    # Create heatmap data
    heatmap_data = df[scale_cols].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 20))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Total Iterations", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(SCALE_PROGRESSION)))
    ax.set_xticklabels(SCALE_LABELS, fontsize=10)
    
    # Create y-axis labels (config names shortened)
    config_labels = []
    for _, row in df.iterrows():
        label = f"{row['model']}-{row['replicas']}-{row['trace'][:4]}-{row['reward'][:4]}"
        config_labels.append(label)
    
    ax.set_yticks(range(len(config_labels)))
    ax.set_yticklabels(config_labels, fontsize=8)
    
    # Add value annotations
    for i in range(len(df)):
        for j in range(len(SCALE_PROGRESSION)):
            val = heatmap_data[i, j]
            if val > 0:
                text_color = "white" if val > 30 else "black"
                ax.text(j, i, int(val), ha="center", va="center", 
                       color=text_color, fontsize=6)
    
    # Labels and title
    ax.set_xlabel("Scale (lower = higher QPS)", fontsize=12)
    ax.set_ylabel("Config", fontsize=12)
    ax.set_title("Training Effort: Total Iterations per Scale\n(Graduation Curriculum Progress)", fontsize=14)
    
    # Invert x-axis so 48 is on left (as per convention)
    ax.invert_xaxis()
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(SCALE_PROGRESSION), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    
    # Also create a compact summary version
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Aggregate by model and trace
    pivot = df.groupby(["model", "trace"])[scale_cols].sum()
    
    # Reorder
    pivot = pivot.reindex([
        ("7b", "Synth"), ("7b", "Tool"), ("7b", "Conv"),
        ("70b", "Synth"), ("70b", "Tool"), ("70b", "Conv")
    ])
    
    im2 = ax2.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.5)
    cbar2.set_label("Total Iterations", fontsize=12)
    
    ax2.set_xticks(range(len(SCALE_LABELS)))
    ax2.set_xticklabels(SCALE_LABELS, fontsize=10)
    
    row_labels = [f"{m}-{t[:4]}" for m, t in pivot.index]
    ax2.set_yticks(range(len(row_labels)))
    ax2.set_yticklabels(row_labels, fontsize=10)
    
    ax2.set_xlabel("Scale", fontsize=12)
    ax2.set_ylabel("Model-Trace", fontsize=12)
    ax2.set_title("Training Effort by Model & Trace", fontsize=14)
    ax2.invert_xaxis()
    
    # Annotate
    for i in range(len(pivot)):
        for j in range(len(SCALE_PROGRESSION)):
            val = pivot.values[i, j]
            if val > 0:
                text_color = "white" if val > 100 else "black"
                ax2.text(j, i, int(val), ha="center", va="center", 
                        color=text_color, fontsize=9)
    
    plt.tight_layout()
    output_summary = output_file.replace(".png", "_summary.png")
    plt.savefig(output_summary, dpi=150, bbox_inches="tight")
    print(f"Saved summary to {output_summary}")


def main():
    parser = argparse.ArgumentParser(description="Plot training effort heatmap")
    parser.add_argument("--input", required=True, help="Input CSV from aggregate_training_effort.py")
    parser.add_argument("--output", default="training_effort_heatmap.png", help="Output PNG file")
    args = parser.parse_args()
    
    plot_training_effort_heatmap(args.input, args.output)


if __name__ == "__main__":
    main()
