#!/usr/bin/env python3
"""
Generate the dense graduation progress plot (all tracks on one chart).

Reads from the CSV produced by aggregate_graduation.py.

Usage:
    python scripts/plot_graduation_progress.py --input graduation_summary.csv --output graduation_progress_dense.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SCALE_ORDER = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]


def make_short_label(config_name):
    """Shorten grad_7b-8R-Synth_h512_512_256_128_re2e_time -> 7b-8R-Synth-e2e-512."""
    parts = config_name.replace("grad_", "")
    # Extract key parts
    import re
    m = re.match(r"(\d+b-\d+R-\w+)_h([\d_]+)_r(.+)", parts)
    if not m:
        return config_name
    base = m.group(1)
    hidden = m.group(2)
    reward = m.group(3)
    # Shorten hidden: 512_512_256_128 -> 512, 1024_1024_512 -> 1024
    h_short = hidden.split("_")[0]
    # Shorten reward
    r_short = "e2e" if "e2e" in reward else "ttft"
    return f"{base}-{r_short}-{h_short}"


def plot_graduation_progress(df, output_path):
    fig, ax = plt.subplots(figsize=(22, 12))

    # Load baseline data
    import os
    baseline_df = None
    baseline_file = "/home/kamil/vidur-rl-working/plots/priority2_all.csv"
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        if "e2e_mean" in baseline_df.columns:
            baseline_df = baseline_df.rename(columns={"e2e_mean": "mean_e2e"})
        print(f"Loaded baselines: {len(baseline_df)} rows")

    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")

    # Graduation Threshold
    ax.axhline(y=10.0, color="green", linestyle="-.", linewidth=2.0, label="Graduation Threshold (10.0s)", zorder=1)

    # Color scheme: blue shades for 7b, red shades for 70b
    configs = sorted(df["config"].unique())
    blue_shades = plt.cm.Blues(np.linspace(0.3, 0.9, 16))
    red_shades = plt.cm.Reds(np.linspace(0.3, 0.9, 16))

    blue_idx = 0
    red_idx = 0
    config_colors = {}
    for c in configs:
        if "7b" in c:
            config_colors[c] = blue_shades[blue_idx % len(blue_shades)]
            blue_idx += 1
        else:
            config_colors[c] = red_shades[red_idx % len(red_shades)]
            red_idx += 1

    baseline_plotted = False
    for config_name in configs:
        cdf = df[df["config"] == config_name].sort_values("scale", ascending=False)
        if cdf.empty:
            continue

        scales = cdf["scale"].values
        e2e_vals = cdf["mean_e2e"].values
        min_scale = cdf["min_scale_reached"].iloc[0]
        color = config_colors[config_name]
        label = make_short_label(config_name)

        ax.plot(scales, e2e_vals, marker="o", markersize=4, linewidth=1.2,
                color=color, label=label, alpha=0.8)

        # Plot baseline if available for this config
        if baseline_df is not None:
            # Parse config parts to match baseline
            import re
            m = re.match(r"grad_(\d+b)-(\d+R)-([A-Za-z0-9]+)_", config_name)
            if m:
                b_model, b_rep, b_trace = m.group(1), m.group(2), m.group(3)
                
                # Find matching baselines
                b_data = baseline_df[(baseline_df["model"] == b_model) & 
                                     (baseline_df["replicas"] == b_rep) & 
                                     (baseline_df["trace"] == b_trace)]
                
                if not b_data.empty:
                    # Calculate best heuristic per scale
                    b_scales = []
                    b_best_vals = []
                    b_random_vals = []
                    
                    for s in sorted(b_data["scale"].unique(), reverse=True):
                        s_data = b_data[b_data["scale"] == s]
                        if not s_data.empty:
                            b_scales.append(s)
                            b_best_vals.append(s_data["mean_e2e"].min())
                            random_data = s_data[s_data["scheduler"] == "RANDOM"]
                            if not random_data.empty:
                                b_random_vals.append(random_data["mean_e2e"].min())
                            else:
                                b_random_vals.append(None)
                    

                    # Generate a unique visible color for baselines instead of blending with the RL line
                    # We can pick from tab20
                    import matplotlib.cm as cm
                    base_hash = hash(b_model + b_rep + b_trace) % 20
                    b_color = cm.tab20(base_hash)

                    # Add baseline traces to plot
                    ax.plot(b_scales, b_best_vals, marker="x", markersize=8, linestyle=":", 
                            linewidth=1.5, color=b_color, alpha=0.9, 
                            label=f"Best Heuristic" if not baseline_plotted else "_nolegend_")
                    
                    if b_scales:
                        end_idx = np.argmin(b_scales)
                        ax.annotate(label + " (Best)",
                                    xy=(b_scales[end_idx], b_best_vals[end_idx]),
                                    xytext=(8, 0),
                                    textcoords="offset points",
                                    fontsize=6,
                                    color=b_color,
                                    alpha=0.9,
                                    fontweight="bold",
                                    va="center",
                                    ha="left",
                                    clip_on=False,
                                    zorder=9)
                            
                    valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                    if valid_random:
                        r_scales, r_vals = zip(*valid_random)
                        ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle="--",
                                linewidth=1.5, color=b_color, alpha=0.9,
                                label=f"Random" if not baseline_plotted else "_nolegend_")
                        
                        end_idx_r = np.argmin(r_scales)
                        ax.annotate(label + " (Random)",
                                    xy=(r_scales[end_idx_r], r_vals[end_idx_r]),
                                    xytext=(8, 0),
                                    textcoords="offset points",
                                    fontsize=6,
                                    color=b_color,
                                    alpha=0.9,
                                    fontweight="bold",
                                    va="center",
                                    ha="left",
                                    clip_on=False,
                                    zorder=8)
                    
                    baseline_plotted = True # Only label baselines once to avoid legend bloat
            else:
                print(f"Regex didn't match: {config_name}")

        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            min_y = min_row["mean_e2e"].values[0]
            ax.plot(min_scale, min_y,
                    marker="o", markersize=10, color=color,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            # Add transparent label directly to the right
            ax.annotate(label,
                        xy=(min_scale, min_y),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        alpha=0.75,
                        va="center",
                        ha="left",
                        clip_on=False,
                        zorder=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Request Scale (48 = easy → 0.75 = hard)", fontsize=12)
    ax.set_ylabel("Mean E2E Latency (s)", fontsize=12)
    ax.set_title("Graduation Progress: All Tracks (● = Current Position)", fontsize=14, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xticks(SCALE_ORDER)
    ax.set_xticklabels([str(s) for s in SCALE_ORDER])
    ax.grid(True, alpha=0.3, which="both")

    # Legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate dense graduation progress plot.")
    parser.add_argument("--input", required=True, help="Input CSV from aggregate_graduation.py")
    parser.add_argument("--output", default="graduation_progress_dense.png", help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("ERROR: Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    plot_graduation_progress(df, args.output)


if __name__ == "__main__":
    main()
