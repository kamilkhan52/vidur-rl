#!/usr/bin/env python3
import argparse
import sys
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SCALE_ORDER = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]

def make_short_label(config_name):
    parts = config_name.replace("grad_", "")
    m = re.match(r"(\d+b-\d+R-\w+)_h([\d_]+)_r(.+)", parts)
    if not m:
        return config_name
    base = m.group(1)
    hidden = m.group(2)
    reward = m.group(3)
    h_short = hidden.split("_")[0]
    r_short = "e2e" if "e2e" in reward else "ttft"
    return f"{base}-{r_short}-{h_short}"

def plot_graduation_progress(df, output_path):
    fig, ax = plt.subplots(figsize=(20, 12))

    # Load baseline data
    baseline_df = None
    baseline_file = "/home/kamil/vidur-rl-working/plots/priority2_all.csv"
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        if "e2e_mean" in baseline_df.columns:
            baseline_df = baseline_df.rename(columns={"e2e_mean": "mean_e2e"})
        print(f"Loaded baselines: {len(baseline_df)} rows")

    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")

    # Color scheme for RL: blue shades for 7b, red shades for 70b
    configs = sorted(df["config"].unique())
    blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, 20))
    red_shades = plt.cm.Reds(np.linspace(0.4, 0.9, 20))

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

    # 1. Plot RL configs (thin, semi-transparent)
    for config_name in configs:
        cdf = df[df["config"] == config_name].sort_values("scale", ascending=False)
        if cdf.empty:
            continue

        scales = cdf["scale"].values
        e2e_vals = cdf["mean_e2e"].values
        min_scale = cdf["min_scale_reached"].iloc[0]
        color = config_colors[config_name]
        label = make_short_label(config_name)

        ax.plot(scales, e2e_vals, marker="o", markersize=3, linewidth=1.0,
                color=color, label=f"RL: {label}", alpha=0.5)

        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            ax.plot(min_scale, min_row["mean_e2e"].values[0],
                    marker="o", markersize=7, color=color,
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    # 2. Plot Baselines per Environment (thick, distinct colors)
    if baseline_df is not None:
        envs = set()
        for c in configs:
            m = re.match(r"grad_(\d+b)-(\d+R)-([A-Za-z0-9]+)_", c)
            if m:
                envs.add((m.group(1), m.group(2), m.group(3)))
                
        # Tab20 colormap for distinct environment colors
        base_cmap = plt.cm.tab20
        env_colors = {}
        for i, env in enumerate(sorted(list(envs))):
            env_colors[env] = base_cmap(i % 20)
            
        for env in sorted(list(envs)):
            b_model, b_rep, b_trace = env
            env_label = f"{b_model}-{b_rep}-{b_trace}"
            
            b_data = baseline_df[(baseline_df["model"] == b_model) & 
                                 (baseline_df["replicas"] == b_rep) & 
                                 (baseline_df["trace"] == b_trace)]
            
            if not b_data.empty:
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
                
                color = env_colors[env]
                
                # Plot Best Heuristic
                ax.plot(b_scales, b_best_vals, marker="x", markersize=8, linestyle="-.", 
                        linewidth=2.5, color=color, alpha=1.0, zorder=10,
                        label=f"★ BEST: {env_label}")
                        
                # Plot Random
                valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                if valid_random:
                    r_scales, r_vals = zip(*valid_random)
                    ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle=":",
                            linewidth=2.0, color=color, alpha=0.8, zorder=9,
                            label=f"■ RAND: {env_label}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Request Scale (48 = easy → 0.75 = hard)", fontsize=14)
    ax.set_ylabel("Mean E2E Latency (s)", fontsize=14)
    ax.set_title("Graduation Progress: RL vs Baselines", fontsize=18, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xticks(SCALE_ORDER)
    ax.set_xticklabels([str(s) for s in SCALE_ORDER])
    ax.grid(True, alpha=0.3, which="both")

    # Legend outside, split into multiple columns if needed
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="graduation_progress_enhanced.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_graduation_progress(df, args.output)

if __name__ == "__main__":
    main()
