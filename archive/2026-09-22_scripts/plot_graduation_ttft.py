#!/usr/bin/env python3
"""
Graduation progress plot using TTFT (prefill_e2e_time) instead of Mean E2E.

Same structure as plot_graduation_progress.py but plots median TTFT for RL configs
and baselines, to check whether RL policies optimized for TTFT reward show a
competitive advantage on that metric even if they don't win on E2E.

Usage:
    python scripts/plot_graduation_ttft.py \
        --input graduation_summary_ttft.csv \
        --output plots/graduation_progress_ttft.png
"""
import argparse
import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCALE_ORDER = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]


def make_short_label(config_name):
    parts = config_name.replace("grad_", "")
    m = re.match(r"(\d+b-\d+R-\w+(?:-PF)?)_h([\d_]+)_r(.+)", parts)
    if not m:
        return config_name
    base = m.group(1).replace("-PF", " PF")
    hidden = m.group(2)
    reward = m.group(3)
    h_short = hidden.split("_")[0]
    r_short = "e2e" if "e2e" in reward else "ttft"
    return f"{base}-{r_short}-{h_short}"


def plot_graduation_ttft(df, output_path):
    fig, ax = plt.subplots(figsize=(26, 16))

    # Load baseline data
    baseline_df = None
    baseline_file = "/home/kamil/vidur-rl-working/plots/priority2_postfix_ttft.csv"
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)

    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale <= 4.0)")

    # Graduation Threshold (10s on E2E; not directly applicable to TTFT, shown for reference)
    ax.axhline(y=10.0, color="green", linestyle="-.", linewidth=3.0,
               label="E2E Graduation Threshold (10.0s)", alpha=0.4, zorder=1)

    # Environment Color Mapping
    envs = set()
    configs = sorted(df["config"].unique())
    for c in configs:
        m = re.match(r"grad_(\d+b)-(\d+R)-([A-Za-z0-9]+)", c)
        if m:
            envs.add((m.group(1), m.group(2), m.group(3)))

    sorted_envs = sorted(list(envs))
    cmap = plt.colormaps["tab20"]
    env_colors = {env: cmap(i % 20) for i, env in enumerate(sorted_envs)}

    # Plot RL configs
    for config_name in configs:
        cdf = df[df["config"] == config_name].sort_values("scale", ascending=False)
        if cdf.empty:
            continue

        # Must have TTFT data
        cdf_ttft = cdf.dropna(subset=["median_ttft"])
        if cdf_ttft.empty:
            continue

        m = re.match(r"grad_(\d+b)-(\d+R)-([A-Za-z0-9]+)", config_name)
        env = (m.group(1), m.group(2), m.group(3)) if m else None
        color = env_colors.get(env, "black")

        scales = cdf_ttft["scale"].values
        ttft_vals = cdf_ttft["median_ttft"].values
        min_scale = cdf_ttft["scale"].min()
        label = make_short_label(config_name)

        ax.plot(scales, ttft_vals, marker="o", markersize=6, linewidth=2.5,
                color=color, linestyle="-", label=label, alpha=0.9, zorder=6)

        # Current position marker
        min_row = cdf_ttft[cdf_ttft["scale"] == min_scale]
        if not min_row.empty:
            min_y = min_row["median_ttft"].values[0]
            ax.plot(min_scale, min_y,
                    marker="o", markersize=12, color=color,
                    markeredgecolor="black", markeredgewidth=2.0, zorder=10)
            ax.annotate(label,
                        xy=(min_scale, min_y),
                        xytext=(10, 0),
                        textcoords="offset points",
                        fontsize=11,
                        color=color,
                        alpha=1.0,
                        fontweight="bold",
                        va="center",
                        ha="left",
                        clip_on=False,
                        zorder=11)

    # Plot Baselines ONCE per environment
    if baseline_df is not None and "median_ttft" in baseline_df.columns:
        for env in sorted_envs:
            b_model, b_rep, b_trace = env
            b_data = baseline_df[
                (baseline_df["model"] == b_model)
                & (baseline_df["replicas"] == b_rep)
                & (baseline_df["trace"] == b_trace)
            ]

            if not b_data.empty:
                b_scales = []
                b_best_vals = []
                b_random_vals = []

                for s in sorted(b_data["scale"].unique(), reverse=True):
                    s_data = b_data[b_data["scale"] == s]
                    if not s_data.empty:
                        b_scales.append(s)
                        b_best_vals.append(s_data["median_ttft"].min())
                        random_data = s_data[s_data["scheduler"] == "RANDOM"]
                        if not random_data.empty:
                            b_random_vals.append(random_data["median_ttft"].values[0])
                        else:
                            b_random_vals.append(None)

                b_color = env_colors[env]
                env_label = f"{b_model}-{b_rep}-{b_trace}"

                # Plot Best Heuristic
                if b_scales:
                    ax.plot(b_scales, b_best_vals, marker="x", markersize=10, linestyle=":",
                            linewidth=3.0, color=b_color, alpha=0.9,
                            label=f"Best: {env_label}", zorder=8)

                    end_idx = np.argmin(b_scales)
                    ax.annotate(env_label + " (Best)",
                                xy=(b_scales[end_idx], b_best_vals[end_idx]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                fontsize=11,
                                color=b_color,
                                alpha=1.0,
                                fontweight="bold",
                                va="center",
                                ha="left",
                                clip_on=False,
                                zorder=11)

                # Plot Random
                valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                if valid_random:
                    r_scales, r_vals = zip(*valid_random)
                    ax.plot(r_scales, r_vals, marker="s", markersize=8, linestyle="--",
                            linewidth=3.0, color=b_color, alpha=0.9,
                            label=f"Random: {env_label}", zorder=7)

                    end_idx_r = np.argmin(r_scales)
                    ax.annotate(env_label + " (Random)",
                                xy=(r_scales[end_idx_r], r_vals[end_idx_r]),
                                xytext=(10, 0),
                                textcoords="offset points",
                                fontsize=11,
                                color=b_color,
                                alpha=1.0,
                                fontweight="bold",
                                va="center",
                                ha="left",
                                clip_on=False,
                                zorder=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Request Scale (48 = easy → 0.75 = hard)", fontsize=16)
    ax.set_ylabel("Median TTFT (s)", fontsize=16)
    ax.set_title("Graduation Progress: Median TTFT (Time-To-First-Token)", fontsize=20, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xticks(SCALE_ORDER)
    ax.set_xticklabels([str(s) for s in SCALE_ORDER], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, alpha=0.3, which="both")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="graduation_progress_ttft.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_graduation_ttft(df, args.output)


if __name__ == "__main__":
    main()
