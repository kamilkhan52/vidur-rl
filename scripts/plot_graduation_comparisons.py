#!/usr/bin/env python3
"""
Generate paired comparison plots and progress tables for graduation sweep.

Reads from the CSV produced by aggregate_graduation.py.

Usage:
    python scripts/plot_graduation_comparisons.py --input graduation_summary.csv --output-dir .

Generates:
    - comparison_model.png      (7b vs 70b)
    - comparison_reward.png     (e2e_time vs ttft)
    - comparison_architecture.png (512 vs 1024)
    - comparison_replicas.png   (8R vs 16R)
    - comparison_trace.png      (Synth vs Tool vs Conv)
    - progress_table_synth.png  (iteration progress for Synth configs)
    - progress_table_toolconv.png (iteration progress for Tool/Conv configs)
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def get_config_key(row, exclude_field):
    """Build a grouping key from config components, excluding one field."""
    parts = {
        "model": row["model"],
        "replicas": row["replicas"],
        "trace": row["trace"],
        "hidden": row["hidden"],
        "reward": row["reward"],
    }
    del parts[exclude_field]
    return tuple(sorted(parts.items()))


def plot_comparison(df, compare_field, compare_values, output_path, colors=None):
    """Create a 2x2 grid comparing configs that differ only in compare_field."""
    if colors is None:
        colors = dict(zip(compare_values, plt.cm.tab10.colors[:len(compare_values)]))

    # Find paired groups
    groups = {}
    for _, row in df.drop_duplicates("config").iterrows():
        key = get_config_key(row, compare_field)
        if key not in groups:
            groups[key] = {}
        groups[key][row[compare_field]] = row["config"]

    # Keep only groups where at least one variant graduated
    valid_groups = {}
    for key, variants in groups.items():
        if len(variants) < 2:
            continue
        for config_name in variants.values():
            cdf = df[df["config"] == config_name]
            if cdf["graduated"].any():
                valid_groups[key] = variants
                break

    if not valid_groups:
        print(f"  No valid comparison groups for {compare_field}")
        return

    # Select up to 4 groups for 2x2 grid
    selected = list(valid_groups.items())[:4]
    nplots = len(selected)
    ncols = min(2, nplots)
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f"Comparison: {compare_field.title()}", fontsize=14, fontweight="bold")

    for idx, (key, variants) in enumerate(selected):
        ax = axes[idx // ncols][idx % ncols]
        key_label = ", ".join(f"{k}={v}" for k, v in key)

        for val in compare_values:
            if val not in variants:
                continue
            config_name = variants[val]
            cdf = df[df["config"] == config_name].sort_values("scale", ascending=False)
            if cdf.empty:
                continue
            ax.plot(cdf["scale"], cdf["mean_e2e"],
                    marker="o", markersize=6, linewidth=2,
                    color=colors.get(val, "gray"), label=str(val))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Scale")
        ax.set_ylabel("Mean E2E (s)")
        ax.set_title(f"{key_label}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(nplots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_progress_table(df, trace_filter, output_path, title):
    """Create a table-style progress visualization."""
    filtered = df[df["trace"].isin(trace_filter)].copy()
    if filtered.empty:
        print(f"  No data for traces: {trace_filter}")
        return

    configs = sorted(filtered["config"].unique())
    rows_data = []

    for config_name in configs:
        cdf = filtered[filtered["config"] == config_name].sort_values("scale", ascending=False)
        if cdf.empty:
            continue
        label = make_short_label(config_name)
        scales = cdf["scale"].values
        e2e_vals = cdf["mean_e2e"].values
        min_scale = cdf["min_scale_reached"].iloc[0]

        # Start, mid, end snapshots
        start_scale = scales[0] if len(scales) > 0 else ""
        start_e2e = f"{e2e_vals[0]:.2f}" if len(e2e_vals) > 0 else ""
        mid_idx = len(scales) // 2
        mid_scale = scales[mid_idx] if len(scales) > 1 else ""
        mid_e2e = f"{e2e_vals[mid_idx]:.2f}" if len(e2e_vals) > 1 else ""
        end_scale = scales[-1] if len(scales) > 0 else ""
        end_e2e = f"{e2e_vals[-1]:.2f}" if len(e2e_vals) > 0 else ""

        rows_data.append([
            label,
            f"{start_scale}", start_e2e,
            f"{mid_scale}", mid_e2e,
            f"{end_scale}", end_e2e,
            f"{min_scale}",
        ])

    if not rows_data:
        return

    col_labels = ["Config", "Start Scale", "E2E (s)", "Mid Scale", "E2E (s)",
                   "End Scale", "E2E (s)", "Min Scale"]

    fig, ax = plt.subplots(figsize=(16, max(3, 0.5 * len(rows_data) + 1)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(len(rows_data)):
        color = "#f0f0f0" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate graduation comparison plots and progress tables.")
    parser.add_argument("--input", required=True, help="Input CSV from aggregate_graduation.py")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print("ERROR: Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Comparison plots
    comparisons = [
        ("model", ["7b", "70b"], {"7b": "blue", "70b": "red"}),
        ("reward", ["e2e_time", "ttft"], {"e2e_time": "blue", "ttft": "green"}),
        ("hidden", sorted(df["hidden"].unique()), None),
        ("replicas", ["8R", "16R"], {"8R": "blue", "16R": "orange"}),
        ("trace", ["Synth", "Tool", "Conv"], {"Synth": "blue", "Tool": "orange", "Conv": "green"}),
    ]

    field_to_filename = {
        "model": "comparison_model.png",
        "reward": "comparison_reward.png",
        "hidden": "comparison_architecture.png",
        "replicas": "comparison_replicas.png",
        "trace": "comparison_trace.png",
    }

    for field, values, colors in comparisons:
        output_path = os.path.join(out, field_to_filename[field])
        print(f"Generating {output_path}...")
        plot_comparison(df, field, values, output_path, colors)

    # Progress tables
    print("Generating progress tables...")
    plot_progress_table(
        df, ["Synth"],
        os.path.join(out, "progress_table_synth.png"),
        "Graduation Progress: SYNTH Traces\n(Iteration → Scale → E2E)"
    )
    plot_progress_table(
        df, ["Tool", "Conv"],
        os.path.join(out, "progress_table_toolconv.png"),
        "Graduation Progress: TOOL + CONV Traces\n(Iteration → Scale → E2E)"
    )


if __name__ == "__main__":
    main()
