#!/usr/bin/env python3
"""
Side-by-side comparison: how competitive is RL vs best heuristic on TTFT vs E2E?

Produces a ratio plot (best RL / best heuristic) at each scale, with a table of
absolute values underneath.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

SCALE_ORDER = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5, 0.75]


def compute_ratios(grad_df, base_df, model, replicas, trace):
    # RL data: include -PF variants for the same base environment
    rl = grad_df[
        (grad_df["model"] == model)
        & (grad_df["replicas"] == replicas)
        & (grad_df["trace"].str.replace("-PF", "") == trace)
    ]
    # Baseline data (no -PF in baselines)
    bl = base_df[
        (base_df["model"] == model)
        & (base_df["replicas"] == replicas)
        & (base_df["trace"] == trace)
    ]
    if rl.empty or bl.empty:
        return None

    rows = []
    for scale in sorted(set(rl["scale"].unique()) & set(bl["scale"].unique()), reverse=True):
        rl_s = rl[rl["scale"] == scale]
        bl_s = bl[bl["scale"] == scale]
        bl_s_no_rand = bl_s[bl_s["scheduler"] != "RANDOM"]
        if bl_s_no_rand.empty:
            continue

        row = {"scale": scale}

        # E2E (median)
        rl_e2e = rl_s["median_e2e"].min()
        bl_e2e = bl_s_no_rand["median_e2e"].min()
        if bl_e2e > 0:
            row["e2e_ratio"] = rl_e2e / bl_e2e
            row["rl_e2e"] = rl_e2e
            row["bl_e2e"] = bl_e2e

        # TTFT
        rl_ttft_df = rl_s.dropna(subset=["median_ttft"])
        bl_ttft_df = bl_s_no_rand.dropna(subset=["median_ttft"])
        if not rl_ttft_df.empty and not bl_ttft_df.empty:
            rl_ttft = rl_ttft_df["median_ttft"].min()
            bl_ttft = bl_ttft_df["median_ttft"].min()
            if bl_ttft > 0:
                row["ttft_ratio"] = rl_ttft / bl_ttft
                row["rl_ttft"] = rl_ttft
                row["bl_ttft"] = bl_ttft

        if "e2e_ratio" in row or "ttft_ratio" in row:
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-input", default="graduation_summary_ttft.csv")
    parser.add_argument("--baseline-input", default="plots/baselines_all_ttft.csv")
    parser.add_argument("--output", default="plots/ttft_vs_e2e_comparison.png")
    args = parser.parse_args()

    grad_df = pd.read_csv(args.grad_input)
    base_df = pd.read_csv(args.baseline_input)

    # Find environments with ≥2 overlapping TTFT scales
    envs_to_plot = []
    for (model, replicas, trace), _ in base_df.groupby(["model", "replicas", "trace"]):
        ratios = compute_ratios(grad_df, base_df, model, replicas, trace)
        if ratios is not None and "ttft_ratio" in ratios.columns:
            ttft_rows = ratios.dropna(subset=["ttft_ratio"])
            if len(ttft_rows) >= 2:
                envs_to_plot.append((model, replicas, trace, ratios))

    if not envs_to_plot:
        print("No environments with sufficient overlapping data.")
        return

    n_envs = len(envs_to_plot)
    fig = plt.figure(figsize=(11 * n_envs, 12))
    gs = GridSpec(2, n_envs, height_ratios=[3, 1], hspace=0.35)

    for idx, (model, replicas, trace, ratios) in enumerate(envs_to_plot):
        ax = fig.add_subplot(gs[0, idx])
        ratios = ratios.sort_values("scale", ascending=False)
        scales = ratios["scale"].values

        # --- Ratio plot ---

        # Reference line at 1.0
        ax.axhline(y=1.0, color="#2ca02c", linestyle="--", linewidth=2.5, alpha=0.8,
                    label="RL = Heuristic", zorder=2)

        # E2E ratio line
        if "e2e_ratio" in ratios.columns:
            e2e_vals = ratios["e2e_ratio"].values
            ax.plot(scales, e2e_vals, "o-", color="#d62728", linewidth=3, markersize=13,
                    label="Median E2E Latency", zorder=5, markeredgecolor="white", markeredgewidth=1.5)
            for s, v in zip(scales, e2e_vals):
                ax.annotate(f"{v:.2f}x", (s, v), textcoords="offset points",
                            xytext=(0, 16), fontsize=13, color="#d62728",
                            ha="center", fontweight="bold")

        # TTFT ratio line
        if "ttft_ratio" in ratios.columns:
            ttft_valid = ratios.dropna(subset=["ttft_ratio"])
            t_scales = ttft_valid["scale"].values
            ttft_vals = ttft_valid["ttft_ratio"].values
            ax.plot(t_scales, ttft_vals, "s-", color="#1f77b4", linewidth=3, markersize=13,
                    label="Median TTFT", zorder=5, markeredgecolor="white", markeredgewidth=1.5)
            for s, v in zip(t_scales, ttft_vals):
                ax.annotate(f"{v:.2f}x", (s, v), textcoords="offset points",
                            xytext=(0, -20), fontsize=13, color="#1f77b4",
                            ha="center", fontweight="bold")

        # Shading
        xlim_lo, xlim_hi = scales.max() * 1.5, scales.min() * 0.7
        all_vals = list(ratios.get("e2e_ratio", pd.Series()).dropna())
        all_vals += list(ratios.get("ttft_ratio", pd.Series()).dropna())
        y_lo = max(0, min(all_vals) - 0.4) if all_vals else 0
        y_hi = max(all_vals) + 0.6 if all_vals else 3
        ax.fill_between([xlim_lo, xlim_hi], y_lo, 1.0, alpha=0.06, color="green", zorder=0)
        ax.fill_between([xlim_lo, xlim_hi], 1.0, y_hi, alpha=0.06, color="red", zorder=0)

        ax.set_xscale("log")
        ax.set_xlabel("Request Scale (higher = easier)", fontsize=15)
        ax.set_ylabel("Ratio: Best RL / Best Heuristic", fontsize=15)
        ax.set_title(f"{model}-{replicas}-{trace}", fontsize=18, fontweight="bold")
        ax.invert_xaxis()
        valid_scales = [s for s in SCALE_ORDER if s >= scales.min() * 0.9 and s <= scales.max() * 1.1]
        ax.set_xticks(valid_scales)
        ax.set_xticklabels([str(s) for s in valid_scales], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=13, loc="upper left", framealpha=0.9)

        # --- Table underneath ---
        ax_table = fig.add_subplot(gs[1, idx])
        ax_table.axis("off")

        table_data = []
        col_labels = ["Scale"]
        has_ttft = "ttft_ratio" in ratios.columns

        col_labels += ["RL E2E", "Best Heur E2E"]
        if has_ttft:
            col_labels += ["RL TTFT", "Best Heur TTFT"]

        for _, row in ratios.sort_values("scale", ascending=False).iterrows():
            r = [f"{row['scale']:.1f}"]
            r.append(f"{row.get('rl_e2e', float('nan')):.2f}s")
            r.append(f"{row.get('bl_e2e', float('nan')):.2f}s")
            if has_ttft:
                rl_t = row.get("rl_ttft", float("nan"))
                bl_t = row.get("bl_ttft", float("nan"))
                r.append(f"{rl_t:.2f}s" if not pd.isna(rl_t) else "—")
                r.append(f"{bl_t:.2f}s" if not pd.isna(bl_t) else "—")
            table_data.append(r)

        table = ax_table.table(cellText=table_data, colLabels=col_labels,
                               loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.6)

        # Color header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#e0e0e0")
            table[0, j].set_text_props(fontweight="bold")

    fig.suptitle("RL Competitiveness: TTFT vs E2E Latency\n(ratio < 1.0 = RL beats best heuristic)",
                 fontsize=20, fontweight="bold", y=0.98)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
