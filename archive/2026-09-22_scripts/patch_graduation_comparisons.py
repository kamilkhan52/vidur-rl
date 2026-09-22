import re

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_comparisons.py", "r") as f:
    code = f.read()

new_func = """def plot_comparison(df, compare_field, compare_values, output_path, colors=None):
    \"\"\"Create a 2x2 grid comparing configs that differ only in compare_field.\"\"\"
    if colors is None:
        colors = dict(zip(compare_values, plt.cm.tab10.colors[:len(compare_values)]))

    # Load baseline data
    import os
    baseline_df = None
    baseline_file = "/home/kamil/vidur-rl-working/plots/priority2_all.csv"
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        if "e2e_mean" in baseline_df.columns:
            baseline_df = baseline_df.rename(columns={"e2e_mean": "mean_e2e"})

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
            
            color = colors.get(val, "gray")
            ax.plot(cdf["scale"], cdf["mean_e2e"],
                    marker="o", markersize=6, linewidth=2,
                    color=color, label=f"{val} (RL)")
                    
            if baseline_df is not None:
                import re
                m = re.match(r"grad_(\\d+b)-(\\d+R)-([A-Za-z0-9]+)_", config_name)
                if m:
                    b_model, b_rep, b_trace = m.group(1), m.group(2), m.group(3)
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
                        if b_scales:
                            ax.plot(b_scales, b_best_vals, marker="x", markersize=6, linestyle=":",
                                    linewidth=1.5, color=color, alpha=0.8, label=f"{val} (Best Heuristic)")
                                    
                        valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                        if valid_random:
                            r_scales, r_vals = zip(*valid_random)
                            ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle="--",
                                    linewidth=1.5, color=color, alpha=0.6, label=f"{val} (Random)")

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
    print(f"Saved: {output_path}")"""

code = code.split("def plot_comparison")[0] + new_func + "\n\n\n" + "def plot_progress_table" + code.split("def plot_progress_table")[1]

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_comparisons_patched.py", "w") as f:
    f.write(code)

