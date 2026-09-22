import re

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress.py", "r") as f:
    code = f.read()

replacement = """def plot_graduation_progress(df, output_path):
    fig, ax = plt.subplots(figsize=(16, 10))

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
                    
                    # Add baseline traces to plot
                    ax.plot(b_scales, b_best_vals, marker="x", markersize=8, linestyle=":", 
                            linewidth=1.5, color=color, alpha=0.8, 
                            label=f"Best Heuristic" if not baseline_plotted else "_nolegend_")
                            
                    valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                    if valid_random:
                        r_scales, r_vals = zip(*valid_random)
                        ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle="--",
                                linewidth=1.5, color=color, alpha=0.6,
                                label=f"Random" if not baseline_plotted else "_nolegend_")
                    
                    baseline_plotted = True # Only label baselines once to avoid legend bloat
            else:
                print(f"Regex didn't match: {config_name}")

        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            ax.plot(min_scale, min_row["mean_e2e"].values[0],
                    marker="o", markersize=10, color=color,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)

    ax.set_xscale("log")"""

code = code.replace("""def plot_graduation_progress(df, output_path):
    fig, ax = plt.subplots(figsize=(16, 10))

    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")

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

        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            ax.plot(min_scale, min_row["mean_e2e"].values[0],
                    marker="o", markersize=10, color=color,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)

    ax.set_xscale("log")""", replacement)

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_with_baselines2.py", "w") as f:
    f.write(code)
