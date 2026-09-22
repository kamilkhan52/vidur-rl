import sys

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_with_baselines2_threshold.py", "r") as f:
    code = f.read()

old_marker = """        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            ax.plot(min_scale, min_row["mean_e2e"].values[0],
                    marker="o", markersize=10, color=color,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)"""

new_marker = """        # Current position marker (black-edged dot at min scale)
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
                        zorder=10)"""

if old_marker not in code:
    print("Error: Could not find target code block.")
    sys.exit(1)

code = code.replace(old_marker, new_marker)

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_with_baselines2_threshold.py", "w") as f:
    f.write(code)
print("Patched successfully.")
