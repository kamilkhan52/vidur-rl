import re
import sys

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_postfix.py", "r") as f:
    code = f.read()

# For RL:
rl_old = """        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            ax.plot(min_scale, min_row["mean_e2e"].values[0],
                    marker="o", markersize=8, color=color,
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)"""

rl_new = """        # Current position marker (black-edged dot at min scale)
        min_row = cdf[cdf["scale"] == min_scale]
        if not min_row.empty:
            min_y = min_row["mean_e2e"].values[0]
            ax.plot(min_scale, min_y,
                    marker="o", markersize=8, color=color,
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)
            # Add transparent label directly to the right
            ax.annotate(f"RL: {label}",
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

code = code.replace(rl_old, rl_new)

# For Baselines:
b_old = """                # Plot Best Heuristic
                ax.plot(b_scales, b_best_vals, marker="x", markersize=9, linestyle="-.", 
                        linewidth=2.5, color=color, alpha=1.0, zorder=10,
                        label=f"★ BEST: {env_label}")
                        
                # Plot Random
                valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                if valid_random:
                    r_scales, r_vals = zip(*valid_random)
                    ax.plot(r_scales, r_vals, marker="s", markersize=7, linestyle=":",
                            linewidth=2.0, color=color, alpha=0.8, zorder=9,
                            label=f"■ RAND: {env_label}")"""

b_new = """                # Plot Best Heuristic
                ax.plot(b_scales, b_best_vals, marker="x", markersize=9, linestyle="-.", 
                        linewidth=2.5, color=color, alpha=1.0, zorder=10,
                        label=f"★ BEST: {env_label}")
                
                if b_scales:
                    end_idx = np.argmin(b_scales)
                    ax.annotate(f"★ BEST: {env_label}",
                                xy=(b_scales[end_idx], b_best_vals[end_idx]),
                                xytext=(8, 0),
                                textcoords="offset points",
                                fontsize=8,
                                color=color,
                                fontweight="bold",
                                alpha=0.9,
                                va="center",
                                ha="left",
                                clip_on=False,
                                zorder=11)
                        
                # Plot Random
                valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                if valid_random:
                    r_scales, r_vals = zip(*valid_random)
                    ax.plot(r_scales, r_vals, marker="s", markersize=7, linestyle=":",
                            linewidth=2.0, color=color, alpha=0.8, zorder=9,
                            label=f"■ RAND: {env_label}")
                            
                    end_idx_r = np.argmin(r_scales)
                    ax.annotate(f"■ RAND: {env_label}",
                                xy=(r_scales[end_idx_r], r_vals[end_idx_r]),
                                xytext=(8, 0),
                                textcoords="offset points",
                                fontsize=7,
                                color=color,
                                fontweight="bold",
                                alpha=0.8,
                                va="center",
                                ha="left",
                                clip_on=False,
                                zorder=11)"""

code = code.replace(b_old, b_new)

# Add threshold line
t_old = """    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")"""

t_new = """    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")
    
    # Graduation Threshold
    ax.axhline(y=10.0, color="green", linestyle="--", linewidth=2.0, label="Graduation Threshold (10.0s)", zorder=1)"""

code = code.replace(t_old, t_new)

# Expand figure size horizontally to fit the labels on the right margin
code = code.replace('fig, ax = plt.subplots(figsize=(20, 12))', 'fig, ax = plt.subplots(figsize=(24, 12))')

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_postfix_final.py", "w") as f:
    f.write(code)

