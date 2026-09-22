import sys

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_with_baselines2_threshold.py", "r") as f:
    code = f.read()

# We need to annotate the best heuristic and random baseline lines too.
old_b_plot = """                    # Add baseline traces to plot
                    ax.plot(b_scales, b_best_vals, marker="x", markersize=8, linestyle=":", 
                            linewidth=1.5, color=color, alpha=0.8, 
                            label=f"Best Heuristic" if not baseline_plotted else "_nolegend_")
                            
                    valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                    if valid_random:
                        r_scales, r_vals = zip(*valid_random)
                        ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle="--",
                                linewidth=1.5, color=color, alpha=0.6,
                                label=f"Random" if not baseline_plotted else "_nolegend_")"""

new_b_plot = """                    # Add baseline traces to plot
                    ax.plot(b_scales, b_best_vals, marker="x", markersize=8, linestyle=":", 
                            linewidth=1.5, color=color, alpha=0.8, 
                            label=f"Best Heuristic" if not baseline_plotted else "_nolegend_")
                    
                    if b_scales:
                        end_idx = np.argmin(b_scales)
                        ax.annotate(label + " (Best)",
                                    xy=(b_scales[end_idx], b_best_vals[end_idx]),
                                    xytext=(8, 0),
                                    textcoords="offset points",
                                    fontsize=6,
                                    color=color,
                                    alpha=0.6,
                                    va="center",
                                    ha="left",
                                    clip_on=False,
                                    zorder=9)
                            
                    valid_random = [(s, v) for s, v in zip(b_scales, b_random_vals) if v is not None]
                    if valid_random:
                        r_scales, r_vals = zip(*valid_random)
                        ax.plot(r_scales, r_vals, marker="s", markersize=6, linestyle="--",
                                linewidth=1.5, color=color, alpha=0.6,
                                label=f"Random" if not baseline_plotted else "_nolegend_")
                        
                        end_idx_r = np.argmin(r_scales)
                        ax.annotate(label + " (Random)",
                                    xy=(r_scales[end_idx_r], r_vals[end_idx_r]),
                                    xytext=(8, 0),
                                    textcoords="offset points",
                                    fontsize=6,
                                    color=color,
                                    alpha=0.5,
                                    va="center",
                                    ha="left",
                                    clip_on=False,
                                    zorder=8)"""

if old_b_plot not in code:
    print("Could not find the baseline plot section to patch.")
    sys.exit(1)

code = code.replace(old_b_plot, new_b_plot)

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_with_baselines2_threshold.py", "w") as f:
    f.write(code)
print("Patched successfully.")
