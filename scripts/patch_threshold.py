import re

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_postfix.py", "r") as f:
    code = f.read()

replacement = """
    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")

    # Graduation Threshold
    ax.axhline(y=10.0, color="green", linestyle="--", linewidth=2.0, label="Graduation Threshold (10.0s)", zorder=1)
"""

code = code.replace("""
    # Tension zone shading
    ax.axvspan(0.5, 4.0, alpha=0.15, color="red", label="Tension Zone (Scale ≤ 4.0)")
""", replacement)

with open("/home/kamil/vidur-rl-working/scripts/plot_graduation_progress_postfix_threshold.py", "w") as f:
    f.write(code)
