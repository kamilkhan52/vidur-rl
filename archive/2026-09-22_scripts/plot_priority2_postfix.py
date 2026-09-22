import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

os.makedirs("/home/kamil/vidur-rl-working/plots/EXP02_Baselines_Postfix", exist_ok=True)

df = pd.read_csv("/home/kamil/vidur-rl-working/plots/priority2_all.csv")

# Ensure it's grouped right
agg_df = df.groupby(["scheduler", "model", "replicas", "trace", "scale"])["mean_e2e"].mean().reset_index()

traces = ["Synth", "Tool", "Conv"]
schedulers = ['RANDOM', 'ROUND_ROBIN', 'LOR', 'STICKY_LOR']

for trace in traces:
    trace_data = agg_df[agg_df["trace"] == trace]
    if trace_data.empty: continue
    
    # Let's plot the scheduler performance grouped by config (7b-8R, 7b-16R, etc)
    configs = trace_data[["model", "replicas"]].drop_duplicates().values.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Postfix Heuristic Baselines: {trace} Trace", fontsize=16)
    
    for i, (model, reps) in enumerate(configs):
        ax = axes[i//2, i%2]
        
        cfg_data = trace_data[(trace_data["model"] == model) & (trace_data["replicas"] == reps)]
        
        for sched in schedulers:
            s_data = cfg_data[cfg_data["scheduler"] == sched].sort_values("scale", ascending=False)
            if not s_data.empty:
                ax.plot(s_data["scale"], s_data["mean_e2e"], marker='o', label=sched)
                
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_title(f"{model}-{reps}")
        ax.set_xlabel("Scale")
        ax.set_ylabel("Mean E2E (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"/home/kamil/vidur-rl-working/plots/EXP02_Baselines_Postfix/{trace}_baselines.png")
    plt.close()

print("Generated baseline plots in /home/kamil/vidur-rl-working/plots/EXP02_Baselines_Postfix")
