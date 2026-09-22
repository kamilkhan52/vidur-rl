import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

def analyze_sweep():
    results_dir = "results/sweep_Feb03"
    data = []

    # Iterate through all experiment folders
    for exp_dir in os.listdir(results_dir):
        path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(path):
            continue
            
        # Parse name: mooncake_toolagent_trace_r4_sticky_lor_s8.0
        parts = exp_dir.split("_")
        # Find scale and scheduler
        # Note: scale is at the end, scheduler is before 's'
        scale = float(exp_dir.split("_s")[-1])
        scheduler = "sticky_lor" if "sticky_lor" in exp_dir else "lor"
        num_replicas = int(exp_dir.split("_r")[1].split("_")[0])
        trace = exp_dir.split("_r")[0]

        # Find the latest metrics file in subdirectories
        metric_files = glob.glob(os.path.join(path, "**/request_metrics.csv"), recursive=True)
        if not metric_files:
            continue
            
        # Get the latest one
        latest_metrics = max(metric_files, key=os.path.getmtime)
        df = pd.read_csv(latest_metrics)
        
        # Calculate stats
        mean_e2e = df["e2e_time"].mean()
        p99_e2e = df["e2e_time"].quantile(0.99)
        completed_requests = len(df)
        
        # QPS (approximate from scale)
        # We'll just use 1/scale as a relative QPS for the x-axis
        relative_qps = 1.0 / scale
        
        data.append({
            "trace": trace,
            "replicas": num_replicas,
            "scheduler": scheduler,
            "scale": scale,
            "relative_qps": relative_qps,
            "mean_e2e": mean_e2e,
            "p99_e2e": p99_e2e,
            "completed": completed_requests
        })

    full_df = pd.DataFrame(data)
    
    # Generate Plots
    for trace in full_df["trace"].unique():
        for replica in full_df["replicas"].unique():
            subset = full_df[(full_df["trace"] == trace) & (full_df["replicas"] == replica)]
            if subset.empty:
                continue
                
            plt.figure(figsize=(10, 6))
            for scheduler in subset["scheduler"].unique():
                sched_subset = subset[subset["scheduler"] == scheduler].sort_values("relative_qps")
                plt.plot(sched_subset["relative_qps"], sched_subset["mean_e2e"], marker='o', label=f"{scheduler}")
            
            plt.title(f"Mean Latency vs Relative QPS ({trace}, {replica} Replicas)")
            plt.xlabel("Relative QPS (1/Scale)")
            plt.ylabel("Mean E2E Latency (s)")
            plt.legend()
            plt.grid(True)
            plt.yscale('log') # Log scale helps see the 'break'
            
            plot_path = f"analysis_{trace}_r{replica}.png"
            plt.savefig(plot_path)
            print(f"Generated {plot_path}")
            plt.close()

if __name__ == "__main__":
    analyze_sweep()
