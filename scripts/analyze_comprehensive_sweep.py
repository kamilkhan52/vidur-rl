import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Setup
RESULTS_DIR = "results/comprehensive_sweep"
OUTPUT_DIR = "plots_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors for schedulers
PALETTE = {
    "lor": "#1f77b4",        # Blue
    "sticky_lor": "#ff7f0e", # Orange
    "sticky": "#ff7f0e",     # Handle shorthand if it appears
    "random": "#2ca02c"      # Green
}

def parse_run_name(path):
    # Example path: results/comprehensive_sweep/7b_mooncake_toolagent_trace_r2_lor_s8.0/...
    parts = path.split(os.sep)
    # The run name is usually the 3rd part
    run_name = parts[2]
    
    try:
        # 7b_mooncake_toolagent_trace_r2_lor_s8.0
        # Split by _ but be careful with the trace name
        model = run_name.split('_')[0]
        # Cluster size is after '_r'
        cluster_size = run_name.split('_r')[1].split('_')[0]
        # Scheduler is after the cluster size
        # Find index of '_r' + cluster_size
        sched_part = run_name.split(f"_r{cluster_size}_")[1]
        scheduler = sched_part.split('_')[0]
        # Scale is the last part
        scale = run_name.split('_s')[-1]
        
        # Determine workload (trace)
        if "toolagent" in run_name:
            workload = "Toolagent"
        elif "synthetic" in run_name:
            workload = "Synthetic"
        else:
            workload = "Unknown"
            
        return {
            "model": model,
            "cluster_size": int(cluster_size),
            "scheduler": scheduler,
            "scale": float(scale),
            "workload": workload,
            "path": path
        }
    except Exception as e:
        print(f"Failed to parse {run_name}: {e}")
        return None

def gather_data():
    all_runs = []
    csv_paths = glob.glob(os.path.join(RESULTS_DIR, "**/request_metrics.csv"), recursive=True)
    
    print(f"Found {len(csv_paths)} result files. Processing...")
    
    for path in csv_paths:
        meta = parse_run_name(path)
        if not meta:
            continue
            
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
                
            # Calculate QPS
            total_time = df['request_arrived_at'].max() - df['request_arrived_at'].min()
            qps = len(df) / total_time if total_time > 0 else 0
            
            # Metrics
            meta["qps"] = qps
            meta["avg_e2e"] = df['request_e2e_time'].mean()
            meta["avg_prefill"] = df['prefill_e2e_time'].mean()
            meta["avg_scheduling"] = df['request_scheduling_delay'].mean()
            meta["avg_decode"] = (df['request_e2e_time'] - df['prefill_e2e_time']).mean()
            
            all_runs.append(meta)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return pd.DataFrame(all_runs)

def plot_grid(data, metric_col, title_suffix, filename):
    # Grid: Workload (Rows) x Cluster Size (Columns)
    workloads = sorted(data['workload'].unique())
    models = sorted(data['model'].unique())
    
    with PdfPages(os.path.join(OUTPUT_DIR, f"{filename}.pdf")) as pdf:
        for model in models:
            model_data = data[data['model'] == model]
            cluster_sizes = sorted(model_data['cluster_size'].unique())
            
            fig, axes = plt.subplots(len(workloads), len(cluster_sizes), 
                                     figsize=(5*len(cluster_sizes), 4*len(workloads)), 
                                     squeeze=False)
            
            # Set Y-axis limit to 500s to zoom in on usable data
            y_max = 500
            
            for i, workload in enumerate(workloads):
                for j, cluster in enumerate(cluster_sizes):
                    ax = axes[i, j]
                    subset = model_data[(model_data['workload'] == workload) & 
                                        (model_data['cluster_size'] == cluster)]
                    
                    if subset.empty:
                        continue
                        
                    sns.lineplot(data=subset, x="qps", y=metric_col, hue="scheduler", 
                                 palette=PALETTE, marker='o', ax=ax)
                    
                    ax.set_title(f"{workload} | {model} | {cluster} Replicas")
                    ax.set_ylim(0, y_max)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    if i < len(workloads) - 1:
                        ax.set_xlabel("")
                    if j > 0:
                        ax.set_ylabel("")
            
            plt.tight_layout()
            plt.suptitle(f"{model} - {title_suffix}", y=1.02, fontsize=16)
            pdf.savefig(bbox_inches='tight')
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_{model}.png"), bbox_inches='tight')
            plt.close()

def main():
    data = gather_data()
    if data.empty:
        print("No data found!")
        return
        
    print("Generating plots...")
    metrics = [
        ("avg_e2e", "End-to-End Latency (s)", "e2e_latency"),
        ("avg_prefill", "Prefill Latency (s)", "prefill_latency"),
        ("avg_decode", "Decode Latency (s)", "decode_latency"),
        ("avg_scheduling", "Scheduling Delay (s)", "scheduling_delay")
    ]
    
    for col, title, fname in metrics:
        plot_grid(data, col, title, fname)
        
    print(f"Done! Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
