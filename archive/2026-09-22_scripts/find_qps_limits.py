import os
import glob
import pandas as pd

RESULTS_DIR = "results/comprehensive_sweep"

def parse_run_name(run_name):
    try:
        model = run_name.split('_')[0]
        cluster_size = run_name.split('_r')[1].split('_')[0]
        sched_part = run_name.split(f"_r{cluster_size}_")[1]
        scheduler = sched_part.split('_')[0]
        scale = run_name.split('_s')[-1]
        workload = "Toolagent" if "toolagent" in run_name else "Synthetic"
        return {
            "model": model,
            "cluster_size": int(cluster_size),
            "scheduler": scheduler,
            "scale": float(scale),
            "workload": workload
        }
    except:
        return None

def main():
    csv_paths = glob.glob(os.path.join(RESULTS_DIR, "**/request_metrics.csv"), recursive=True)
    all_runs = []
    
    for path in csv_paths:
        run_name = path.split(os.sep)[2]
        meta = parse_run_name(run_name)
        if not meta: continue
            
        df = pd.read_csv(path)
        if df.empty: continue
        
        meta["avg_e2e"] = df['request_e2e_time'].mean()
        all_runs.append(meta)
    
    df_all = pd.DataFrame(all_runs)
    
    # Group by everything except scheduler and scale to find the thresholds
    # We want to find for each (Workload, Model, ClusterSize, Scale) if ALL schedulers are > 1000s
    groups = df_all.groupby(['workload', 'model', 'cluster_size', 'scale'])['avg_e2e'].max()
    
    print("=== Suggested QPS Scale Limits (Max Scale where at least one scheduler is < 1000s) ===")
    results = []
    for (workload, model, cluster), group in df_all.groupby(['workload', 'model', 'cluster_size']):
        # Find the max scale for this group where the MIN latency across schedulers is < 500
        valid_scales = []
        for scale in sorted(group['scale'].unique()):
            scale_group = group[group['scale'] == scale]
            if scale_group['avg_e2e'].min() < 500:
                valid_scales.append(scale)
        
        max_scale = max(valid_scales) if valid_scales else 0
        results.append({
            "Workload": workload,
            "Model": model,
            "Replicas": cluster,
            "Max Useful Scale": max_scale
        })
    
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
