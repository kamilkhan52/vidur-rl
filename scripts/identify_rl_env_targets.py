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
    
    # Filter for E2E between 10s and 250s
    # We want environments where at least one scheduler falls in this range
    # OR environments where the 'rivals' (lor/sticky_lor) are in this range
    
    print("=== Environments satisfying 10s < E2E < 250s (Tension Zone) ===")
    
    # Group by Workload, Model, Replicas, Scale
    results = []
    for (workload, model, replicas, scale), group in df_all.groupby(['workload', 'model', 'cluster_size', 'scale']):
        min_e2e = group['avg_e2e'].min()
        max_e2e = group['avg_e2e'].max()
        
        # Check if the environment is "interesting" (in the 10s-250s range)
        if (min_e2e < 250) and (max_e2e > 10):
            # Find specific scheduler stats for this point
            lor_e2e = group[group['scheduler'] == 'lor']['avg_e2e'].values
            slor_e2e = group[group['scheduler'] == 'sticky_lor']['avg_e2e'].values
            
            results.append({
                "Workload": workload,
                "Model": model,
                "R": replicas,
                "Scale": scale,
                "Min_E2E": round(min_e2e, 2),
                "Max_E2E": round(max_e2e, 2),
                "LOR_E2E": round(lor_e2e[0], 2) if len(lor_e2e) > 0 else "N/A",
                "SLOR_E2E": round(slor_e2e[0], 2) if len(slor_e2e) > 0 else "N/A"
            })
    
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
