import pandas as pd
import json
from pathlib import Path

csv_path = Path("/projects/kamil@colostate.edu/CORMem/evaluation_scripts/plots/aggregated_data/scalability_summary.csv")
sens_path = Path("/projects/kamil@colostate.edu/CORMem/evaluation_scripts/plots/aggregated_data/sensitivity_summary.csv")
output_path = Path("/projects/kamil@colostate.edu/CORMem/evaluation_scripts/latency_analysis_results.json")

def convert():
    results = []
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            results.append({
                "category": "scalability_analysis",
                "scale": row['scale'],
                "routing": row['routing'],
                "allocation": row['allocation'],
                "latency_mean": row['mean'],
                "latency_p50": row['p50'],
                "latency_p99": row['p99']
            })
            
    if sens_path.exists():
        df = pd.read_csv(sens_path)
        for _, row in df.iterrows():
            results.append({
                "category": "sensitivity_analysis",
                "lr": row['lr'],
                "decay": row.get('decay', 0),
                "epsilon": row.get('epsilon', 0),
                "routing": row['routing'],
                "allocation": row['allocation'],
                "latency_mean": row['mean'],
                "latency_p50": row['p50'],
                "latency_p99": row['p99']
            })
            
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    convert()
