import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

def find_csv_files(base_path, category):
    csv_files = []
    cat_path = base_path / category
    if cat_path.exists():
        for exp_dir in cat_path.iterdir():
            if exp_dir.is_dir():
                # Some dirs have subdirs with the same name containing the actual CSVs
                # Example: small_ecmp_round_robin/small_ecmp_round_robin/*.csv
                for csv_file in exp_dir.rglob("*.csv"):
                    if "epoch" in csv_file.name: continue
                    csv_files.append({
                        "path": csv_file,
                        "category": category,
                        "experiment": exp_dir.name,
                        "filename": csv_file.name
                    })
    return csv_files

def parse_scalability(name):
    # Fix for cases where parts are separated by underscore
    parts = name.split("_")
    scale = parts[0]
    # Check if 'very' is part of the scale name
    if scale == "very" and len(parts) > 1:
        scale = "very_large"
        routing = parts[2]
        allocation = "_".join(parts[3:])
    else:
        routing = parts[1]
        allocation = "_".join(parts[2:])
    return {"scale": scale, "routing": routing, "allocation": allocation}

def parse_sensitivity(name):
    parts = name.split("_")
    lr = float(parts[0].replace("lr", ""))
    if "decay" in parts[1]:
        val = float(parts[1].replace("decay", ""))
        key = "decay"
    else:
        val = float(parts[1].replace("er", ""))
        key = "epsilon"
    return {"lr": lr, key: val, "routing": "qrouting", "allocation": "latency_aware"}

def analyze_file(csv_info):
    try:
        print(f"  {csv_info['experiment']} ({csv_info['filename']})")
        # Optimization: only read needed columns
        df = pd.read_csv(csv_info['path'], usecols=lambda x: x in ['total_time', 'latency'])
        lat_col = 'total_time' if 'total_time' in df.columns else 'latency'
        if lat_col not in df.columns:
             return None
        
        lat = df[df[lat_col] > 0][lat_col]
        if lat.empty: return None

        return {
            "mean": lat.mean(),
            "p50": lat.median(),
            "p99": lat.quantile(0.99)
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None

base_path = Path("/scratch/alpine/kamil@colostate.edu/cormem_results")
output_dir = Path("/projects/kamil@colostate.edu/CORMem/evaluation_scripts/plots/aggregated_data")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Scalability
print("Analyzing Scalability...")
scal_files = find_csv_files(base_path, "scalability_analysis")
scal_results = []
for f in scal_files:
    metrics = analyze_file(f)
    if metrics:
        scal_results.append({**parse_scalability(f['experiment']), **metrics})
pd.DataFrame(scal_results).to_csv(output_dir / "scalability_summary.csv", index=False)

# 2. Sensitivity
print("Analyzing Sensitivity...")
sens_files = find_csv_files(base_path, "sensitivity_analysis")
sens_results = []
for f in sens_files:
    metrics = analyze_file(f)
    if metrics:
        sens_results.append({**parse_sensitivity(f['experiment']), **metrics})
pd.DataFrame(sens_results).to_csv(output_dir / "sensitivity_summary.csv", index=False)

print("Done.")
