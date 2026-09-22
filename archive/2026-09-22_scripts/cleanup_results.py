import os
import shutil

results_dir = "/projects/kamil@colostate.edu/vidur-rl/results/comprehensive_sweep"

if not os.path.exists(results_dir):
    print("Results directory not found.")
    exit()

print(f"Cleaning up {results_dir}...")

for run_dir in os.listdir(results_dir):
    run_path = os.path.join(results_dir, run_dir)
    if not os.path.isdir(run_path):
        continue
        
    # Find all timestamped subdirectories
    for item in os.listdir(run_path):
        item_path = os.path.join(run_path, item)
        if os.path.isdir(item_path):
            has_metrics = False
            # Check if this specific timestamped folder has completion files
            for root, dirs, files in os.walk(item_path):
                if "request_metrics.csv" in files or "metrics.json" in files:
                    has_metrics = True
                    break
            
            if not has_metrics:
                print(f"Removing partial/failed run: {item_path}")
                shutil.rmtree(item_path)

print("Cleanup complete.")
