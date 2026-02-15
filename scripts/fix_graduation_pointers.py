import os
import re
import glob

# Paths
PROJECT_ROOT = "/projects/kamil@colostate.edu/vidur-rl"
MODEL_ROOT = f"{PROJECT_ROOT}/rl_models/graduation"
TRACK_MAP = f"{PROJECT_ROOT}/graduation_resume.map"

def fix_pointers_for_class(model_dir):
    """
    Finds the best checkpoint for each scale and ensures the _latest.pt 
    link/copy points to it.
    """
    if not os.path.exists(model_dir):
        return

    pt_files = glob.glob(f"{model_dir}/*.pt")
    pt_files = [f for f in pt_files if "_epsilon.pt" not in f and "_latest" not in f]
    
    # Group by scale
    scales = {}
    for f in pt_files:
        basename = os.path.basename(f)
        scale_match = re.search(r"scale([\d.]+)", basename)
        if not scale_match: continue
        scale = scale_match.group(1)
        
        req_match = re.search(r"req(\d+)", basename)
        reqs = int(req_match.group(1)) if req_match else 0
        
        if os.path.getsize(f) > 0:
            if scale not in scales: scales[scale] = []
            scales[scale].append({"path": f, "reqs": reqs, "mtime": os.path.getmtime(f)})

    for scale, versions in scales.items():
        # Sort versions to find the absolute best for this scale
        versions.sort(key=lambda x: (-x["reqs"], -x["mtime"]))
        best_v = versions[0]
        
        target_latest = f"{model_dir}/scale{scale}_latest.pt"
        target_epsilon = target_latest.replace(".pt", "_epsilon.pt")
        
        source_latest = best_v["path"]
        source_epsilon = source_latest.replace(".pt", "_epsilon.pt")

        # Update the latest pointer
        print(f"  [Scale {scale}] Linking {os.path.basename(source_latest)} -> {os.path.basename(target_latest)}")
        
        # We use copy to be safe with script assumptions
        if os.path.exists(target_latest):
            os.remove(target_latest)
        os.system(f"cp {source_latest} {target_latest}")
        
        if os.path.exists(source_epsilon):
            if os.path.exists(target_epsilon):
                os.remove(target_epsilon)
            os.system(f"cp {source_epsilon} {target_epsilon}")

def main():
    if not os.path.exists(TRACK_MAP):
        print("Error: Resume map not found.")
        return

    with open(TRACK_MAP, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        config_name = parts[0]
        hidden_dims = parts[2]
        reward_type = parts[3]
        model_class = f"grad_{config_name}_h{hidden_dims.replace(',', '_')}_r{reward_type}"
        
        print(f"Processing {model_class}...")
        fix_pointers_for_class(f"{MODEL_ROOT}/{model_class}")

if __name__ == '__main__':
    main()
