import os
import re
import glob

# Paths
# Note: Since this runs on the assistant's local environment but targets Alpine paths, 
# we expect it to be copied and run on Alpine.
PROJECT_ROOT = "/projects/kamil@colostate.edu/vidur-rl"
MODEL_ROOT = f"{PROJECT_ROOT}/rl_models/graduation"
TRACK_MAP = f"{PROJECT_ROOT}/graduation_tracks.map"
RESUME_MAP = f"{PROJECT_ROOT}/graduation_resume.map"

def get_latest_checkpoint(model_dir):
    """
    Scans a directory for checkpoints and returns the lowest scale and the best .pt file.
    """
    if not os.path.exists(model_dir):
        return None, None

    # Find all .pt files (excluding epsilon files)
    pt_files = glob.glob(f"{model_dir}/*.pt")
    pt_files = [f for f in pt_files if "_epsilon.pt" not in f]
    
    if not pt_files:
        return None, None

    # Parse scales and request counts
    # Pattern: scale{SCALE}_latest[_req{REQS}].pt
    results = []
    for f in pt_files:
        basename = os.path.basename(f)
        # Match scale
        scale_match = re.search(r"scale([\d.]+)", basename)
        if not scale_match:
            continue
        scale = float(scale_match.group(1))
        
        # Match request count if present
        req_match = re.search(r"req(\d+)", basename)
        reqs = int(req_match.group(1)) if req_match else 0
        
        # Check if file is valid (not 0 bytes)
        try:
            if os.path.getsize(f) > 0:
                results.append({
                    "path": f,
                    "scale": scale,
                    "reqs": reqs,
                    "mtime": os.path.getmtime(f)
                })
        except OSError:
            continue

    if not results:
        return None, None

    # Priority 1: Lowest Scale (most progressed)
    # Priority 2: Highest Request Count (most advanced in that scale)
    # Priority 3: Latest mtime
    results.sort(key=lambda x: (x["scale"], -x["reqs"], -x["mtime"]))
    
    best = results[0]
    return best["scale"], best["path"]

def main():
    if not os.path.exists(TRACK_MAP):
        print(f"Error: Track map {TRACK_MAP} not found.")
        return

    with open(TRACK_MAP, 'r') as f:
        lines = f.readlines()

    resume_entries = []
    
    print(f"{'TASK CONFIG':<30} | {'ORIG SCALE':<10} | {'RESUME SCALE':<12} | {'ACTION'}")
    print("-" * 75)

    for line in lines:
        parts = line.strip().split()
        if not parts or len(parts) < 4:
            continue
            
        config_name = parts[0]
        orig_scale = float(parts[1])
        hidden_dims = parts[2]
        reward_type = parts[3]

        model_class = f"grad_{config_name}_h{hidden_dims.replace(',', '_')}_r{reward_type}"
        model_dir = f"{MODEL_ROOT}/{model_class}"
        
        best_scale, best_path = get_latest_checkpoint(model_dir)
        
        if best_scale is not None:
            resume_scale = best_scale
            action = f"Resume scale {resume_scale}"
        else:
            resume_scale = orig_scale
            action = "Start fresh"

        print(f"{config_name:<30} | {orig_scale:<10} | {resume_scale:<12} | {action}")
        resume_entries.append(f"{config_name} {resume_scale} {hidden_dims} {reward_type}\n")

    with open(RESUME_MAP, 'w') as f:
        f.writelines(resume_entries)
    
    print("-" * 75)
    print(f"Resume map written to: {RESUME_MAP}")

if __name__ == '__main__':
    main()
