"""
Precompute baseline E2E times for STICKY_LOR and RANDOM schedulers.

Run this once before starting the OpenEvolve search:
    python openevolve/precompute_baseline.py

Output: openevolve/baselines/sticky_lor_e2e.json
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Fixed evaluation setup ---
REPO_ROOT = Path(__file__).parent.parent

# Use the venv Python if available (for correct dependency resolution)
_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
VIDUR_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
TRACE_FILE = str(REPO_ROOT / "data/processed_traces/mooncake_synthetic_trace.csv")
BASELINE_DIR = Path(__file__).parent / "baselines"

FIXED_ARGS = [
    "--replica_config_device", "a100",
    "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
    "--replica_config_tensor_parallel_size", "1",
    "--cluster_config_num_replicas", "8",
    "--replica_scheduler_config_type", "vllm_v1",
    "--request_generator_config_type", "trace",
    "--trace_request_generator_config_trace_file", TRACE_FILE,
    "--trace_request_generator_config_time_scale_factor", "6.0",
    "--trace_request_generator_config_num_requests", "2000",
    "--cache_config_enable_prefix_caching",
    "--no-metrics_config_enable_chrome_trace",
    "--no-metrics_config_write_json_trace",
]

TIMEOUT_SECONDS = 600


def run_vidur(extra_args: list, output_dir: str) -> float | None:
    """Run vidur.main with given args, return mean E2E time or None on failure."""
    cmd = [VIDUR_PYTHON, "-m", "vidur.main"] + FIXED_ARGS + extra_args + [
        "--metrics_config_output_dir", output_dir,
    ]
    print(f"Running: {' '.join(cmd[:10])} ...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            print(f"ERROR (exit {result.returncode}):\n{result.stderr[-2000:]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None

    return parse_mean_e2e(output_dir)


def parse_mean_e2e(output_dir: str) -> float | None:
    """Parse request_metrics.csv from the vidur output directory."""
    # Vidur writes metrics into a timestamped subdir
    base = Path(output_dir)
    csv_files = list(base.rglob("request_metrics.csv"))
    if not csv_files:
        print(f"ERROR: No request_metrics.csv found under {output_dir}")
        return None

    csv_file = sorted(csv_files)[-1]  # latest if multiple
    e2e_times = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e2e_times.append(float(row["request_e2e_time"]))
            except (KeyError, ValueError):
                pass

    if not e2e_times:
        print(f"ERROR: No e2e_time values in {csv_file}")
        return None

    mean_e2e = sum(e2e_times) / len(e2e_times)
    print(f"  → mean E2E: {mean_e2e:.4f}s  (n={len(e2e_times)} requests)")
    return mean_e2e


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BASELINE_DIR / "sticky_lor_e2e.json"

    if output_path.exists():
        existing = json.loads(output_path.read_text())
        print(f"Baseline already exists: {output_path}")
        print(json.dumps(existing, indent=2))
        return

    baselines = {}

    # --- STICKY_LOR ---
    print("\n=== Running STICKY_LOR ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        e2e = run_vidur(["--global_scheduler_config_type", "sticky_lor"], tmpdir)
    if e2e is None:
        print("STICKY_LOR run failed. Check errors above.")
        sys.exit(1)
    baselines["sticky_lor_e2e"] = e2e
    print(f"STICKY_LOR baseline: {e2e:.4f}s")

    # --- RANDOM (floor reference) ---
    print("\n=== Running RANDOM ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        e2e_random = run_vidur(["--global_scheduler_config_type", "random"], tmpdir)
    if e2e_random is not None:
        baselines["random_e2e"] = e2e_random
        print(f"RANDOM baseline: {e2e_random:.4f}s")
    else:
        print("RANDOM run failed (non-critical).")

    # --- Save ---
    output_path.write_text(json.dumps(baselines, indent=2))
    print(f"\nSaved to: {output_path}")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
