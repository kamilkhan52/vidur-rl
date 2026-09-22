"""
Precompute baseline E2E stats for STICKY_LOR and RANDOM schedulers.

Run this once before starting the OpenEvolve search:
    python openevolve/precompute_baseline.py

Uses the same config as evaluator.py: 7b-16R Tool Scale 1.5, full trace.
Stores median, mean, p95, p99 — evaluator scores on median.

Output: openevolve/baselines/sticky_lor_e2e.json
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Fixed evaluation setup (must match evaluator.py) ---
REPO_ROOT = Path(__file__).parent.parent

# Use the venv Python if available (for correct dependency resolution)
_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
VIDUR_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
TRACE_FILE = str(REPO_ROOT / "data/processed_traces/mooncake_toolagent_trace.csv")
BASELINE_DIR = Path(__file__).parent / "baselines"

# 7b-16R Tool Scale 1.5 — must match evaluator.py FIXED_ARGS
FIXED_ARGS = [
    "--replica_config_device", "a100",
    "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
    "--replica_config_tensor_parallel_size", "1",
    "--cluster_config_num_replicas", "16",
    "--replica_scheduler_config_type", "vllm_v1",
    "--request_generator_config_type", "trace",
    "--trace_request_generator_config_trace_file", TRACE_FILE,
    # KV-capacity filter: 7b/TP1 replica holds 122752 tokens; larger requests wedge it
    "--trace_request_generator_config_max_tokens", "122752",
    "--trace_request_generator_config_time_scale_factor", "1.5",
    "--cache_config_enable_prefix_caching",
    "--no-metrics_config_enable_chrome_trace",
    "--no-metrics_config_write_json_trace",
]

TIMEOUT_SECONDS = 10800  # 3 hours


def _percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from a pre-sorted list (0-100 scale)."""
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def run_vidur(extra_args: list, output_dir: str) -> dict | None:
    """Run vidur.main with given args, return stats dict or None on failure."""
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

    return parse_e2e_stats(output_dir)


def parse_e2e_stats(output_dir: str) -> dict | None:
    """Parse request_metrics.csv and return latency distribution stats."""
    base = Path(output_dir)
    csv_files = list(base.rglob("request_metrics.csv"))
    if not csv_files:
        print(f"ERROR: No request_metrics.csv found under {output_dir}")
        return None

    csv_file = sorted(csv_files)[-1]
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

    e2e_times.sort()
    n = len(e2e_times)
    stats = {
        "median": _percentile(e2e_times, 50),
        "mean": sum(e2e_times) / n,
        "p95": _percentile(e2e_times, 95),
        "p99": _percentile(e2e_times, 99),
        "min": e2e_times[0],
        "max": e2e_times[-1],
        "n_requests": n,
    }
    print(f"  → median={stats['median']:.4f}s  mean={stats['mean']:.4f}s  "
          f"p95={stats['p95']:.4f}s  p99={stats['p99']:.4f}s  (n={n})")
    return stats


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BASELINE_DIR / "sticky_lor_e2e.json"

    if output_path.exists():
        existing = json.loads(output_path.read_text())
        print(f"Existing baseline: {output_path}")
        print(json.dumps(existing, indent=2))
        print("\nRecomputing with current config (7b-16R Tool Scale 1.5, full trace)...")

    baselines = {"config": "7b-16R-Tool-Scale1.5-FullTrace"}

    # --- STICKY_LOR ---
    print("\n=== Running STICKY_LOR (7b-16R Tool Scale 1.5, full trace) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        stats = run_vidur(["--global_scheduler_config_type", "sticky_lor"], tmpdir)
    if stats is None:
        print("STICKY_LOR run failed. Check errors above.")
        sys.exit(1)
    baselines["sticky_lor_median"] = stats["median"]
    baselines["sticky_lor_mean"] = stats["mean"]
    baselines["sticky_lor_p95"] = stats["p95"]
    baselines["sticky_lor_p99"] = stats["p99"]
    baselines["sticky_lor_n_requests"] = stats["n_requests"]
    print(f"STICKY_LOR baseline median: {stats['median']:.4f}s")

    # --- RANDOM (floor reference) ---
    print("\n=== Running RANDOM ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_random = run_vidur(["--global_scheduler_config_type", "random"], tmpdir)
    if stats_random is not None:
        baselines["random_median"] = stats_random["median"]
        baselines["random_mean"] = stats_random["mean"]
        baselines["random_p95"] = stats_random["p95"]
        baselines["random_p99"] = stats_random["p99"]
        print(f"RANDOM baseline median: {stats_random['median']:.4f}s")
    else:
        print("RANDOM run failed (non-critical).")

    # --- Save ---
    output_path.write_text(json.dumps(baselines, indent=2))
    print(f"\nSaved to: {output_path}")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
