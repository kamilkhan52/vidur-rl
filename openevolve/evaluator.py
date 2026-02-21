"""
OpenEvolve evaluator for Vidur-RL formulation search.

This module is called by OpenEvolve to evaluate a candidate RL formulation.
It runs a Vidur simulation with the evolved RL config + reward function,
compares against the precomputed STICKY_LOR baseline, and returns a score dict.

Usage (standalone test):
    python openevolve/evaluator.py openevolve/initial_program.py

Score:
    combined_score = improvement = (baseline_e2e - rl_e2e) / baseline_e2e
    Positive = RL beats STICKY_LOR, Negative = RL worse.
    Crashes/timeouts → score = -1.0
"""

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# --- Paths ---
REPO_ROOT = Path(__file__).parent.parent

# Use the venv Python if available (for correct dependency resolution)
_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
VIDUR_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
TRACE_FILE = str(REPO_ROOT / "data/processed_traces/mooncake_synthetic_trace.csv")
BASELINE_FILE = Path(__file__).parent / "baselines" / "sticky_lor_e2e.json"

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


def load_program(program_path: str):
    """Dynamically import the evolved program module."""
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_reward_plugin(compute_reward_fn, plugin_path: str):
    """
    Write a standalone Python file containing the compute_reward function.
    This is passed as reward_plugin_path to Vidur.
    """
    import inspect
    src = inspect.getsource(compute_reward_fn)
    # Strip leading indentation in case function was defined inside a class
    src = textwrap.dedent(src)
    Path(plugin_path).write_text(src)


def build_rl_args(config: dict, plugin_path: str) -> list:
    """Build CLI args for the RL cache-aware scheduler."""
    return [
        "--global_scheduler_config_type", "rl_cache_aware",
        "--r_l_cache_aware_global_scheduler_config_mode", "training",
        "--r_l_cache_aware_global_scheduler_config_reward_plugin_path", plugin_path,
        "--r_l_cache_aware_global_scheduler_config_hidden_dims",
            config.get("hidden_dims", "1024,1024,512"),
        "--r_l_cache_aware_global_scheduler_config_learning_rate",
            str(config.get("learning_rate", 0.0003)),
        "--r_l_cache_aware_global_scheduler_config_batch_size",
            str(config.get("batch_size", 64)),
        "--r_l_cache_aware_global_scheduler_config_update_frequency",
            str(config.get("update_frequency", 100)),
        "--r_l_cache_aware_global_scheduler_config_target_network_update_frequency",
            str(config.get("target_network_update_frequency", 1000)),
        "--r_l_cache_aware_global_scheduler_config_replay_buffer_size",
            str(config.get("replay_buffer_size", 10000)),
        "--r_l_cache_aware_global_scheduler_config_epsilon_start",
            str(config.get("epsilon_start", 1.0)),
        "--r_l_cache_aware_global_scheduler_config_epsilon_end",
            str(config.get("epsilon_end", 0.05)),
        "--r_l_cache_aware_global_scheduler_config_epsilon_decay_steps",
            str(config.get("epsilon_decay_steps", 1000)),
        "--r_l_cache_aware_global_scheduler_config_lookahead_k",
            str(config.get("lookahead_k", 5)),
        "--r_l_cache_aware_global_scheduler_config_reward_normalization_factor",
            str(config.get("reward_normalization_factor", 100.0)),
    ]


def parse_mean_e2e(output_dir: str):
    """Parse request_metrics.csv from the vidur output directory. Returns float or None."""
    base = Path(output_dir)
    csv_files = list(base.rglob("request_metrics.csv"))
    if not csv_files:
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
        return None

    return sum(e2e_times) / len(e2e_times)


def run_simulation(extra_args: list, output_dir: str):
    """Run vidur.main, return (mean_e2e or None, stderr_tail str)."""
    cmd = [VIDUR_PYTHON, "-m", "vidur.main"] + FIXED_ARGS + extra_args + [
        "--metrics_config_output_dir", output_dir,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(REPO_ROOT),
        )
        stderr_tail = result.stderr[-3000:] if result.stderr else ""
        if result.returncode != 0:
            return None, f"EXIT {result.returncode}\n{stderr_tail}"
        mean_e2e = parse_mean_e2e(output_dir)
        return mean_e2e, stderr_tail
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {TIMEOUT_SECONDS}s"
    except Exception as e:
        return None, str(e)


def evaluate(program_path: str) -> dict:
    """
    Main evaluation function called by OpenEvolve.

    Args:
        program_path: Path to the evolved Python file (written by OpenEvolve to a temp file)

    Returns:
        Dict with:
          - combined_score: primary optimisation target (improvement over STICKY_LOR)
          - rl_e2e, baseline_e2e, improvement_pct: for LLM feedback artifacts
    """
    # Load baseline
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}\n"
            "Run: python openevolve/precompute_baseline.py"
        )
    baseline_data = json.loads(BASELINE_FILE.read_text())
    baseline_e2e = baseline_data["sticky_lor_e2e"]

    # Load evolved program
    try:
        mod = load_program(program_path)
        rl_config = mod.get_rl_config()
        compute_reward_fn = mod.compute_reward
    except Exception as e:
        return {
            "combined_score": -1.0,
            "rl_e2e": -1.0,
            "baseline_e2e": baseline_e2e,
            "improvement_pct": -100.0,
            "error": str(e),
        }

    with tempfile.TemporaryDirectory() as plugin_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        plugin_path = os.path.join(plugin_dir, "reward_plugin.py")
        write_reward_plugin(compute_reward_fn, plugin_path)

        rl_args = build_rl_args(rl_config, plugin_path)
        rl_e2e, stderr = run_simulation(rl_args, output_dir)

    if rl_e2e is None:
        return {
            "combined_score": -1.0,
            "rl_e2e": -1.0,
            "baseline_e2e": baseline_e2e,
            "improvement_pct": -100.0,
            "stderr": stderr[-500:] if stderr else "",
        }

    improvement = (baseline_e2e - rl_e2e) / baseline_e2e
    score = max(-1.0, improvement)

    return {
        "combined_score": score,
        "rl_e2e": rl_e2e,
        "baseline_e2e": baseline_e2e,
        "improvement_pct": improvement * 100,
        "stderr": stderr[-500:] if stderr else "",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <program_path>")
        sys.exit(1)

    program_path = sys.argv[1]
    print(f"Evaluating: {program_path}")
    result = evaluate(program_path)
    print(json.dumps(result, indent=2))
