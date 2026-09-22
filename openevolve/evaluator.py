"""
OpenEvolve evaluator for Vidur-RL formulation search.

This module is called by OpenEvolve to evaluate a candidate RL formulation.
It runs a Vidur simulation with the evolved RL config + reward function,
compares against the precomputed STICKY_LOR baseline, and returns a score dict.

Multi-iteration training: Runs NUM_TRAINING_ITERATIONS training iterations
using the full toolagent trace (~23k requests), saving/loading checkpoints
between iterations so the policy converges. Scores on the best training
iteration's median E2E.

Usage (standalone test):
    python openevolve/evaluator.py openevolve/initial_program.py

Score (based on median E2E, robust to tail latency outliers):
    combined_score = (baseline_median - rl_median) / baseline_median
    Positive = RL beats STICKY_LOR, Negative = RL worse.
    Crashes/timeouts → score = -1.0
    Artifacts include mean, p95, p99 for LLM to reason about tail behavior.
"""

import csv
import importlib.util
import json
import math
import os
import shutil
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
TRACE_FILE = str(REPO_ROOT / "data/processed_traces/mooncake_toolagent_trace.csv")
BASELINE_FILE = Path(__file__).parent / "baselines" / "sticky_lor_e2e.json"

# 7b-16R Tool Scale 1.5 — tension zone config, full trace (~23k requests)
FIXED_ARGS = [
    "--replica_config_device", "a100",
    "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
    "--replica_config_tensor_parallel_size", "1",
    "--cluster_config_num_replicas", "16",
    "--replica_scheduler_config_type", "vllm_v1",
    "--request_generator_config_type", "trace",
    "--trace_request_generator_config_trace_file", TRACE_FILE,
    "--trace_request_generator_config_time_scale_factor", "1.5",
    "--cache_config_enable_prefix_caching",
    "--no-metrics_config_enable_chrome_trace",
    "--no-metrics_config_write_json_trace",
]

# Multi-iteration training config
NUM_TRAINING_ITERATIONS = int(os.environ.get("NUM_TRAINING_ITERATIONS", "8"))

# Per-iteration timeout (full toolagent trace ~23k requests)
TIMEOUT_PER_ITERATION = int(os.environ.get("TIMEOUT_PER_ITERATION", "10800"))

# Optional: persist per-iteration artifacts for collapse debugging.
# If set, checkpoints and output dirs are copied under this path.
PERSIST_EVAL_ARTIFACTS_DIR = os.environ.get("PERSIST_EVAL_ARTIFACTS_DIR", "").strip()


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
    src = textwrap.dedent(src)
    Path(plugin_path).write_text(src)


def build_rl_args(config: dict, plugin_path: str, checkpoint_path: str = "",
                  mode: str = "training") -> list:
    """Build CLI args for the RL cache-aware scheduler."""
    args = [
        "--global_scheduler_config_type", "rl_cache_aware",
        "--r_l_cache_aware_global_scheduler_config_mode", mode,
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
            str(config.get("epsilon_decay_steps", 300)),
        "--r_l_cache_aware_global_scheduler_config_lookahead_k",
            str(config.get("lookahead_k", 5)),
        "--r_l_cache_aware_global_scheduler_config_reward_normalization_factor",
            str(config.get("reward_normalization_factor", 100.0)),
    ]
    if checkpoint_path:
        args += [
            "--r_l_cache_aware_global_scheduler_config_model_checkpoint_path",
            checkpoint_path,
        ]
    return args


def _percentile(sorted_vals: list, p: float) -> float:
    """Compute percentile from a pre-sorted list (0-100 scale)."""
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def parse_e2e_stats(output_dir: str) -> dict | None:
    """Parse request_metrics.csv and return latency distribution stats.

    Returns dict with: median, mean, p95, p99, min, max, n_requests.
    Returns None if no data found.
    """
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

    e2e_times.sort()
    n = len(e2e_times)
    return {
        "median": _percentile(e2e_times, 50),
        "mean": sum(e2e_times) / n,
        "p95": _percentile(e2e_times, 95),
        "p99": _percentile(e2e_times, 99),
        "min": e2e_times[0],
        "max": e2e_times[-1],
        "n_requests": n,
    }


def run_simulation(extra_args: list, output_dir: str, timeout: int = 2400):
    """Run vidur.main, return (stats_dict or None, stderr_tail str).

    stats_dict contains: median, mean, p95, p99, min, max, n_requests.
    """
    cmd = [VIDUR_PYTHON, "-m", "vidur.main"] + FIXED_ARGS + extra_args + [
        "--metrics_config_output_dir", output_dir,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        stderr_tail = result.stderr[-3000:] if result.stderr else ""
        if result.returncode != 0:
            return None, f"EXIT {result.returncode}\n{stderr_tail}"
        stats = parse_e2e_stats(output_dir)
        return stats, stderr_tail
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return None, str(e)


def evaluate(program_path: str) -> dict:
    """
    Main evaluation function called by OpenEvolve.

    Runs NUM_TRAINING_ITERATIONS training iterations using the full trace,
    saving/loading checkpoints between iterations. Epsilon state persists across
    iterations via checkpoint, so exploration decays naturally over the full run.

    Scores based on the best training iteration's median E2E (not a separate
    inference pass, which can collapse with partially-trained Q-values).

    Args:
        program_path: Path to the evolved Python file

    Returns:
        Dict with combined_score, training/eval E2E, per-iteration details.
    """
    # Load baseline
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}\n"
            "Run: python openevolve/precompute_baseline.py"
        )
    baseline_data = json.loads(BASELINE_FILE.read_text())
    baseline_median = baseline_data["sticky_lor_median"]

    # Load evolved program
    try:
        mod = load_program(program_path)
        rl_config = mod.get_rl_config()
        compute_reward_fn = mod.compute_reward
    except Exception as e:
        return {
            "combined_score": -1.0,
            "rl_median": -1.0,
            "baseline_median": baseline_median,
            "improvement_pct": -100.0,
            "error": str(e),
        }

    num_iters = NUM_TRAINING_ITERATIONS
    per_iteration_stats = []
    last_stderr = ""
    artifact_run_dir = ""

    if PERSIST_EVAL_ARTIFACTS_DIR:
        program_stem = Path(program_path).stem
        artifact_run_dir = str(Path(PERSIST_EVAL_ARTIFACTS_DIR) / f"{program_stem}_{os.getpid()}")
        os.makedirs(artifact_run_dir, exist_ok=True)
        Path(os.path.join(artifact_run_dir, "program_path.txt")).write_text(program_path)

    with tempfile.TemporaryDirectory() as work_dir:
        plugin_path = os.path.join(work_dir, "reward_plugin.py")
        write_reward_plugin(compute_reward_fn, plugin_path)
        checkpoint_path = os.path.join(work_dir, "model.pt")

        # === Training phase ===
        for iteration in range(num_iters):
            output_dir = os.path.join(work_dir, f"output_iter{iteration}")
            os.makedirs(output_dir, exist_ok=True)

            # First iteration: no checkpoint to load (fresh policy)
            # Subsequent iterations: load from previous checkpoint
            ckpt = checkpoint_path if iteration > 0 and os.path.exists(checkpoint_path) else ""
            rl_args = build_rl_args(rl_config, plugin_path, checkpoint_path=ckpt,
                                    mode="training")

            # Always set checkpoint path for saving (even iter 0)
            if not ckpt:
                rl_args += [
                    "--r_l_cache_aware_global_scheduler_config_model_checkpoint_path",
                    checkpoint_path,
                ]

            stats, stderr = run_simulation(
                rl_args, output_dir, timeout=TIMEOUT_PER_ITERATION
            )
            last_stderr = stderr

            if stats is None:
                # If training iteration fails, abort
                if per_iteration_stats:
                    print(f"  [Eval] Training iter {iteration+1} failed, proceeding to eval with {len(per_iteration_stats)} iters", flush=True)
                    break
                else:
                    return {
                        "combined_score": -1.0,
                        "rl_median": -1.0,
                        "baseline_median": baseline_median,
                        "improvement_pct": -100.0,
                        "per_iteration_stats": [],
                        "num_iterations_completed": 0,
                        "artifact_dir": artifact_run_dir,
                        "stderr": stderr[-500:] if stderr else "",
                    }

            per_iteration_stats.append(stats)

            if artifact_run_dir:
                if os.path.exists(checkpoint_path):
                    shutil.copy2(
                        checkpoint_path,
                        os.path.join(artifact_run_dir, f"model_iter{iteration+1}.pt"),
                    )
                iter_artifact_dir = os.path.join(artifact_run_dir, f"output_iter{iteration+1}")
                shutil.copytree(output_dir, iter_artifact_dir, dirs_exist_ok=True)
                Path(os.path.join(artifact_run_dir, f"stats_iter{iteration+1}.json")).write_text(
                    json.dumps(stats, indent=2)
                )

            print(f"  [Eval] Training iter {iteration+1}/{num_iters}: "
                  f"median={stats['median']:.2f}s  mean={stats['mean']:.2f}s  "
                  f"p95={stats['p95']:.2f}s  p99={stats['p99']:.2f}s", flush=True)

    # Score based on the best training iteration median.
    # We use the best (not last) because later iterations with low epsilon
    # can collapse if Q-values are wrong. The best iteration reflects peak
    # policy capability. The baseline was also run with a deterministic policy.
    iter_medians = [s["median"] for s in per_iteration_stats]
    best_idx = iter_medians.index(min(iter_medians))
    best_stats = per_iteration_stats[best_idx]
    best_median = best_stats["median"]
    improvement = (baseline_median - best_median) / baseline_median
    score = max(-1.0, improvement)

    print(f"  [Eval] Best iteration: {best_idx+1} (median={best_median:.2f}s, "
          f"improvement={improvement*100:.1f}%)", flush=True)

    return {
        "combined_score": score,
        "rl_median": best_median,
        "rl_mean": best_stats["mean"],
        "rl_p95": best_stats["p95"],
        "rl_p99": best_stats["p99"],
        "baseline_median": baseline_median,
        "improvement_pct": improvement * 100,
        "per_iteration_medians": iter_medians,
        "best_iteration": best_idx + 1,
        "num_iterations_completed": len(per_iteration_stats),
        "artifact_dir": artifact_run_dir,
        "stderr": last_stderr[-500:] if last_stderr else "",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <program_path>")
        sys.exit(1)

    program_path = sys.argv[1]
    print(f"Evaluating: {program_path}")
    print(f"Config: 7b-16R Tool Scale 1.5, full trace (~23k req), "
          f"{NUM_TRAINING_ITERATIONS} training iters, scored on best iter median")
    result = evaluate(program_path)
    print(json.dumps(result, indent=2))
