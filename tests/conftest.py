"""Shared helpers for the Vidur RL regression tests.

The end-to-end tests replay a tiny slice of the Mooncake synthetic trace through
the real simulator (about ten seconds each on a laptop). They need the predictor
cache under ``cache/`` and the processed trace under ``data/processed_traces``.
"""

import atexit
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_TRACE = REPO_ROOT / "data" / "processed_traces" / "mooncake_synthetic_trace.csv"
KV_CAPACITY_7B = 122752  # tokens per Llama-2-7b replica on an 80 GB A100 (TP=1)


def _require_data():
    if not SYNTH_TRACE.exists():
        pytest.skip(f"trace not available: {SYNTH_TRACE}")
    if not (REPO_ROOT / "cache").exists():
        pytest.skip("execution-time predictor cache not available")


@pytest.fixture(scope="session")
def tiny_trace(tmp_path_factory) -> Path:
    """First 24 synthetic requests that fit a 7b replica's KV cache."""
    _require_data()
    df = pd.read_csv(SYNTH_TRACE)
    df = df[(df.num_prefill_tokens + df.num_decode_tokens) <= KV_CAPACITY_7B].head(24)
    path = tmp_path_factory.mktemp("trace") / "tiny_trace.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def tiny_trace_with_oversize(tmp_path_factory, tiny_trace) -> Path:
    """The tiny trace plus one request larger than the whole KV cache."""
    df = pd.read_csv(tiny_trace)
    total = 130_016  # > 122752; multiple of the 16-token block size
    row = df.iloc[0].copy()
    row["arrived_at"] = float(df.arrived_at.iloc[5])
    row["num_prefill_tokens"] = total - 16
    row["num_decode_tokens"] = 16
    row["block_hash_ids"] = "[" + ", ".join(str(10_000_000 + i) for i in range(total // 16)) + "]"
    df = pd.concat([df.iloc[:6], pd.DataFrame([row]), df.iloc[6:]]).reset_index(drop=True)
    path = tmp_path_factory.mktemp("trace") / "tiny_trace_oversize.csv"
    df.to_csv(path, index=False)
    return path


def base_args(trace: Path, out_dir: Path, num_replicas: int = 2, **overrides):
    """CLI argument list for a small vLLM-v1 cluster replaying ``trace``."""
    args = {
        "replica_config_device": "a100",
        "replica_config_model_name": "meta-llama/Llama-2-7b-hf",
        "replica_config_tensor_parallel_size": "1",
        "cluster_config_num_replicas": str(num_replicas),
        "replica_scheduler_config_type": "vllm_v1",
        "request_generator_config_type": "trace",
        "trace_request_generator_config_trace_file": str(trace),
        "trace_request_generator_config_time_scale_factor": "20.0",
        "metrics_config_output_dir": str(out_dir),
        "seed": "7",
    }
    args.update(overrides)
    argv = ["vidur"]
    for k, v in args.items():
        if v is None:
            continue
        argv.append(f"--{k}")
        if v != "":
            argv.append(str(v))
    argv += [
        "--cache_config_enable_prefix_caching",
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_metrics",
    ]
    return argv


def build_simulator(argv):
    """Create a Simulator from CLI-style argv without touching the real sys.argv."""
    from vidur.config import SimulationConfig
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    saved = sys.argv
    sys.argv = list(argv)
    try:
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = saved
    set_seeds(config.seed)
    simulator = Simulator(config)
    # The simulator registers its output writer with atexit; tests do not want
    # dozens of writers firing at interpreter exit.
    atexit.unregister(simulator._write_output)
    return simulator


def run_simulator(argv):
    """Build, run to completion, apply the end-of-run RL hook, return the simulator."""
    from vidur.scheduler.global_scheduler.rl_cache_aware_global_scheduler import (
        RLCacheAwareGlobalScheduler,
    )

    simulator = build_simulator(argv)
    simulator.run()
    if isinstance(simulator._scheduler, RLCacheAwareGlobalScheduler):
        simulator._scheduler.save_final_checkpoint()
    return simulator
