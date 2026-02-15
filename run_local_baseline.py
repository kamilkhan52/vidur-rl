#!/usr/bin/env python3
"""
Simple baseline runner for local testing.
Run with: python3 run_local_baseline.py
"""

import os
import sys

# Add vidur to path
sys.path.insert(0, '/home/kamil/vidur-rl-working')

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
import logging

logger = logging.getLogger(__name__)

def main():
    # Create a minimal simulation config programmatically
    # We'll use the default values and override key params
    
    # Check if trace file exists
    trace_file = '/home/kamil/vidur-rl-working/data/processed_traces/mooncake_synthetic_trace.csv'
    if not os.path.exists(trace_file):
        print(f"ERROR: Trace file not found: {trace_file}")
        return 1
    
    # Get first few requests to limit the run
    import csv
    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        request_count = sum(1 for _ in reader)
    print(f"Trace has {request_count} requests")
    
    # Use CLI args with reduced requests via environment or config
    # For testing, let's just use the config file approach or CLI
    cmd = [
        'python3', '-m', 'vidur.main',
        '--replica_config_device', 'a100',
        '--replica_config_model_name', 'meta-llama/Llama-2-7b-hf',
        '--replica_config_tensor_parallel_size', '1',
        '--cluster_config_num_replicas', '16',
        '--replica_scheduler_config_type', 'vllm_v1',
        '--global_scheduler_config_type', 'lop',
        '--trace_request_generator_config_time_scale_factor', '1.5',
        '--trace_request_generator_config_trace_file', trace_file,
        '--request_generator_config_type', 'trace',
        '--cache_config_enable_prefix_caching',
        '--no-metrics_config_enable_chrome_trace',
        '--no-metrics_config_write_json_trace',
        '--metrics_config_output_dir', '/tmp/vidur_test_output',
    ]
    
    print(f"Running: {' '.join(cmd)}")
    os.makedirs('/tmp/vidur_test_output', exist_ok=True)
    
    os.chdir('/home/kamil/vidur-rl-working')
    os.execvp('python3', cmd)

if __name__ == '__main__':
    sys.exit(main())
