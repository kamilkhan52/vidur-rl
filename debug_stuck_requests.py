#!/usr/bin/env python3
"""
Debug script to understand why requests are stuck in waiting queues.
"""

import sys
sys.path.insert(0, '/home/kamil/vidur-rl-working')

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.logger import init_logger

# Initialize logger
init_logger("vidur")
logger = init_logger(__name__)

def main():
    # Create config for Scale 1.5
    config = SimulationConfig.create_from_args([
        '--replica_config_device', 'a100',
        '--replica_config_model_name', 'meta-llama/Llama-2-7b-hf',
        '--replica_config_tensor_parallel_size', '1',
        '--cluster_config_num_replicas', '16',
        '--replica_scheduler_config_type', 'vllm_v1',
        '--global_scheduler_config_type', 'lop',
        '--trace_request_generator_config_time_scale_factor', '1.5',
        '--trace_request_generator_config_trace_file', '/home/kamil/vidur-rl-working/data/processed_traces/mooncake_synthetic_trace.csv',
        '--request_generator_config_type', 'trace',
        '--cache_config_enable_prefix_caching',
        '--no-metrics_config_enable_chrome_trace',
        '--no-metrics_config_write_json_trace',
        '--metrics_config_output_dir', '/tmp/vidur_debug_output',
    ])
    
    # Create simulator and patch to track waiting queue state
    simulator = Simulator(config)
    
    # Patch the scheduler to log waiting queue state
    original_schedule = simulator._scheduler.schedule
    original_add_request = None
    
    def patched_schedule():
        result = original_schedule()
        if simulator._time > 1537.0:  # Near end of trace
            for rid, rs in simulator._scheduler._replica_schedulers.items():
                waiting = len(rs._waiting_queue)
                running = len(rs._running)
                if waiting > 0:
                    next_request = rs._waiting_queue.peek() if hasattr(rs._waiting_queue, 'peek') else None
                    if next_request:
                        logger.info(
                            f"Time {simulator._time}: Replica {rid} has {waiting} waiting, {running} running. "
                            f"Next request: id={next_request.id}, prefill_tokens={next_request.num_prefill_tokens}"
                        )
        return result
    
    simulator._scheduler.schedule = patched_schedule
    
    logger.info("Starting debug simulation...")
    simulator.run()
    logger.info(f"Simulation ended at time {simulator._time}")

if __name__ == '__main__':
    main()
