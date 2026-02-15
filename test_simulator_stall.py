#!/usr/bin/env python3
"""
Test script to reproduce the simulator stall issue at low scales.
This should complete all requests, not leave any pending in the scheduler.
"""
import sys
import time
from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig
from vidur.config.config import SimulationConfig
from vidur.simulator import Simulator

def main():
    # Create a minimal config for LOP baseline at Scale 1.5
    config_str = """
cluster_config:
  replica_config:
    model_name: meta-llama/Llama-2-7b-hf
    num_pipeline_stages: 1
  num_replicas: 16
  request_queue_config:
    queue_size: -1
  replica_scheduler_config:
    name: sarathi
    batch_size_cap: 256
    max_batch_size: 256
  global_scheduler_config:
    name: lop_batch_end
    schedule_batch_end_only: true

request_generator_config:
  type: synthetic
  length: -1
  interval_generator_config:
    type: static
    qps: 1.5
  num_requests: 50
  seed: 42
  trace_file: data/processed_traces/mooncake_synthetic_trace.csv

duration: 3600
metrics_config:
  enable_chrome_trace: false
  write_json_trace: false
  
skip_cpu_overhead_modeling: true
"""
    
    # Write config to temp file
    import yaml
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_str)
        config_path = f.name
    
    try:
        # Load config and run simulator
        print(f"Loading config from {config_path}")
        config = SimulationConfig.create_from_file(config_path)
        
        print(f"Starting simulation with {config.request_generator_config.num_requests} requests at QPS {config.request_generator_config.interval_generator_config.qps}")
        print(f"Expected duration: ~{config.request_generator_config.num_requests / config.request_generator_config.interval_generator_config.qps:.1f}s")
        
        start_time = time.time()
        simulator = Simulator(config)
        simulator.run()
        elapsed = time.time() - start_time
        
        print(f"\n=== Simulation Complete ===")
        print(f"Wall-clock time: {elapsed:.2f}s")
        print(f"Simulated time: {simulator._time:.2f}s")
        print(f"Scheduler empty: {simulator._scheduler.is_empty()}")
        print(f"Time limit reached: {simulator._time_limit_reached}")
        
        if not simulator._scheduler.is_empty():
            print("\n⚠️  ERROR: Scheduler still has pending requests!")
            return 1
        else:
            print("\n✅ SUCCESS: All requests completed")
            return 0
            
    finally:
        os.unlink(config_path)

if __name__ == "__main__":
    sys.exit(main())
