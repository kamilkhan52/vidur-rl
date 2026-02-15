#!/usr/bin/env python3
"""
Test script to verify the simulator no longer stalls at low scales.

This creates a minimal config with low QPS and runs a short simulation
to ensure all requests complete successfully.
"""

import sys
import os

# Add vidur to path
sys.path.insert(0, '/home/kamil/vidur-rl-working')

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds

# Import the config builder
from scripts.vidur_config_alpine import build_simulation_config

def test_low_scale_simulation():
    """Test that simulation completes at low scale (Scale 1.5)"""
    
    print("=" * 80)
    print("Testing Simulator at Scale 1.5 (Low Request Rate)")
    print("=" * 80)
    
    # Build config for LOP scheduler at Scale 1.5
    # Using parameters similar to baseline sweep
    config = build_simulation_config(
        scheduler_name="lop_batch_end",
        model_name="meta-llama/Llama-2-7b-hf",
        num_replicas=16,
        trace_type="synthetic_trace",
        qps=1.5,  # Low scale
        tensor_parallel_size=1,
        batch_size_cap=256,
        max_batch_size=256,
        block_size=16,
        num_blocks=1638,
        enable_prefix_caching=False,
        output_dir="/tmp/test_simulator",
        duration=3600,
        num_runs=1  # Just one iteration for testing
    )
    
    # Use a subset of requests for faster testing
    config.request_generator_config.num_requests = 100
    
    print(f"\nConfiguration:")
    print(f"  Scheduler: {config.cluster_config.global_scheduler_config.name}")
    print(f"  Model: {config.cluster_config.replica_config.model_name}")
    print(f"  Replicas: {config.cluster_config.num_replicas}")
    print(f"  QPS: {config.request_generator_config.interval_generator_config.qps}")
    print(f"  Requests: {config.request_generator_config.num_requests}")
    print(f"  Trace: {config.request_generator_config.trace_file}")
    
    expected_duration = config.request_generator_config.num_requests / config.request_generator_config.interval_generator_config.qps
    print(f"  Expected arrival span: ~{expected_duration:.1f}s")
    
    # Run simulation
    print(f"\nStarting simulation...")
    set_seeds(config.seed)
    
    simulator = Simulator(config)
    
    try:
        simulator.run()
        
        print(f"\n{'=' * 80}")
        print("SIMULATION RESULTS")
        print(f"{'=' * 80}")
        print(f"  Simulated time: {simulator._time:.2f}s")
        print(f"  Scheduler empty: {simulator._scheduler.is_empty()}")
        print(f"  Time limit reached: {simulator._time_limit_reached}")
        
        if not simulator._scheduler.is_empty():
            print(f"\n  ❌ FAILURE: Scheduler still has pending requests!")
            print(f"  This indicates the simulator stalled before completing all work.")
            return False
        else:
            print(f"\n  ✅ SUCCESS: All requests completed successfully!")
            print(f"  The simulator fix is working correctly.")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: Simulation failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_low_scale_simulation()
    sys.exit(0 if success else 1)
