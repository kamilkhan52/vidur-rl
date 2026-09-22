#!/usr/bin/env python3
"""
Evaluate schedulers on mooncake workload with different time_scale_factors.

Tests how schedulers perform under different load conditions by varying
the request arrival rate via time_scale_factor.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def run_simulation(
    scheduler: str,
    time_scale_factor: float,
    output_dir: Path,
    num_requests: int = 2000,
    enable_cache: bool = True,
) -> Optional[Dict]:
    """Run a single simulation."""
    
    # Scheduler config mapping
    scheduler_configs = {
        'round_robin': 'round_robin',
        'lor': 'lor',
        'lop_uncached': 'lop_uncached',
        'tolerant_sticky': 'tolerant_sticky_lop_uncached',
    }
    
    scheduler_config = scheduler_configs[scheduler]
    
    # Build command
    # Note: max_tokens=122752 is the A100 KV cache capacity for Llama-2-7b
    # (7672 blocks * 16 tokens/block). Requests exceeding this cannot be scheduled.
    cmd = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", "4",
        "--replica_scheduler_config_type", "vllm_v1",
        "--global_scheduler_config_type", scheduler_config,
        "--request_generator_config_type", "trace",
        "--trace_request_generator_config_trace_file", "data/processed_traces/mooncake_conversation_trace.csv",
        "--trace_request_generator_config_time_scale_factor", str(time_scale_factor),
        "--trace_request_generator_config_num_requests", str(num_requests),
        "--trace_request_generator_config_max_tokens", "122752",
        "--metrics_config_output_dir", str(output_dir),
        # Quick mode - disable expensive metrics
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_enable_chrome_trace",
    ]
    
    if enable_cache:
        cmd.append("--cache_config_enable_prefix_caching")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {scheduler} @ scale={time_scale_factor}")
    print(f"  QPS: {3.4 / time_scale_factor:.3f} (original: 3.4)")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        
        # Save logs
        (output_dir / "stdout.log").write_text(result.stdout)
        (output_dir / "stderr.log").write_text(result.stderr)
        
        if result.returncode != 0:
            print(f"  ❌ FAILED with exit code {result.returncode}")
            print(f"  stderr: {result.stderr[-500:]}")
            return None
        
        # Find output directory
        sim_dirs = sorted([d for d in output_dir.glob("*") if d.is_dir()])
        if not sim_dirs:
            print(f"  ❌ No output directory found")
            return None

        latest_dir = sim_dirs[-1]

        # Extract simulation end time from stdout.log (logs go to stdout, not a separate file)
        stdout_log = output_dir / "stdout.log"
        if not stdout_log.exists():
            print(f"  ❌ No stdout.log found")
            return None

        # Parse simulation end time
        end_time = None
        with open(stdout_log) as f:
            for line in f:
                if "Simulation ended at:" in line:
                    end_time = float(line.split("at:")[1].strip().rstrip('s'))
                    break

        if end_time is None:
            print(f"  ❌ Could not find simulation end time in stdout.log")
            return None

        # Read actual metrics from request_metrics.csv
        metrics_csv = latest_dir / "request_metrics.csv"
        if not metrics_csv.exists():
            print(f"  ❌ No request_metrics.csv found")
            return None

        req_df = pd.read_csv(metrics_csv)

        metrics = {
            'scheduler': scheduler,
            'time_scale_factor': time_scale_factor,
            'qps': 3.4 / time_scale_factor,
            'num_requests': len(req_df),
            'simulation_end_time': end_time,
            'mean_e2e_time': req_df['request_e2e_time'].mean(),
            'median_e2e_time': req_df['request_e2e_time'].median(),
            'p95_e2e_time': req_df['request_e2e_time'].quantile(0.95),
            'p99_e2e_time': req_df['request_e2e_time'].quantile(0.99),
            'mean_scheduling_delay': req_df['request_scheduling_delay'].mean(),
            'mean_execution_time': req_df['request_execution_time'].mean(),
        }

        # Add cache hit rate if available
        if 'request_num_prefill_tokens_cached' in req_df.columns:
            cached = req_df['request_num_prefill_tokens_cached'].sum()
            total = req_df['request_num_prefill_tokens'].sum()
            metrics['cache_hit_rate'] = cached / total if total > 0 else 0

        print(f"  ✅ SUCCESS - Mean E2E: {metrics['mean_e2e_time']:.2f}s, Median: {metrics['median_e2e_time']:.2f}s, P95: {metrics['p95_e2e_time']:.2f}s")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  TIMEOUT after 2 hours")
        return None
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate mooncake workload with different time scales")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/eval_mooncake_scaling',
        help='Output directory for results'
    )
    parser.add_argument(
        '--schedulers',
        nargs='+',
        default=['round_robin', 'lor', 'lop_uncached'],
        help='Schedulers to evaluate'
    )
    parser.add_argument(
        '--scale_factors',
        nargs='+',
        type=float,
        default=[16.0, 32.0, 64.0],
        help='Time scale factors to test'
    )
    parser.add_argument(
        '--num_requests',
        type=int,
        default=2000,
        help='Number of requests to simulate'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable prefix caching'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_file = output_dir / "evaluation.log"
    
    print("=" * 80)
    print("Mooncake Workload Scaling Evaluation")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print(f"Schedulers: {', '.join(args.schedulers)}")
    print(f"Scale factors: {args.scale_factors}")
    print(f"Requests: {args.num_requests}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Run all combinations
    total = len(args.schedulers) * len(args.scale_factors)
    current = 0
    
    for scale_factor in args.scale_factors:
        for scheduler in args.schedulers:
            current += 1
            print(f"\n[{current}/{total}] {scheduler} @ scale={scale_factor}")
            print("-" * 80)
            
            run_dir = output_dir / f"{scheduler}_scale{scale_factor}"
            
            result = run_simulation(
                scheduler=scheduler,
                time_scale_factor=scale_factor,
                output_dir=run_dir,
                num_requests=args.num_requests,
                enable_cache=not args.no_cache,
            )
            
            if result:
                all_results.append(result)
    
    # Save results
    if all_results:
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary table
        df = pd.DataFrame(all_results)

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        # E2E Latency metrics
        print("\nMean E2E Latency (seconds):")
        pivot = df.pivot(index='scheduler', columns='time_scale_factor', values='mean_e2e_time')
        print(pivot.round(2).to_string())

        print("\n\nMedian E2E Latency (seconds):")
        pivot_median = df.pivot(index='scheduler', columns='time_scale_factor', values='median_e2e_time')
        print(pivot_median.round(2).to_string())

        print("\n\nP95 E2E Latency (seconds):")
        pivot_p95 = df.pivot(index='scheduler', columns='time_scale_factor', values='p95_e2e_time')
        print(pivot_p95.round(2).to_string())

        print("\n\nP99 E2E Latency (seconds):")
        pivot_p99 = df.pivot(index='scheduler', columns='time_scale_factor', values='p99_e2e_time')
        print(pivot_p99.round(2).to_string())

        print("\n\nMean Scheduling Delay (seconds):")
        pivot_sched = df.pivot(index='scheduler', columns='time_scale_factor', values='mean_scheduling_delay')
        print(pivot_sched.round(2).to_string())

        if 'cache_hit_rate' in df.columns:
            print("\n\nCache Hit Rate (%):")
            pivot_cache = df.pivot(index='scheduler', columns='time_scale_factor', values='cache_hit_rate')
            print((pivot_cache * 100).round(1).to_string())

        print("\n\nQuery Rate (QPS):")
        for scale in args.scale_factors:
            print(f"  scale={scale}: {3.4/scale:.3f} QPS")
        
        # Save CSV
        csv_file = output_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n📊 Results saved to: {csv_file}")
    else:
        print("\n❌ No successful runs!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
