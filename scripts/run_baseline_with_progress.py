#!/usr/bin/env python3
"""
Run baseline scheduler on full trace with progress output.

Usage:
    python scripts/run_baseline_with_progress.py [--scheduler round_robin] [--scale 8.0]
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Run baseline scheduler with progress output')
    parser.add_argument('--scheduler', type=str, default='round_robin',
                        choices=['round_robin', 'lor', 'lop_uncached', 'random',
                                 'sticky_round_robin', 'sticky_lor',
                                 'ranked_sticky_lop_uncached', 'tolerant_sticky_lop_uncached'],
                        help='Scheduler to use')
    parser.add_argument('--scale', type=float, default=8.0,
                        help='Time scale factor')
    parser.add_argument('--num_requests', type=int, default=None,
                        help='Number of requests (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/baseline_run',
                        help='Output directory')
    parser.add_argument('--trace', type=str, default='conversation',
                        choices=['conversation', 'synthetic', 'toolagent'],
                        help='Which trace to use')
    args = parser.parse_args()

    # Configuration
    trace_files = {
        'conversation': 'data/processed_traces/mooncake_conversation_trace.csv',
        'synthetic': 'data/processed_traces/mooncake_synthetic_trace.csv',
        'toolagent': 'data/processed_traces/mooncake_toolagent_trace.csv',
    }
    trace_file = trace_files[args.trace]
    output_dir = Path(args.output_dir) / f"{args.trace}_{args.scheduler}_scale{args.scale}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total requests (after filtering)
    print("Loading trace file...")
    df = pd.read_csv(trace_file)
    total_tokens = df['num_prefill_tokens'] + df['num_decode_tokens']
    df_filtered = df[total_tokens <= 122752]
    total_requests = args.num_requests if args.num_requests else len(df_filtered)
    total_requests = min(total_requests, len(df_filtered))

    print("=" * 80)
    print(f"  Baseline Run: {args.scheduler.upper()}")
    print("=" * 80)
    print(f"  Scheduler:    {args.scheduler}")
    print(f"  Scale factor: {args.scale}x (effective QPS: {3.4/args.scale:.2f})")
    print(f"  Requests:     {total_requests:,}")
    print(f"  Output:       {output_dir}")
    print("=" * 80)
    print()

    # Build command
    cmd = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", "4",
        "--replica_scheduler_config_type", "vllm_v1",
        "--global_scheduler_config_type", args.scheduler,
        "--request_generator_config_type", "trace",
        "--trace_request_generator_config_trace_file", trace_file,
        "--trace_request_generator_config_time_scale_factor", str(args.scale),
        "--trace_request_generator_config_max_tokens", "122752",
        "--cache_config_enable_prefix_caching",
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_enable_chrome_trace",
        "--metrics_config_output_dir", str(output_dir),
    ]

    if args.num_requests:
        cmd.extend(["--trace_request_generator_config_num_requests", str(args.num_requests)])

    print("Starting simulation (this may take a moment to initialize)...")
    print()
    sys.stdout.flush()

    start_time = time.time()
    last_progress_time = start_time
    last_count = 0
    metrics_file = None

    # Run simulation with real-time output
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        import select

        # Stream output and monitor progress
        while True:
            # Check if process has output (non-blocking)
            if process.stdout in select.select([process.stdout], [], [], 1.0)[0]:
                line = process.stdout.readline()
                if not line:
                    break  # EOF
                line = line.rstrip()
                if line:
                    # Show important lines, skip verbose debug
                    if any(x in line for x in ['INFO', 'WARNING', 'ERROR', 'Starting simulation', 'Simulation ended', 'Loaded trace']):
                        print(line)
                        sys.stdout.flush()

            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                for line in process.stdout:
                    line = line.rstrip()
                    if line and any(x in line for x in ['INFO', 'WARNING', 'ERROR', 'Starting simulation', 'Simulation ended', 'Loaded trace']):
                        print(line)
                        sys.stdout.flush()
                break

            # Check for metrics file periodically (every iteration = ~1 second)
            current_time = time.time()
            if current_time - last_progress_time > 5.0:  # Every 5 seconds
                last_progress_time = current_time

                # Find metrics file
                if metrics_file is None:
                    sim_dirs = sorted([d for d in output_dir.glob("*") if d.is_dir()])
                    if sim_dirs:
                        candidate = sim_dirs[-1] / "request_metrics.csv"
                        if candidate.exists():
                            metrics_file = candidate
                            print(f"[Progress] Found metrics file, tracking progress...")
                            sys.stdout.flush()

                # Read progress
                if metrics_file and metrics_file.exists():
                    try:
                        df_progress = pd.read_csv(metrics_file)
                        current_count = len(df_progress)

                        if current_count > last_count:
                            elapsed = current_time - start_time
                            rate = current_count / elapsed if elapsed > 0 else 0
                            remaining = (total_requests - current_count) / rate if rate > 0 else 0
                            progress = 100.0 * current_count / total_requests

                            # Recent stats
                            recent = df_progress.tail(100)
                            recent_e2e = recent['request_e2e_time'].mean()
                            overall_e2e = df_progress['request_e2e_time'].mean()

                            print(f"[Progress] {progress:5.1f}% ({current_count:>5}/{total_requests}) | "
                                  f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f}m | "
                                  f"E2E: {recent_e2e:.1f}s (recent) / {overall_e2e:.1f}s (avg)")
                            sys.stdout.flush()

                            last_count = current_count
                    except Exception:
                        pass  # File might be locked

        process.wait()
        elapsed = time.time() - start_time

        print()

        if process.returncode != 0:
            print(f"ERROR: Simulation failed with exit code {process.returncode}")
            return 1

        # Final results
        print("=" * 80)
        print("  RESULTS")
        print("=" * 80)

        # Find metrics file
        sim_dirs = sorted([d for d in output_dir.glob("*") if d.is_dir()])
        if sim_dirs:
            metrics_file = sim_dirs[-1] / "request_metrics.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)

                print(f"\n  Completed:     {len(df):,} requests in {elapsed:.1f}s")
                print(f"  Throughput:    {len(df)/elapsed:.2f} requests/s")
                print()
                print("  E2E Latency:")
                print(f"    Mean:        {df['request_e2e_time'].mean():.2f}s")
                print(f"    Median:      {df['request_e2e_time'].median():.2f}s")
                print(f"    P95:         {df['request_e2e_time'].quantile(0.95):.2f}s")
                print(f"    P99:         {df['request_e2e_time'].quantile(0.99):.2f}s")
                print(f"    Max:         {df['request_e2e_time'].max():.2f}s")
                print()
                print("  Scheduling Delay:")
                print(f"    Mean:        {df['request_scheduling_delay'].mean():.2f}s")
                print(f"    Max:         {df['request_scheduling_delay'].max():.2f}s")
                print()

                # Latency by time segment
                print("  Latency by Trace Segment:")
                segment_size = len(df) // 6
                for i in range(6):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < 5 else len(df)
                    segment = df.iloc[start_idx:end_idx]
                    print(f"    Req {start_idx:>5}-{end_idx:>5}: Mean E2E = {segment['request_e2e_time'].mean():.2f}s")

                # Cache hit rate
                if 'request_num_prefill_tokens_cached' in df.columns:
                    cached = df['request_num_prefill_tokens_cached'].sum()
                    total = df['request_num_prefill_tokens'].sum()
                    cache_rate = 100 * cached / total if total > 0 else 0
                    print(f"\n  Cache Hit Rate: {cache_rate:.1f}%")

                # Load distribution
                print("\n  Load Distribution:")
                for replica in sorted(df['replica'].unique()):
                    count = (df['replica'] == replica).sum()
                    pct = 100 * count / len(df)
                    print(f"    Replica {replica}: {count:>5} ({pct:5.1f}%)")

        print()
        print(f"  Output saved to: {output_dir}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        process.kill()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
