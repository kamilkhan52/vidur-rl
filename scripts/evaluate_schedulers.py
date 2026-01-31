#!/usr/bin/env python3
"""
Comprehensive scheduler evaluation framework.

Runs multiple scheduling policies across different workloads and generates
comparison plots, CSV summaries, and analysis reports.

Supports:
  - 6 schedulers: random, round_robin, lor, lop_uncached, tolerant_sticky, rl_cache_aware
  - 4 workloads: splitwise_conv, arxiv, splitwise_code, mooncake
  - Multiple runs for statistical significance
  - Time scale factor adjustment for trace replay
  - Comprehensive visualization (6 plot types + summary table)

Usage:
    # Evaluate all schedulers on all workloads
    python scripts/evaluate_schedulers.py --output_dir results/eval_full

    # Quick test with specific schedulers
    python scripts/evaluate_schedulers.py --schedulers rl_cache_aware lop_uncached --quick

    # Mooncake evaluation with scaled arrival rate
    python scripts/evaluate_schedulers.py --workloads mooncake --time_scale_factor 16.0

    # Multiple runs with RL model
    python scripts/evaluate_schedulers.py --num_runs 3 --rl_model_path rl_models/model.pt
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class SchedulerEvaluator:
    """Evaluates multiple scheduling policies across workloads."""

    # Available schedulers and their config names
    SCHEDULERS = {
        'random': 'random',
        'round_robin': 'round_robin',
        'lor': 'lor',
        'lop_uncached': 'lop_uncached',
        'tolerant_sticky': 'tolerant_sticky_lop_uncached',
        'rl_cache_aware': 'rl_cache_aware',
    }

    # Workload configurations
    WORKLOADS = {
        'splitwise_conv': {
            'request_generator_config_type': 'trace',
            'trace_request_generator_config_trace_file': 'data/processed_traces/splitwise_conv.csv',
            'description': 'Splitwise Conversational (19K requests)',
        },
        'arxiv': {
            'request_generator_config_type': 'trace',
            'trace_request_generator_config_trace_file': 'data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv',
            'description': 'ArXiv Summarization (28K requests)',
        },
        'splitwise_code': {
            'request_generator_config_type': 'trace',
            'trace_request_generator_config_trace_file': 'data/processed_traces/splitwise_code.csv',
            'description': 'Splitwise Code Generation (8.8K requests)',
        },
        'mooncake': {
            'request_generator_config_type': 'trace',
            'trace_request_generator_config_trace_file': 'data/processed_traces/mooncake_conversation_trace.csv',
            'trace_request_generator_config_max_tokens': 122752,  # A100 KV cache capacity
            'description': 'Mooncake Conversations (12K requests, multi-turn, session-aware)',
        },
    }

    def __init__(self, output_dir: str, schedulers: Optional[List[str]] = None,
                 workloads: Optional[List[str]] = None, quick: bool = False,
                 rl_model_path: Optional[str] = None,
                 time_scale_factor: Optional[float] = None):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to store results
            schedulers: List of scheduler names to evaluate (None = all)
            workloads: List of workload names to evaluate (None = all)
            quick: If True, use faster settings (fewer requests, less detailed metrics)
            rl_model_path: Path to RL model checkpoint (for rl_cache_aware scheduler)
            time_scale_factor: Scale factor for trace arrival times (higher = slower QPS)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.schedulers = schedulers or list(self.SCHEDULERS.keys())
        self.workloads = workloads or list(self.WORKLOADS.keys())
        self.quick = quick
        self.rl_model_path = rl_model_path
        self.time_scale_factor = time_scale_factor

        # Results storage
        self.results = []

        # Logging
        self.log_file = self.output_dir / "evaluation.log"
        self.log_file.write_text(f"Evaluation started at {datetime.now()}\n")

    def log(self, message: str):
        """Log message to file and stdout."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def run_simulation(self, scheduler: str, workload: str, run_id: int = 0) -> Optional[Dict]:
        """
        Run a single simulation.

        Args:
            scheduler: Scheduler name
            workload: Workload name
            run_id: Run identifier for multiple runs

        Returns:
            Dictionary with metrics, or None if failed
        """
        workload_config = self.WORKLOADS[workload]
        scheduler_config_name = self.SCHEDULERS[scheduler]

        # Build output directory for this run
        run_dir = self.output_dir / "simulations" / f"{scheduler}_{workload}_run{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable, "-m", "vidur.main",
            "--replica_config_device", "a100",
            "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
            "--cluster_config_num_replicas", "4",
            "--replica_scheduler_config_type", "vllm_v1",
            "--global_scheduler_config_type", scheduler_config_name,
            "--metrics_config_output_dir", str(run_dir),
        ]

        # Add workload-specific config
        for key, value in workload_config.items():
            if key == 'description':
                continue
            cmd.extend([f"--{key}", str(value)])

        # Enable prefix caching for Mooncake (has block_hash_ids)
        if workload == 'mooncake':
            cmd.append("--cache_config_enable_prefix_caching")

        # Add time scale factor if specified
        if self.time_scale_factor is not None:
            cmd.extend([
                "--trace_request_generator_config_time_scale_factor",
                str(self.time_scale_factor),
            ])

        # Add scheduler-specific config
        if scheduler == 'rl_cache_aware':
            cmd.extend([
                "--r_l_cache_aware_global_scheduler_config_mode", "inference",
            ])
            if self.rl_model_path:
                cmd.extend([
                    "--r_l_cache_aware_global_scheduler_config_model_checkpoint_path",
                    self.rl_model_path,
                ])

        # Always disable chrome trace to save disk space
        cmd.append("--no-metrics_config_enable_chrome_trace")

        # Quick mode: disable some expensive metrics
        if self.quick:
            cmd.extend([
                "--no-metrics_config_write_json_trace",
                "--no-metrics_config_store_operation_metrics",
                "--no-metrics_config_store_batch_metrics",
            ])

        self.log(f"Running: {scheduler} on {workload} (run {run_id})")
        self.log(f"  Command: {' '.join(cmd)}")

        try:
            # Run simulation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            # Save stdout/stderr for debugging
            stdout_file = run_dir / "stdout.log"
            stderr_file = run_dir / "stderr.log"
            stdout_file.write_text(result.stdout)
            stderr_file.write_text(result.stderr)

            if result.returncode != 0:
                self.log(f"  ❌ FAILED with exit code {result.returncode}")
                self.log(f"  stderr: {result.stderr[-500:]}")  # Last 500 chars
                return None

            # Find the actual output directory (has timestamp)
            sim_dirs = sorted([d for d in run_dir.glob("*") if d.is_dir()])
            if not sim_dirs:
                self.log(f"  ❌ No output directory found")
                return None

            latest_dir = sim_dirs[-1]

            # Load metrics
            metrics = self.extract_metrics(latest_dir, scheduler, workload, run_id)

            if metrics:
                self.log(f"  ✅ SUCCESS - Mean E2E: {metrics['mean_e2e_time']:.3f}s")
            else:
                self.log(f"  ⚠️  Completed but failed to extract metrics")

            return metrics

        except subprocess.TimeoutExpired:
            self.log(f"  ❌ TIMEOUT after 1 hour")
            return None
        except Exception as e:
            self.log(f"  ❌ ERROR: {e}")
            return None

    def extract_metrics(self, result_dir: Path, scheduler: str, workload: str, run_id: int) -> Optional[Dict]:
        """
        Extract metrics from simulation output.

        Args:
            result_dir: Directory with simulation results
            scheduler: Scheduler name
            workload: Workload name
            run_id: Run identifier

        Returns:
            Dictionary with extracted metrics
        """
        metrics_file = result_dir / "request_metrics.csv"

        if not metrics_file.exists():
            return None

        try:
            df = pd.read_csv(metrics_file)

            # Calculate metrics
            metrics = {
                'scheduler': scheduler,
                'workload': workload,
                'run_id': run_id,
                'num_requests': len(df),

                # E2E latency
                'mean_e2e_time': float(df['request_e2e_time'].mean()),
                'median_e2e_time': float(df['request_e2e_time'].quantile(0.5)),
                'p90_e2e_time': float(df['request_e2e_time'].quantile(0.9)),
                'p95_e2e_time': float(df['request_e2e_time'].quantile(0.95)),
                'p99_e2e_time': float(df['request_e2e_time'].quantile(0.99)),
                'std_e2e_time': float(df['request_e2e_time'].std()),

                # TTFT (prefill latency)
                'mean_ttft': float(df['prefill_e2e_time'].mean()),
                'median_ttft': float(df['prefill_e2e_time'].quantile(0.5)),
                'p90_ttft': float(df['prefill_e2e_time'].quantile(0.9)),
                'p99_ttft': float(df['prefill_e2e_time'].quantile(0.99)),

                # Throughput
                'total_tokens': int(df['request_num_tokens'].sum()),
                'simulation_time': float(df['request_arrived_at'].max()),
                'throughput': float(df['request_num_tokens'].sum() / df['request_arrived_at'].max()),

                # Load balance
                'replica_counts': df['replica'].value_counts().to_dict(),
                'load_imbalance': self._calculate_load_imbalance(df),

                # Scheduling
                'mean_scheduling_delay': float(df['request_scheduling_delay'].mean()),
                'mean_execution_time': float(df['request_execution_time'].mean()),
            }

            # Cache hit rate metrics (if available)
            if 'request_num_prefill_tokens_cached' in df.columns:
                cached = df['request_num_prefill_tokens_cached']
                total = df['request_num_prefill_tokens']
                # Token-weighted: total cached tokens / total prefill tokens
                metrics['cache_hit_rate_token'] = float(100 * cached.sum() / total.sum()) if total.sum() > 0 else 0.0
                # Request-weighted: mean of per-request hit ratios
                hit_ratios = cached / total
                metrics['cache_hit_rate_request'] = float(100 * hit_ratios.mean())
                # Percentage of requests with any cache hit
                metrics['pct_requests_with_cache_hit'] = float(100 * (cached > 0).sum() / len(df))

            return metrics

        except Exception as e:
            self.log(f"  Error extracting metrics: {e}")
            return None

    def _calculate_load_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate load imbalance metric (0 = perfect balance, 1 = maximum imbalance)."""
        counts = df['replica'].value_counts()
        if len(counts) == 0:
            return 0.0
        max_load = counts.max()
        min_load = counts.min()
        return float((max_load - min_load) / len(df))

    def run_all(self, num_runs: int = 1):
        """
        Run all combinations of schedulers and workloads.

        Args:
            num_runs: Number of runs per combination (for statistical significance)
        """
        total = len(self.schedulers) * len(self.workloads) * num_runs
        current = 0

        self.log(f"\n{'='*80}")
        self.log(f"Starting evaluation: {len(self.schedulers)} schedulers × {len(self.workloads)} workloads × {num_runs} runs")
        self.log(f"Total simulations: {total}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"{'='*80}\n")

        for scheduler in self.schedulers:
            for workload in self.workloads:
                for run in range(num_runs):
                    current += 1
                    self.log(f"\n[{current}/{total}] {scheduler} × {workload} (run {run})")

                    metrics = self.run_simulation(scheduler, workload, run)
                    if metrics:
                        self.results.append(metrics)

        self.log(f"\n{'='*80}")
        self.log(f"Evaluation complete: {len(self.results)}/{total} successful")
        self.log(f"{'='*80}\n")

    def save_results(self):
        """Save results to CSV and JSON."""
        if not self.results:
            self.log("No results to save")
            return

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        self.log(f"Saved results to {csv_path}")

        # Save as JSON (includes nested replica_counts)
        json_path = self.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"Saved results to {json_path}")

    def generate_plots(self):
        """Generate comparison plots."""
        if not self.results:
            self.log("No results to plot")
            return

        df = pd.DataFrame(self.results)
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        self.log("\nGenerating plots...")

        # 1. Mean E2E Latency Comparison
        self._plot_metric_comparison(
            df, 'mean_e2e_time',
            "Mean End-to-End Latency by Scheduler",
            "Mean E2E Latency (s)",
            plots_dir / "mean_e2e_latency.png"
        )

        # 2. P99 E2E Latency Comparison
        self._plot_metric_comparison(
            df, 'p99_e2e_time',
            "P99 End-to-End Latency by Scheduler",
            "P99 E2E Latency (s)",
            plots_dir / "p99_e2e_latency.png"
        )

        # 3. TTFT Comparison
        self._plot_metric_comparison(
            df, 'mean_ttft',
            "Mean Time-to-First-Token (TTFT) by Scheduler",
            "Mean TTFT (s)",
            plots_dir / "mean_ttft.png"
        )

        # 4. Throughput Comparison
        self._plot_metric_comparison(
            df, 'throughput',
            "Throughput by Scheduler",
            "Throughput (tokens/s)",
            plots_dir / "throughput.png",
            higher_is_better=True
        )

        # 5. Load Imbalance Comparison
        self._plot_metric_comparison(
            df, 'load_imbalance',
            "Load Imbalance by Scheduler",
            "Load Imbalance (lower is better)",
            plots_dir / "load_imbalance.png"
        )

        # 6. Latency Distribution (box plot)
        self._plot_latency_distributions(df, plots_dir / "latency_distributions.png")

        # 7. Summary table
        self._generate_summary_table(df, plots_dir / "summary_table.png")

        self.log(f"Plots saved to {plots_dir}")

    def _plot_metric_comparison(self, df: pd.DataFrame, metric: str, title: str,
                                 ylabel: str, output_path: Path, higher_is_better: bool = False):
        """Plot bar chart comparing a metric across schedulers."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by scheduler and workload
        pivot = df.pivot_table(values=metric, index='scheduler', columns='workload', aggfunc='mean')

        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Scheduler", fontsize=12)
        ax.legend(title="Workload", fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_latency_distributions(self, df: pd.DataFrame, output_path: Path):
        """Plot latency percentiles as grouped bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, workload in enumerate(df['workload'].unique()):
            ax = axes[idx] if len(df['workload'].unique()) > 1 else axes

            workload_df = df[df['workload'] == workload]

            # Prepare data for grouped bar chart
            metrics = ['median_e2e_time', 'p90_e2e_time', 'p95_e2e_time', 'p99_e2e_time']
            labels = ['P50', 'P90', 'P95', 'P99']

            x = np.arange(len(workload_df['scheduler'].unique()))
            width = 0.2

            for i, (metric, label) in enumerate(zip(metrics, labels)):
                values = workload_df.groupby('scheduler')[metric].mean()
                offset = width * (i - len(metrics)/2 + 0.5)
                ax.bar(x + offset, values, width, label=label)

            ax.set_title(f"Latency Percentiles - {workload}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Latency (s)", fontsize=11)
            ax.set_xlabel("Scheduler", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(workload_df['scheduler'].unique(), rotation=45, ha='right')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_table(self, df: pd.DataFrame, output_path: Path):
        """Generate summary table as image."""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')

        # Aggregate metrics by scheduler
        agg_dict = {
            'mean_e2e_time': 'mean',
            'p99_e2e_time': 'mean',
            'mean_ttft': 'mean',
            'throughput': 'mean',
            'load_imbalance': 'mean',
        }
        col_names = ['Mean E2E (s)', 'P99 E2E (s)', 'Mean TTFT (s)',
                     'Throughput (tok/s)', 'Load Imbalance']

        # Add cache metrics if available
        if 'cache_hit_rate_token' in df.columns:
            agg_dict['cache_hit_rate_token'] = 'mean'
            agg_dict['cache_hit_rate_request'] = 'mean'
            col_names.extend(['Cache % (token)', 'Cache % (req)'])

        summary = df.groupby('scheduler').agg(agg_dict).round(4)

        # Format for display
        summary.columns = col_names

        # Create table
        table = ax.table(
            cellText=summary.values,
            rowLabels=summary.index,
            colLabels=summary.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(summary.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight best values
        for col_idx, col in enumerate(summary.columns):
            values = summary[col].values
            if 'Imbalance' in col or 'E2E' in col or 'TTFT' in col:
                best_idx = np.argmin(values)  # Lower is better
            elif 'Cache' in col or 'Throughput' in col:
                best_idx = np.argmax(values)  # Higher is better
            else:
                best_idx = np.argmax(values)  # Default: higher is better

            table[(best_idx + 1, col_idx)].set_facecolor('#FFE082')
            table[(best_idx + 1, col_idx)].set_text_props(weight='bold')

        plt.title("Scheduler Performance Summary", fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple scheduling policies across workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all schedulers on all workloads
  python scripts/evaluate_schedulers.py --output_dir results/eval_full

  # Quick test with specific schedulers
  python scripts/evaluate_schedulers.py --schedulers rl_cache_aware round_robin --quick

  # Multiple runs for statistical significance
  python scripts/evaluate_schedulers.py --num_runs 3 --output_dir results/eval_3runs

  # Specific workload
  python scripts/evaluate_schedulers.py --workloads splitwise_conv
        """
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=f'results/eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Output directory for results and plots'
    )

    parser.add_argument(
        '--schedulers',
        nargs='+',
        choices=list(SchedulerEvaluator.SCHEDULERS.keys()),
        help='Schedulers to evaluate (default: all)'
    )

    parser.add_argument(
        '--workloads',
        nargs='+',
        choices=list(SchedulerEvaluator.WORKLOADS.keys()),
        help='Workloads to test (default: all)'
    )

    parser.add_argument(
        '--num_runs',
        type=int,
        default=1,
        help='Number of runs per configuration (default: 1)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: disable expensive metrics for faster evaluation'
    )

    parser.add_argument(
        '--rl_model_path',
        type=str,
        help='Path to RL model checkpoint for rl_cache_aware scheduler (e.g., rl_models/model_latest.pt)'
    )

    parser.add_argument(
        '--time_scale_factor',
        type=float,
        help='Scale factor for trace arrival times (higher = slower QPS, e.g., 20.0)'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = SchedulerEvaluator(
        output_dir=args.output_dir,
        schedulers=args.schedulers,
        workloads=args.workloads,
        quick=args.quick,
        rl_model_path=args.rl_model_path,
        time_scale_factor=args.time_scale_factor,
    )

    # Run evaluations
    evaluator.run_all(num_runs=args.num_runs)

    # Save results
    evaluator.save_results()

    # Generate plots
    evaluator.generate_plots()

    print(f"\n{'='*80}")
    print(f"✅ Evaluation complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 View plots in: {args.output_dir}/plots/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
