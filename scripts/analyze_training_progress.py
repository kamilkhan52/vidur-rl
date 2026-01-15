#!/usr/bin/env python3
"""
Analyze RL training progress from logs and simulation outputs.

Extracts and visualizes:
- Loss over training steps
- Reward over requests
- E2E latency over time
- Epsilon decay
- Load balance over time

Usage:
    python scripts/analyze_training_progress.py results/rl_training/
    python scripts/analyze_training_progress.py --log /tmp/rl_training_logs/training.log
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class TrainingAnalyzer:
    """Analyze RL training progress."""

    def __init__(self, result_dir: Optional[Path] = None, log_file: Optional[Path] = None):
        """
        Initialize analyzer.

        Args:
            result_dir: Directory with simulation results
            log_file: Path to training.log file
        """
        self.result_dir = result_dir
        self.log_file = log_file or Path("/tmp/rl_training_logs/training.log")

        # Parsed data
        self.training_steps = []
        self.request_counts = []
        self.losses = []
        self.rewards = []
        self.epsilons = []
        self.buffer_sizes = []

        self.request_metrics = None

    def parse_training_log(self) -> bool:
        """
        Parse training log file.

        Returns:
            True if successful, False otherwise
        """
        if not self.log_file.exists():
            print(f"Training log not found: {self.log_file}")
            return False

        print(f"Parsing training log: {self.log_file}")

        with open(self.log_file) as f:
            for line in f:
                # Parse training progress lines
                # Format: [Training] Steps: 100 | Requests: 10019 | Loss: 0.0655 | Reward: -0.1547 | Epsilon: 0.810 | Buffer: 10000
                if "Steps:" in line and "Loss:" in line:
                    try:
                        steps_match = re.search(r'Steps:\s+(\d+)', line)
                        requests_match = re.search(r'Requests:\s+(\d+)', line)
                        loss_match = re.search(r'Loss:\s+([\d.]+)', line)
                        reward_match = re.search(r'Reward:\s+([-\d.]+)', line)
                        epsilon_match = re.search(r'Epsilon:\s+([\d.]+)', line)
                        buffer_match = re.search(r'Buffer:\s+(\d+)', line)

                        if all([steps_match, requests_match, loss_match, reward_match, epsilon_match, buffer_match]):
                            self.training_steps.append(int(steps_match.group(1)))
                            self.request_counts.append(int(requests_match.group(1)))
                            self.losses.append(float(loss_match.group(1)))
                            self.rewards.append(float(reward_match.group(1)))
                            self.epsilons.append(float(epsilon_match.group(1)))
                            self.buffer_sizes.append(int(buffer_match.group(1)))
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()}")
                        print(f"Error: {e}")
                        continue

        print(f"Parsed {len(self.training_steps)} training step records")
        return len(self.training_steps) > 0

    def load_request_metrics(self) -> bool:
        """
        Load request metrics from simulation output.

        Returns:
            True if successful, False otherwise
        """
        if not self.result_dir:
            print("No result directory specified")
            return False

        # Find request_metrics.csv
        metrics_files = list(self.result_dir.glob("**/request_metrics.csv"))

        if not metrics_files:
            print(f"No request_metrics.csv found in {self.result_dir}")
            return False

        # Use the most recent one
        metrics_file = sorted(metrics_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Loading request metrics: {metrics_file}")

        self.request_metrics = pd.read_csv(metrics_file)
        print(f"Loaded {len(self.request_metrics)} request records")
        return True

    def plot_training_progress(self, output_dir: Path):
        """
        Generate training progress plots.

        Args:
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.training_steps:
            print("No training data to plot")
            return

        # Create 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')

        # 1. Loss over training steps
        ax = axes[0, 0]
        ax.plot(self.training_steps, self.losses, linewidth=2, color='#e74c3c')
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title('Training Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(self.training_steps) > 1:
            z = np.polyfit(self.training_steps, self.losses, 1)
            p = np.poly1d(z)
            ax.plot(self.training_steps, p(self.training_steps), "--",
                   alpha=0.5, color='#c0392b', label=f'Trend: {z[0]:.6f}x + {z[1]:.4f}')
            ax.legend(fontsize=9)

        # 2. Reward over requests
        ax = axes[0, 1]
        ax.plot(self.request_counts, self.rewards, linewidth=2, color='#3498db')
        ax.set_xlabel('Requests Completed', fontsize=11)
        ax.set_ylabel('Average Reward', fontsize=11)
        ax.set_title('Average Reward (Last 100)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Add trend line
        if len(self.request_counts) > 1:
            z = np.polyfit(self.request_counts, self.rewards, 1)
            p = np.poly1d(z)
            ax.plot(self.request_counts, p(self.request_counts), "--",
                   alpha=0.5, color='#2980b9', label=f'Trend: {z[0]:.8f}x + {z[1]:.4f}')
            ax.legend(fontsize=9)

        # 3. Epsilon decay
        ax = axes[0, 2]
        ax.plot(self.request_counts, self.epsilons, linewidth=2, color='#2ecc71')
        ax.set_xlabel('Requests Completed', fontsize=11)
        ax.set_ylabel('Epsilon', fontsize=11)
        ax.set_title('Exploration Rate (Epsilon)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # 4. Buffer size
        ax = axes[1, 0]
        ax.plot(self.request_counts, self.buffer_sizes, linewidth=2, color='#9b59b6')
        ax.set_xlabel('Requests Completed', fontsize=11)
        ax.set_ylabel('Buffer Size', fontsize=11)
        ax.set_title('Replay Buffer Size', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. Loss vs Reward scatter
        ax = axes[1, 1]
        scatter = ax.scatter(self.losses, self.rewards, c=self.training_steps,
                           cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel('Loss', fontsize=11)
        ax.set_ylabel('Average Reward', fontsize=11)
        ax.set_title('Loss vs Reward (colored by step)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Training Steps', fontsize=10)

        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')

        stats_text = f"""
        Training Summary

        Total Training Steps: {max(self.training_steps) if self.training_steps else 0}
        Total Requests: {max(self.request_counts) if self.request_counts else 0}

        Loss:
          Initial: {self.losses[0]:.4f}
          Final: {self.losses[-1]:.4f}
          Best: {min(self.losses):.4f}
          Change: {((self.losses[-1] - self.losses[0]) / self.losses[0] * 100):.1f}%

        Reward:
          Initial: {self.rewards[0]:.4f}
          Final: {self.rewards[-1]:.4f}
          Best: {max(self.rewards):.4f}
          Change: {((self.rewards[-1] - self.rewards[0]) / abs(self.rewards[0]) * 100):.1f}%

        Epsilon:
          Initial: {self.epsilons[0]:.3f}
          Final: {self.epsilons[-1]:.3f}

        Buffer: {self.buffer_sizes[-1]} / {max(self.buffer_sizes)}
        """

        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center')

        plt.tight_layout()
        output_path = output_dir / "training_progress.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_request_latency_over_time(self, output_dir: Path):
        """
        Plot request latency evolution during training.

        Args:
            output_dir: Directory to save plots
        """
        if self.request_metrics is None or len(self.request_metrics) == 0:
            print("No request metrics to plot")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort by arrival time
        df = self.request_metrics.sort_values('request_arrived_at')

        # Compute rolling averages
        window = 100
        df['rolling_e2e'] = df['request_e2e_time'].rolling(window=window, min_periods=1).mean()
        df['rolling_ttft'] = df['prefill_e2e_time'].rolling(window=window, min_periods=1).mean()

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Request Latency Evolution During Training', fontsize=16, fontweight='bold')

        # 1. E2E latency over time
        ax = axes[0, 0]
        ax.scatter(df['request_arrived_at'], df['request_e2e_time'],
                  alpha=0.3, s=10, label='Individual requests')
        ax.plot(df['request_arrived_at'], df['rolling_e2e'],
               color='red', linewidth=2, label=f'Rolling avg (window={window})')
        ax.set_xlabel('Arrival Time (s)', fontsize=11)
        ax.set_ylabel('E2E Latency (s)', fontsize=11)
        ax.set_title('End-to-End Latency Over Time', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. TTFT over time
        ax = axes[0, 1]
        ax.scatter(df['request_arrived_at'], df['prefill_e2e_time'],
                  alpha=0.3, s=10, label='Individual requests')
        ax.plot(df['request_arrived_at'], df['rolling_ttft'],
               color='blue', linewidth=2, label=f'Rolling avg (window={window})')
        ax.set_xlabel('Arrival Time (s)', fontsize=11)
        ax.set_ylabel('TTFT (s)', fontsize=11)
        ax.set_title('Time-to-First-Token Over Time', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Latency distribution by time periods
        ax = axes[1, 0]
        num_periods = 4
        df['period'] = pd.cut(df['request_arrived_at'], bins=num_periods,
                             labels=[f'Period {i+1}' for i in range(num_periods)])

        df.boxplot(column='request_e2e_time', by='period', ax=ax)
        ax.set_xlabel('Time Period', fontsize=11)
        ax.set_ylabel('E2E Latency (s)', fontsize=11)
        ax.set_title('Latency Distribution by Time Period', fontweight='bold')
        plt.suptitle('')  # Remove automatic title from boxplot

        # 4. Load balance over time
        ax = axes[1, 1]
        # Compute load distribution in time windows
        num_windows = 10
        df['time_window'] = pd.cut(df['request_arrived_at'], bins=num_windows)
        load_by_window = df.groupby(['time_window', 'replica']).size().unstack(fill_value=0)

        load_by_window.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Time Window', fontsize=11)
        ax.set_ylabel('Request Count', fontsize=11)
        ax.set_title('Load Distribution Over Time', fontweight='bold')
        ax.legend(title='Replica', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_path = output_dir / "latency_over_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def print_summary(self):
        """Print analysis summary to console."""
        print("\n" + "=" * 80)
        print("TRAINING ANALYSIS SUMMARY")
        print("=" * 80)

        if self.training_steps:
            print(f"\n📊 Training Progress:")
            print(f"  Total steps: {max(self.training_steps)}")
            print(f"  Total requests: {max(self.request_counts)}")
            print(f"\n📉 Loss:")
            print(f"  Initial: {self.losses[0]:.4f}")
            print(f"  Final: {self.losses[-1]:.4f}")
            print(f"  Best: {min(self.losses):.4f}")
            print(f"  Change: {((self.losses[-1] - self.losses[0]) / self.losses[0] * 100):.1f}%")

            print(f"\n🎯 Reward (negative latency):")
            print(f"  Initial: {self.rewards[0]:.4f}")
            print(f"  Final: {self.rewards[-1]:.4f}")
            print(f"  Best: {max(self.rewards):.4f}")
            print(f"  Change: {((self.rewards[-1] - self.rewards[0]) / abs(self.rewards[0]) * 100):.1f}%")

            print(f"\n🎲 Exploration:")
            print(f"  Initial epsilon: {self.epsilons[0]:.3f}")
            print(f"  Final epsilon: {self.epsilons[-1]:.3f}")

            # Determine learning status
            loss_improving = self.losses[-1] < self.losses[0]
            reward_improving = self.rewards[-1] > self.rewards[0]

            print(f"\n✅ Learning Status:")
            print(f"  Loss trend: {'✓ Decreasing' if loss_improving else '✗ Not decreasing'}")
            print(f"  Reward trend: {'✓ Increasing' if reward_improving else '✗ Not increasing'}")

            if loss_improving and reward_improving:
                print(f"  Overall: ✓ Agent is learning!")
            elif loss_improving or reward_improving:
                print(f"  Overall: ⚠️ Partial learning (check plots)")
            else:
                print(f"  Overall: ✗ No clear learning signal")

        if self.request_metrics is not None:
            print(f"\n⏱️  Request Latency:")
            print(f"  Mean E2E: {self.request_metrics['request_e2e_time'].mean():.3f}s")
            print(f"  P50 E2E: {self.request_metrics['request_e2e_time'].quantile(0.5):.3f}s")
            print(f"  P90 E2E: {self.request_metrics['request_e2e_time'].quantile(0.9):.3f}s")
            print(f"  P99 E2E: {self.request_metrics['request_e2e_time'].quantile(0.99):.3f}s")

            # Check if latency improves over time
            df = self.request_metrics.sort_values('request_arrived_at')
            early = df.head(1000)['request_e2e_time'].mean()
            late = df.tail(1000)['request_e2e_time'].mean()
            improvement = (early - late) / early * 100

            print(f"\n📈 Latency Evolution:")
            print(f"  Early (first 1k): {early:.3f}s")
            print(f"  Late (last 1k): {late:.3f}s")
            print(f"  Change: {improvement:.1f}%")
            if improvement > 0:
                print(f"  Status: ✓ Latency improved over time")
            else:
                print(f"  Status: ✗ Latency degraded over time")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RL training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from result directory
  python scripts/analyze_training_progress.py results/rl_training/

  # Analyze from specific log file
  python scripts/analyze_training_progress.py --log /tmp/rl_training_logs/training.log

  # Both
  python scripts/analyze_training_progress.py results/rl_training/ --log /tmp/rl_training_logs/training.log
        """
    )

    parser.add_argument(
        'result_dir',
        type=str,
        nargs='?',
        help='Directory with simulation results (contains request_metrics.csv)'
    )

    parser.add_argument(
        '--log',
        type=str,
        help='Path to training.log file (default: /tmp/rl_training_logs/training.log)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots/)'
    )

    args = parser.parse_args()

    # Create analyzer
    result_dir = Path(args.result_dir) if args.result_dir else None
    log_file = Path(args.log) if args.log else None

    analyzer = TrainingAnalyzer(result_dir=result_dir, log_file=log_file)

    # Parse training log
    has_training_data = analyzer.parse_training_log()

    # Load request metrics
    has_request_data = analyzer.load_request_metrics() if result_dir else False

    if not has_training_data and not has_request_data:
        print("No data found to analyze!")
        return 1

    # Generate plots
    output_dir = Path(args.output)

    if has_training_data:
        analyzer.plot_training_progress(output_dir)

    if has_request_data:
        analyzer.plot_request_latency_over_time(output_dir)

    # Print summary
    analyzer.print_summary()

    print(f"\n📊 Plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
