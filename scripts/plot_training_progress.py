#!/usr/bin/env python3
"""
Generate training progress plot from simulation logs.

Usage:
    python scripts/plot_training_progress.py <output_path>

The output_path should be inside the training log directory, e.g.:
    /tmp/mooncake_training_logs/16gpu_toolagent_scale3.0_20260122_103217_6118/training_progress.png

The script will:
1. Extract the checkpoint name from the log directory (e.g., "16gpu_toolagent_scale3.0")
2. Find all result directories that trained this specific model
3. Plot only data from those directories
"""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_checkpoint_name(log_dir):
    """
    Extract checkpoint name from log directory.

    Log dir format: /tmp/mooncake_training_logs/{checkpoint_name}_{date}_{time}_{pid}
    Examples:
        - 16gpu_toolagent_scale3.0_20260122_103217_6118 -> 16gpu_toolagent_scale3.0
        - mooncake_continuous_conversation_v2_scale20.0_20260122_... -> mooncake_continuous_conversation_v2_scale20.0
    """
    dirname = os.path.basename(log_dir)
    # Match pattern: name_YYYYMMDD_HHMMSS_PID
    match = re.match(r'^(.+)_(\d{8})_(\d{6})_(\d+)$', dirname)
    if match:
        return match.group(1)
    return None


def find_simulation_logs_for_checkpoint(checkpoint_name):
    """Find all simulation logs for a specific checkpoint/model."""
    if not checkpoint_name:
        return []

    # Build patterns based on checkpoint name
    # Handle both naming conventions:
    # - results/16gpu_toolagent_training_iter*_...
    # - results/mooncake_continuous_iter*_...

    patterns = []

    if checkpoint_name.startswith("16gpu_"):
        # New naming: 16gpu_toolagent_scale3.0 -> results/16gpu_toolagent_training_iter*
        base_name = checkpoint_name.rsplit("_scale", 1)[0]  # 16gpu_toolagent
        patterns.append(f"results/{base_name}_training_iter*_*/simulation.log")
    elif checkpoint_name.startswith("mooncake_continuous_"):
        # Old naming: mooncake_continuous_conversation_v2_scale20.0 -> results/mooncake_continuous_iter*
        patterns.append("results/mooncake_continuous_iter*_*/simulation.log")
    else:
        # Fallback: try to match based on checkpoint name prefix
        patterns.append(f"results/{checkpoint_name.split('_scale')[0]}*iter*_*/simulation.log")

    log_files = []
    for pattern in patterns:
        log_files.extend(glob.glob(pattern))

    return sorted(log_files, key=lambda x: os.path.getmtime(x))


def parse_training_history(log_dir):
    """Parse training history from the specific log directory."""
    history_file = os.path.join(log_dir, "training_history.txt")

    if not os.path.exists(history_file):
        return None

    iterations = []
    steps = []
    e2e_latencies = []

    with open(history_file) as f:
        for line in f:
            match = re.match(r'Iter (\d+) \| Steps: (\d+) \| E2E: ([\d.]+)s', line)
            if match:
                iterations.append(int(match.group(1)))
                steps.append(int(match.group(2)))
                e2e_latencies.append(float(match.group(3)))

    if not iterations:
        return None

    return {
        'iterations': iterations,
        'steps': steps,
        'e2e_latencies': e2e_latencies,
        'history_file': history_file
    }


def extract_iteration_data(log_files, log_dir):
    """Extract training metrics from simulation logs."""
    all_data = []

    # Get the start time of the current run from log_dir
    log_dir_basename = os.path.basename(log_dir)
    run_start_match = re.search(r'_(\d{8}_\d{6})_\d+$', log_dir_basename)
    run_start_time = run_start_match.group(1) if run_start_match else None

    for log_file in log_files:
        match = re.search(r'iter(\d+)_(\d{8}_\d{6})', log_file)
        if not match:
            continue

        iter_num = int(match.group(1))
        timestamp = match.group(2)

        # Only include logs from this run or later (for resumed training)
        # This filters out old runs with the same checkpoint name
        if run_start_time and timestamp < run_start_time:
            continue

        result_dir = os.path.dirname(log_file)

        # Parse simulation log
        losses = []
        rewards = []
        final_step = 0
        final_epsilon = 1.0

        try:
            with open(log_file) as f:
                for line in f:
                    if "Steps:" in line and "Loss:" in line:
                        step_match = re.search(r'Steps:\s+(\d+)', line)
                        loss_match = re.search(r'Loss:\s+([\d.]+)', line)
                        reward_match = re.search(r'Reward:\s+([-\d.]+)', line)
                        epsilon_match = re.search(r'Epsilon:\s+([\d.]+)', line)

                        if step_match and loss_match and reward_match:
                            final_step = int(step_match.group(1))
                            losses.append(float(loss_match.group(1)))
                            rewards.append(float(reward_match.group(1)))
                            if epsilon_match:
                                final_epsilon = float(epsilon_match.group(1))
        except Exception:
            continue

        # Get metrics from request_metrics.csv
        metrics_files = glob.glob(f"{result_dir}/**/request_metrics.csv", recursive=True)
        p99_e2e = None
        mean_e2e = None
        p50_e2e = None

        if metrics_files:
            try:
                df = pd.read_csv(metrics_files[0])
                p99_e2e = df['request_e2e_time'].quantile(0.99)
                p50_e2e = df['request_e2e_time'].quantile(0.50)
                mean_e2e = df['request_e2e_time'].mean()
            except Exception:
                pass

        avg_loss = np.mean(losses) if losses else None
        avg_reward = np.mean(rewards) if rewards else None

        all_data.append({
            'iteration': iter_num,
            'timestamp': timestamp,
            'final_step': final_step,
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'final_epsilon': final_epsilon,
            'mean_e2e': mean_e2e,
            'p50_e2e': p50_e2e,
            'p99_e2e': p99_e2e,
        })

    # Sort by timestamp and assign sequential numbers
    all_data = sorted(all_data, key=lambda x: x['timestamp'])
    for i, d in enumerate(all_data):
        d['seq'] = i + 1

    return pd.DataFrame(all_data)


def generate_plot(output_path, log_dir):
    """Generate the training progress plot for a specific training run."""
    checkpoint_name = extract_checkpoint_name(log_dir)

    if not checkpoint_name:
        print(f"Could not extract checkpoint name from: {log_dir}")
        return False

    print(f"Checkpoint name: {checkpoint_name}")

    log_files = find_simulation_logs_for_checkpoint(checkpoint_name)

    if not log_files:
        print(f"No simulation logs found for checkpoint: {checkpoint_name}")
        return False

    print(f"Found {len(log_files)} log files")

    df = extract_iteration_data(log_files, log_dir)
    df_valid = df.dropna(subset=['avg_loss', 'p99_e2e'])

    if len(df_valid) < 2:
        print(f"Not enough data points ({len(df_valid)} iterations with complete data)")
        return False

    print(f"Plotting {len(df_valid)} iterations")

    # Also get history data for E2E trend
    history = parse_training_history(log_dir)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'RL Training Progress: {checkpoint_name}', fontsize=14, fontweight='bold')

    seq = df_valid['seq'].values

    # 1. Loss over iterations
    ax = axes[0, 0]
    ax.plot(seq, df_valid['avg_loss'], 'r-o', markersize=4, linewidth=1.5, label='Avg Loss')
    z_loss = np.polyfit(seq, df_valid['avg_loss'].values, 1)
    p = np.poly1d(z_loss)
    ax.plot(seq, p(seq), 'k--', alpha=0.7, label=f'Trend: {z_loss[0]:+.5f}/iter')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Reward over iterations
    ax = axes[0, 1]
    ax.plot(seq, df_valid['avg_reward'], 'b-o', markersize=4, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    z_reward = np.polyfit(seq, df_valid['avg_reward'].values, 1)
    p = np.poly1d(z_reward)
    ax.plot(seq, p(seq), 'k--', alpha=0.7, label=f'Trend: {z_reward[0]:+.5f}/iter')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward (neg latency)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Epsilon decay
    ax = axes[0, 2]
    ax.plot(seq, df_valid['final_epsilon'], 'g-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.set_ylim([0, max(0.1, df_valid['final_epsilon'].max() * 1.1)])
    ax.grid(True, alpha=0.3)

    # 4. Mean E2E over iterations (use history if available for cleaner data)
    ax = axes[1, 0]
    if history and len(history['iterations']) >= len(seq):
        h_iter = history['iterations'][-len(seq):]
        h_e2e = history['e2e_latencies'][-len(seq):]
        ax.plot(range(1, len(h_e2e)+1), h_e2e, 'purple', marker='o', markersize=4, linewidth=1.5, label='Mean E2E')
        mean_val = np.mean(h_e2e)
    else:
        ax.plot(seq, df_valid['mean_e2e'], 'purple', marker='o', markersize=4, linewidth=1.5, label='Mean E2E')
        mean_val = df_valid['mean_e2e'].mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}s')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean E2E Latency (s)')
    ax.set_title('Mean E2E Latency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. P99 E2E over iterations (exclude extreme outliers for better viz)
    ax = axes[1, 1]
    p99_vals = df_valid['p99_e2e'].values
    # Cap extreme values for visualization
    p99_cap = np.percentile(p99_vals, 95) * 2
    p99_display = np.minimum(p99_vals, p99_cap)

    ax.plot(seq, p99_display, 'orange', marker='o', markersize=4, linewidth=1.5, label='P99 E2E')

    # Only fit trend on non-outlier values
    mask = p99_vals < p99_cap
    if mask.sum() > 2:
        z_p99 = np.polyfit(seq[mask], p99_vals[mask], 1)
        p = np.poly1d(z_p99)
        ax.plot(seq[mask], p(seq[mask]), 'k--', alpha=0.7, label=f'Trend: {z_p99[0]:+.2f}s/iter')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('P99 E2E Latency (s)')
    ax.set_title('P99 E2E Latency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark if there are capped values
    outlier_count = (~mask).sum()
    if outlier_count > 0:
        ax.text(0.02, 0.98, f'{outlier_count} outliers capped', transform=ax.transAxes,
                fontsize=8, verticalalignment='top', color='red')

    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')

    latest = df_valid.iloc[-1]
    best_loss_idx = df_valid['avg_loss'].idxmin()
    best_p99_idx = df_valid['p99_e2e'].idxmin()

    # Calculate cumulative steps
    cumulative_steps = df_valid['final_step'].sum()

    summary = f"""
    Training Summary (Iter {int(latest['seq'])})
    ════════════════════════════════

    Loss:
      Current: {latest['avg_loss']:.4f}
      Best:    {df_valid.loc[best_loss_idx, 'avg_loss']:.4f} (iter {int(df_valid.loc[best_loss_idx, 'seq'])})
      Trend:   {z_loss[0]:+.5f}/iter {'↓' if z_loss[0] < 0 else '↑'}

    P99 Latency:
      Current: {latest['p99_e2e']:.2f}s
      Best:    {df_valid.loc[best_p99_idx, 'p99_e2e']:.2f}s (iter {int(df_valid.loc[best_p99_idx, 'seq'])})

    Mean E2E:
      Current: {latest['mean_e2e']:.2f}s

    Epsilon:  {latest['final_epsilon']:.4f}
    Steps:    {int(cumulative_steps)}

    Status: {'✓ Learning' if z_loss[0] < 0 else '⚠ Check loss'}
    """

    ax.text(0.1, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_training_progress.py <output_path>")
        print("  output_path should be inside the training log directory")
        return 1

    output_path = sys.argv[1]
    log_dir = os.path.dirname(output_path)

    if not log_dir or not os.path.isdir(log_dir):
        print(f"Error: Could not determine log directory from output path: {output_path}")
        return 1

    success = generate_plot(output_path, log_dir)

    if success:
        print(f"Plot saved: {output_path}")
        return 0
    else:
        print("Failed to generate plot")
        return 1


if __name__ == "__main__":
    sys.exit(main())
