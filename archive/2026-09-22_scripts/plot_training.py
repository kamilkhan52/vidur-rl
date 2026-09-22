#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse training log
steps = []
losses = []
rewards = []
epsilons = []

with open('/tmp/rl_training_logs/training.log', 'r') as f:
    for line in f:
        match = re.search(r'Steps: (\d+).*Loss: ([\d.]+).*Reward: (-?[\d.]+).*Epsilon: ([\d.]+)', line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            rewards.append(float(match.group(3)))
            epsilons.append(float(match.group(4)))

# Parse training history
iter_steps = []
iter_e2e = []

with open('/tmp/rl_training_logs/training_history.txt', 'r') as f:
    for line in f:
        match = re.search(r'Iter \d+ \| Steps: (\d+) \| E2E: ([\d.]+)s', line)
        if match:
            iter_steps.append(int(match.group(1)))
            iter_e2e.append(float(match.group(2)))

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss over time
axes[0, 0].plot(steps, losses, linewidth=1.5, alpha=0.7)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title(f'Training Loss (Steps {steps[0]}-{steps[-1]})')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(left=0)

# Plot 2: Reward over time
axes[0, 1].plot(steps, rewards, linewidth=1.5, alpha=0.7, color='green')
axes[0, 1].set_xlabel('Training Steps')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].set_title(f'Training Reward (Steps {steps[0]}-{steps[-1]})')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(left=0)

# Plot 3: Epsilon decay
axes[1, 0].plot(steps, epsilons, linewidth=1.5, alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Training Steps')
axes[1, 0].set_ylabel('Epsilon')
axes[1, 0].set_title(f'Epsilon Decay (Steps {steps[0]}-{steps[-1]})')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(left=0)

# Plot 4: E2E latency over iterations
if iter_steps:
    axes[1, 1].plot(iter_steps, iter_e2e, marker='o', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('E2E Latency (s)')
    axes[1, 1].set_title('End-to-End Latency After Each Iteration')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(left=0)

plt.tight_layout()
plt.savefig('/tmp/rl_training_plots.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to /tmp/rl_training_plots.png")

# Print summary statistics
print(f"\n=== Training Summary ===")
print(f"Training steps logged: {steps[0]} to {steps[-1]}")
print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
print(f"Reward: {rewards[0]:.4f} → {rewards[-1]:.4f}")
print(f"Epsilon: {epsilons[0]:.4f} → {epsilons[-1]:.4f}")
if iter_e2e:
    print(f"\nIteration history ({len(iter_e2e)} iterations):")
    print(f"E2E Latency: {iter_e2e[0]:.3f}s → {iter_e2e[-1]:.3f}s ({(iter_e2e[-1]/iter_e2e[0]-1)*100:+.1f}%)")
    print(f"\nNote: Training log only contains steps {steps[0]}+")
    print(f"      Earlier iterations' logs were not preserved")
