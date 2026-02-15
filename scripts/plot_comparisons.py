#!/usr/bin/env python3
"""
Create fair comparison plots - one variable at a time with all else equal.
Compares reward, architecture, replicas, trace, and model while keeping others constant.
"""

import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fetch slurm logs
print("Fetching slurm logs...")
cmd = "ssh -o ControlMaster=no alpine 'cat /home/kamil@colostate.edu/slurm_logs/vidur_grad_23669902_*.out'"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
content = result.stdout
print(f"Got {len(content):,} bytes")

# Parse per-task data
sections = re.split(r'Vidur Graduation Track: (\d+)', content)
training_history = defaultdict(list)
iteration_counters = defaultdict(int)

for i in range(1, len(sections), 2):
    if i + 1 >= len(sections):
        continue
    task_id = int(sections[i])
    section_text = sections[i + 1]
    
    # Extract config
    config_match = re.search(r'Config:\s+(\S+)', section_text)
    reward_match = re.search(r'Reward:\s+(\S+)', section_text)
    hidden_match = re.search(r'RL Size:\s+([\d,]+)', section_text)
    
    if not config_match:
        continue
    
    config_str = config_match.group(1)
    parts = config_str.split('-')
    model = parts[0]
    replicas = parts[1]
    trace = parts[2]
    
    reward = reward_match.group(1).replace('_time', '') if reward_match else 'e2e'
    hidden = hidden_match.group(1).replace(',', '') if hidden_match else '512'
    arch = '1024' if hidden.startswith('1024') else '512'
    
    config_key = f"{model}-{replicas}-{trace}-{reward}-{arch}"
    
    # Parse iterations
    for line in section_text.split('\n'):
        match = re.search(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s', line, re.IGNORECASE)
        if match:
            scale = float(match.group(1))
            e2e = float(match.group(2))
            iteration_counters[config_key] += 1
            training_history[config_key].append({
                'iteration': iteration_counters[config_key],
                'scale': scale,
                'e2e': e2e
            })

print(f"Parsed {len(training_history)} configs\n")

# === PLOT 1: Reward Signal Comparison ===
# Keep everything constant: model=7b, replicas=8R, trace=Synth, arch=512
# Compare: e2e_time vs ttft

fig, ax = plt.subplots(figsize=(10, 6))

reward_comparisons = [
    ('7b-8R-Synth-e2e-512', '7b-8R-Synth-ttft-512'),
]

for config1, config2 in reward_comparisons:
    if config1 in training_history and config2 in training_history:
        h1 = training_history[config1]
        h2 = training_history[config2]
        
        ax.plot([h['iteration'] for h in h1], [h['e2e'] for h in h1], 
                'o-', label=config1.replace('7b-8R-Synth-', '').replace('-512', ''), 
                linewidth=2, markersize=6)
        ax.plot([h['iteration'] for h in h2], [h['e2e'] for h in h2], 
                's--', label=config2.replace('7b-8R-Synth-', '').replace('-512', ''), 
                linewidth=2, markersize=6)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Mean E2E Latency (s)', fontsize=12)
ax.set_title('Reward Signal Comparison\n(7b-8R-Synth, arch=512)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/comparison_reward.png', dpi=150)
print("Saved: comparison_reward.png")

# === PLOT 2: Architecture Comparison ===
# Compare: 512 vs 1024 for same model, replicas, trace, reward

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

arch_comparisons = [
    ('7b-8R-Synth-e2e-512', '7b-8R-Synth-e2e-1024'),
    ('7b-8R-Synth-ttft-512', '7b-8R-Synth-ttft-1024'),
]

for idx, (c512, c1024) in enumerate(arch_comparisons):
    ax = axes[idx]
    
    if c512 in training_history:
        h = training_history[c512]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                'o-', label='512 arch', linewidth=2, markersize=6)
    
    if c1024 in training_history:
        h = training_history[c1024]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                's--', label='1024 arch', linewidth=2, markersize=6)
    
    reward = 'e2e' if 'e2e' in c512 else 'ttft'
    ax.set_title(f'Architecture: 512 vs 1024\n({reward} reward)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/comparison_architecture.png', dpi=150)
print("Saved: comparison_architecture.png")

# === PLOT 3: Replica Count Comparison ===
# Compare: 8R vs 16R for same model, trace, reward, arch

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

replica_comparisons = [
    ('7b-8R-Synth-e2e-512', '7b-16R-Synth-e2e-512'),
    ('7b-8R-Synth-ttft-512', '7b-16R-Synth-ttft-512'),
    ('7b-8R-Tool-e2e-512', '7b-16R-Tool-e2e-512'),
    ('7b-8R-Tool-ttft-512', '7b-16R-Tool-ttft-512'),
]

for idx, (c8r, c16r) in enumerate(replica_comparisons):
    ax = axes[idx // 2, idx % 2]
    
    if c8r in training_history:
        h = training_history[c8r]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                'o-', label='8R', linewidth=2, markersize=6)
    
    if c16r in training_history:
        h = training_history[c16r]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                's--', label='16R', linewidth=2, markersize=6)
    
    trace = 'Synth' if 'Synth' in c8r else 'Tool'
    reward = 'e2e' if 'e2e' in c8r else 'ttft'
    ax.set_title(f'Replicas: 8R vs 16R\n({trace}, {reward})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/comparison_replicas.png', dpi=150)
print("Saved: comparison_replicas.png")

# === PLOT 4: Trace Type Comparison ===
# Compare: Synth vs Tool vs Conv for same model, replicas, reward, arch

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

trace_comparisons = [
    ('7b-8R-Synth-e2e-512', '7b-8R-Tool-e2e-512', '7b-8R-Conv-e2e-512'),
    ('7b-8R-Synth-ttft-512', '7b-8R-Tool-ttft-512', '7b-8R-Conv-ttft-512'),
    ('7b-16R-Synth-e2e-512', '7b-16R-Tool-e2e-512', '7b-16R-Conv-e2e-512'),
    ('7b-16R-Synth-ttft-512', '7b-16R-Tool-ttft-512', '7b-16R-Conv-ttft-512'),
]

for idx, (csynth, ctool, cconv) in enumerate(trace_comparisons):
    ax = axes[idx // 2, idx % 2]
    replicas = '8R' if '8R' in csynth else '16R'
    reward = 'e2e' if 'e2e' in csynth else 'ttft'
    
    colors = {'Synth': 'green', 'Tool': 'orange', 'Conv': 'purple'}
    markers = {'Synth': 'o', 'Tool': 's', 'Conv': '^'}
    
    for config, trace_name in [(csynth, 'Synth'), (ctool, 'Tool'), (cconv, 'Conv')]:
        if config in training_history:
            h = training_history[config]
            ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                    f'{markers[trace_name]}-', color=colors[trace_name], 
                    label=trace_name, linewidth=2, markersize=6)
    
    ax.set_title(f'Trace: Synth vs Tool vs Conv\n({replicas}, {reward})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/comparison_trace.png', dpi=150)
print("Saved: comparison_trace.png")

# === PLOT 5: Model Comparison ===
# Compare: 7b vs 70b for same replicas, trace, reward, arch

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

model_comparisons = [
    ('7b-8R-Synth-e2e-512', '70b-8R-Synth-e2e-512'),
    ('7b-8R-Synth-ttft-512', '70b-8R-Synth-ttft-512'),
    ('7b-16R-Synth-e2e-512', '70b-16R-Synth-e2e-512'),
    ('7b-16R-Synth-ttft-512', '70b-16R-Synth-ttft-512'),
]

for idx, (c7b, c70b) in enumerate(model_comparisons):
    ax = axes[idx // 2, idx % 2]
    replicas = '8R' if '8R' in c7b else '16R'
    reward = 'e2e' if 'e2e' in c7b else 'ttft'
    
    if c7b in training_history:
        h = training_history[c7b]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                'o-', label='7b', linewidth=2, markersize=6, color='blue')
    
    if c70b in training_history:
        h = training_history[c70b]
        ax.plot([x['iteration'] for x in h], [x['e2e'] for x in h], 
                's--', label='70b', linewidth=2, markersize=6, color='red')
    
    ax.set_title(f'Model: 7b vs 70b\n({replicas}, Synth, {reward})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E2E Latency (s)')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/comparison_model.png', dpi=150)
print("Saved: comparison_model.png")

print("\n" + "="*60)
print("All comparison plots generated!")
print("="*60)
