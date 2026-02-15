#!/usr/bin/env python3
"""Simple comparison from slurm logs."""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

print("Fetching logs...")
cmd = "ssh -o ControlMaster=no alpine 'cat /home/kamil@colostate.edu/slurm_logs/vidur_grad_23669902_*.out'"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
content = result.stdout
print(f"Got {len(content):,} bytes")

# Parse config lines with task ID context
# Each log starts with "Vidur Graduation Track: X" followed by Config
data = defaultdict(lambda: defaultdict(list))

# Split by graduation track markers
sections = re.split(r'Vidur Graduation Track: (\d+)', content)

for i in range(1, len(sections), 2):
    if i+1 >= len(sections):
        continue
    
    task_id = int(sections[i])
    section_text = sections[i+1]
    
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
    
    # Get metrics
    matches = re.findall(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s', section_text)
    for scale, e2e in matches:
        data[config_key][float(scale)].append(float(e2e))
    
    if matches:
        print(f"Task {task_id}: {config_key} -> {len(matches)} data points")

print(f"\nParsed {len(data)} configurations")

# Group by dimension
comparisons = {
    'model': defaultdict(lambda: defaultdict(list)),
    'trace': defaultdict(lambda: defaultdict(list)),
    'reward': defaultdict(lambda: defaultdict(list)),
    'replicas': defaultdict(lambda: defaultdict(list)),
    'arch': defaultdict(lambda: defaultdict(list)),
}

for config_key, scales_dict in data.items():
    parts = config_key.split('-')
    if len(parts) == 5:
        model, replicas, trace, reward, arch = parts
        for scale, e2es in scales_dict.items():
            comparisons['model'][model][scale].extend(e2es)
            comparisons['trace'][trace][scale].extend(e2es)
            comparisons['reward'][reward][scale].extend(e2es)
            comparisons['replicas'][replicas][scale].extend(e2es)
            comparisons['arch'][arch][scale].extend(e2es)

# Print summary
print("\nData by dimension:")
for dim, dim_data in comparisons.items():
    print(f"  {dim}: {dict(dim_data)}")

# Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Graduation Training: Configuration Comparison', fontsize=14, fontweight='bold')

colors = {
    'model': {'7b': '#1f77b4', '70b': '#d62728'},
    'trace': {'Synth': '#2ca02c', 'Tool': '#ff7f0e', 'Conv': '#9467bd'},
    'reward': {'e2e': '#17becf', 'ttft': '#e377c2'},
    'replicas': {'8R': '#8c564b', '16R': '#7f7f7f'},
    'arch': {'512': '#2ca02c', '1024': '#d62728'},
}

plot_configs = [
    ('model', 0, 0, 'Model Size'),
    ('trace', 0, 1, 'Trace Type'),
    ('reward', 0, 2, 'Reward Type'),
    ('replicas', 1, 0, 'Replica Count'),
    ('arch', 1, 1, 'Architecture'),
]

for dim, row, col, title in plot_configs:
    ax = axes[row, col]
    dim_data = comparisons[dim]
    dim_colors = colors.get(dim, {})
    
    for label, scales_dict in sorted(dim_data.items()):
        scales = sorted(scales_dict.keys())
        if not scales:
            continue
        avgs = [np.mean(scales_dict[s]) for s in scales]
        stds = [np.std(scales_dict[s]) if len(scales_dict[s]) > 1 else 0 for s in scales]
        ax.errorbar(scales, avgs, yerr=stds, marker='o', label=label, 
                   color=dim_colors.get(label, '#333'), capsize=3, linewidth=2, markersize=6)
    
    ax.set_xlabel('Request Scale')
    ax.set_ylabel('Mean E2E Latency (s)')
    ax.set_title(title, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axvspan(0.5, 4, alpha=0.1, color='red')

# Summary
ax = axes[1, 2]
ax.axis('off')
summary_text = "=== Scale 3.0 (Tension Zone) ===\n\n"
for dim in ['model', 'trace', 'reward', 'replicas', 'arch']:
    dim_data = comparisons[dim]
    summary_text += f"[{dim.upper()}]\n"
    for label in sorted(dim_data.keys()):
        if 3.0 in dim_data[label]:
            e2es = dim_data[label][3.0]
            summary_text += f"  {label:>6}: {np.mean(e2es):>8.1f}s ± {np.std(e2es):>6.1f}s\n"
    summary_text += "\n"
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = "/home/kamil/.openclaw/workspace/graduation_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_path}")
