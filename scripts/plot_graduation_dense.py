#!/usr/bin/env python3
"""
Dense, clean graduation progress plot.
- X-axis inverted: Scale 1.5 (left) → Scale 48 (right)
- Shows E2E for each track with current position marked
- All on one compact figure
"""

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

# Parse data by task
data_by_task = {}
current_task = None

sections = re.split(r'Vidur Graduation Track: (\d+)', content)
for i in range(1, len(sections), 2):
    if i+1 >= len(sections):
        continue
    task_id = int(sections[i])
    text = sections[i+1]
    
    config_match = re.search(r'Config:\s+(\S+)', text)
    if not config_match:
        continue
    
    config = config_match.group(1)
    matches = re.findall(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s', text)
    
    scales_e2e = [(float(s), float(e)) for s, e in matches]
    if scales_e2e:
        data_by_task[task_id] = {
            'config': config,
            'data': sorted(scales_e2e),  # Sort by scale ascending
            'current_scale': scales_e2e[-1][0] if scales_e2e else None
        }

print(f"Parsed {len(data_by_task)} tasks")

# Create dense, clean plot
fig, ax = plt.subplots(figsize=(14, 8))

# Color map for configs
unique_configs = sorted(set(d['config'] for d in data_by_task.values()))
cmap = plt.cm.tab20(np.linspace(0, 1, len(unique_configs)))
color_map = {c: cmap[i] for i, c in enumerate(unique_configs)}

# Plot each track
for task_id, info in sorted(data_by_task.items()):
    scales = [s for s, e in info['data']]
    e2es = [e for s, e in info['data']]
    current = info['current_scale']
    
    # Plot line with dots
    ax.plot(scales, e2es, 'o-', color=color_map[info['config']], 
            linewidth=1.5, markersize=4, alpha=0.7, label=info['config'])
    
    # Mark current position with larger marker
    if current:
        current_e2e = next((e for s, e in info['data'] if s == current), None)
        if current_e2e:
            ax.scatter([current], [current_e2e], s=100, color=color_map[info['config']], 
                      edgecolors='black', linewidths=2, zorder=10)

# Invert x-axis so graduation goes left→right (1.5→48)
ax.invert_xaxis()

# Log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Labels
ax.set_xlabel('Request Scale (1.5 ← Graduating → 48)', fontsize=12)
ax.set_ylabel('Mean E2E Latency (s)', fontsize=12)
ax.set_title('Graduation Progress: All Tracks (● = Current Position)', fontsize=14, fontweight='bold')

# Add tension zone shading
ax.axvspan(1.5, 4, alpha=0.1, color='red', label='Tension Zone')
ax.text(2.5, ax.get_ylim()[1]*0.7, 'Tension\nZone', ha='center', fontsize=10, color='red', alpha=0.8)

# Legend outside plot
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)

# Grid
ax.grid(True, alpha=0.3, which='both')

# Tight
plt.tight_layout()

# Save
output = "/home/kamil/.openclaw/workspace/graduation_progress_dense.png"
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output}")

# Print summary
print("\n=== Current Positions ===")
for task_id in sorted(data_by_task.keys())[:10]:
    info = data_by_task[task_id]
    print(f"Task {task_id:2d}: {info['config']:20s} @ Scale {info['current_scale']}")
print("...")
