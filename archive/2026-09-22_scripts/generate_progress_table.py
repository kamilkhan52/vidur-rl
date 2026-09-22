#!/usr/bin/env python3
"""
Generate progress tables showing iteration, QPS (scale), and E2E for each config.
This is the status/data indicator table for the graduation sweep.
"""

import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Colors for table
HEADER_BG = '#4472C4'
HEADER_FG = 'white'
EVEN_ROW = '#D6DCE4'
ODD_ROW = 'white'
BORDER = '#000000'

# Fetch slurm logs
print("Fetching slurm logs...")
cmd = "ssh -o ControlMaster=no alpine 'cat /home/kamil@colostate.edu/slurm_logs/vidur_grad_23669902_*.out'"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
content = result.stdout
print(f"Got {len(content):,} bytes\n")

# Parse per-task data
sections = re.split(r'Vidur Graduation Track: (\d+)', content)
training_data = defaultdict(list)
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
            training_data[config_key].append({
                'iteration': iteration_counters[config_key],
                'scale': scale,
                'e2e': e2e,
                'task_id': task_id
            })

print("="*80)
print("GRADUATION PROGRESS TABLE - Iteration vs Scale vs E2E")
print("="*80)

# === PRINT ASCII TABLE ===
print(f"\n{'Config':<35} | {'Iter':>4} | {'Scale':>7} | {'E2E (s)':>10} | {'Progress'}")
print("-"*80)

for config_key in sorted(training_data.keys()):
    data = training_data[config_key]
    
    # Get last few iterations to show
    for i, d in enumerate(data):
        iter_num = d['iteration']
        scale = d['scale']
        e2e = d['e2e']
        
        # Mark scale changes
        if i == 0:
            progress = "START"
        elif d['scale'] != data[i-1]['scale']:
            progress = f"↓ {data[i-1]['scale']:.1f}→{scale:.1f}"
        else:
            progress = ""
        
        print(f"{config_key:<35} | {iter_num:>4} | {scale:>7.1f} | {e2e:>10.2f} | {progress}")

# === ANALYSIS ===
print("\n" + "="*80)
print("ANALYSIS: Iteration, Scale Changes, and Training Progress")
print("="*80)

# Group by category
synth_configs = {k: v for k, v in training_data.items() if 'Synth' in k}
tool_configs = {k: v for k, v in training_data.items() if 'Tool' in k}
conv_configs = {k: v for k, v in training_data.items() if 'Conv' in k}

print("\n### SYNTH TRACES ###")
for config_key in sorted(synth_configs.keys()):
    data = training_data[config_key]
    scales = [d['scale'] for d in data]
    unique_scales = sorted(set(scales), reverse=True)
    
    # Check if graduated
    graduated = len(unique_scales) > 1
    
    # Check improvement at each scale
    improvements = []
    current_scale = None
    scale_improvements = {}
    
    for d in data:
        if d['scale'] != current_scale:
            if current_scale is not None:
                scale_improvements[current_scale] = improvements
            current_scale = d['scale']
            improvements = []
        improvements.append(d['e2e'])
    if current_scale is not None:
        scale_improvements[current_scale] = improvements
    
    # Calculate improvement per scale
    scale_str = " → ".join([f"{s:.1f}" for s in unique_scales])
    
    print(f"\n{config_key}:")
    print(f"  Scales: {scale_str}")
    print(f"  Iterations: {len(data)}")
    
    for scale, e2es in sorted(scale_improvements.items(), key=lambda x: -x[0]):
        if len(e2es) > 1:
            first, last = e2es[0], e2es[-1]
            pct = (first - last) / first * 100 if first > 0 else 0
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s → {e2es[-1]:.2f}s ({pct:+.1f}%, n={len(e2es)})")
        else:
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s (single iter)")

print("\n### TOOL TRACES ###")
for config_key in sorted(tool_configs.keys()):
    data = training_data[config_key]
    scales = [d['scale'] for d in data]
    unique_scales = sorted(set(scales), reverse=True)
    
    print(f"\n{config_key}:")
    print(f"  Scales: {' → '.join([f'{s:.1f}' for s in unique_scales])}")
    print(f"  Iterations: {len(data)}")
    
    # Group by scale
    scale_e2es = defaultdict(list)
    for d in data:
        scale_e2es[d['scale']].append(d['e2e'])
    
    for scale in sorted(scale_e2es.keys(), reverse=True):
        e2es = scale_e2es[scale]
        if len(e2es) > 1:
            first, last = e2es[0], e2es[-1]
            pct = (first - last) / first * 100 if first > 0 else 0
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s → {e2es[-1]:.2f}s ({pct:+.1f}%, n={len(e2es)})")
        else:
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s (single iter)")

print("\n### CONV TRACES ###")
for config_key in sorted(conv_configs.keys()):
    data = training_data[config_key]
    scales = [d['scale'] for d in data]
    unique_scales = sorted(set(scales), reverse=True)
    
    print(f"\n{config_key}:")
    print(f"  Scales: {' → '.join([f'{s:.1f}' for s in unique_scales])}")
    print(f"  Iterations: {len(data)}")
    
    scale_e2es = defaultdict(list)
    for d in data:
        scale_e2es[d['scale']].append(d['e2e'])
    
    for scale in sorted(scale_e2es.keys(), reverse=True):
        e2es = scale_e2es[scale]
        if len(e2es) > 1:
            first, last = e2es[0], e2es[-1]
            pct = (first - last) / first * 100 if first > 0 else 0
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s → {e2es[-1]:.2f}s ({pct:+.1f}%, n={len(e2es)})")
        else:
            print(f"    Scale {scale:.1f}: {e2es[0]:.2f}s (single iter)")

# === CREATE VISUAL TABLE ===
def draw_table(fig, ax, data, title):
    """Draw a styled table showing iteration progression."""
    ax.axis('off')
    
    # Get unique configs
    configs = sorted(data.keys())
    
    # Prepare data: for each config, show key iterations (first, scale changes, last)
    table_data = []
    for config in configs:
        history = training_data[config]
        row_data = [config]
        
        # Show first iteration
        row_data.append(f"Iter {history[0]['iteration']}")
        row_data.append(f"{history[0]['scale']:.1f}")
        row_data.append(f"{history[0]['e2e']:.2f}")
        
        # Show key transitions or last iteration
        if len(history) > 3:
            mid_idx = len(history) // 2
            row_data.append(f"Iter {history[mid_idx]['iteration']}")
            row_data.append(f"{history[mid_idx]['scale']:.1f}")
            row_data.append(f"{history[mid_idx]['e2e']:.2f}")
        elif len(history) > 1:
            row_data.append(f"Iter {history[1]['iteration']}")
            row_data.append(f"{history[1]['scale']:.1f}")
            row_data.append(f"{history[1]['e2e']:.2f}")
        else:
            row_data.append("-")
            row_data.append("-")
            row_data.append("-")
        
        # Show last iteration
        last = history[-1]
        row_data.append(f"Iter {last['iteration']}")
        row_data.append(f"{last['scale']:.1f}")
        row_data.append(f"{last['e2e']:.2f}")
        
        table_data.append(row_data)
    
    # Headers
    headers = ['Config', 'Start Iter', 'Scale', 'E2E', 'Mid Iter', 'Scale', 'E2E', 'End Iter', 'Scale', 'E2E']
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=[HEADER_BG]*10
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color=HEADER_FG, fontweight='bold')
    
    # Style rows
    for row_idx in range(1, len(table_data) + 1):
        bg = EVEN_ROW if row_idx % 2 == 0 else ODD_ROW
        for col_idx in range(10):
            table[(row_idx, col_idx)].set_facecolor(bg)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

# Figure 1: Synth configs
fig1, ax1 = plt.subplots(figsize=(16, 10))
draw_table(fig1, ax1, synth_configs, 'Graduation Progress: SYNTH Traces\n(Iteration → Scale → E2E)')
plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/progress_table_synth.png', dpi=150, bbox_inches='tight')
print("\nSaved: progress_table_synth.png")

# Figure 2: Tool + Conv configs
fig2, ax2 = plt.subplots(figsize=(16, 12))
all_other = {**tool_configs, **conv_configs}
draw_table(fig2, ax2, all_other, 'Graduation Progress: TOOL & CONV Traces\n(Iteration → Scale → E2E)')
plt.tight_layout()
plt.savefig('/home/kamil/.openclaw/workspace/progress_table_toolconv.png', dpi=150, bbox_inches='tight')
print("Saved: progress_table_toolconv.png")

# === SUMMARY STATISTICS ===
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Configs that graduated (changed scale)
graduated_configs = []
for config_key, data in training_data.items():
    scales = [d['scale'] for d in data]
    if len(set(scales)) > 1:
        graduated_configs.append(config_key)

print(f"\nTotal configs: {len(training_data)}")
print(f"Configs that graduated (changed scale): {len(graduated_configs)}")
print(f"Configs that stayed at same scale: {len(training_data) - len(graduated_configs)}")

print("\n### FRONT-RUNNERS (Graduated to Scale ≤3) ###")
for config in sorted(graduated_configs):
    data = training_data[config]
    min_scale = min(d['scale'] for d in data)
    if min_scale <= 3:
        final = data[-1]
        print(f"  {config}: Reached Scale {min_scale:.1f}, Final E2E: {final['e2e']:.2f}s")

print("\n### STALLED (Still at Scale 48) ###")
for config in sorted(training_data.keys()):
    data = training_data[config]
    max_scale = max(d['scale'] for d in data)
    if max_scale >= 48 and config not in graduated_configs:
        print(f"  {config}: {len(data)} iterations at Scale 48")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
