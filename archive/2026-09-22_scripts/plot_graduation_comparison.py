#!/usr/bin/env python3
"""
Compare graduation performance across configurations.
Fetches individual task logs and parses with context.
"""

import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# Task config mapping
TASK_CONFIGS = {}

base_configs = [
    {"model": "7b", "replicas": "8R", "trace": "Synth", "arch": "512", "reward": "e2e"},
    {"model": "7b", "replicas": "8R", "trace": "Tool",  "arch": "512", "reward": "e2e"},
    {"model": "7b", "replicas": "8R", "trace": "Conv",  "arch": "512", "reward": "e2e"},
    {"model": "7b", "replicas": "16R", "trace": "Synth", "arch": "512", "reward": "e2e"},
    {"model": "7b", "replicas": "16R", "trace": "Tool",  "arch": "512", "reward": "e2e"},
    {"model": "7b", "replicas": "16R", "trace": "Conv",  "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "8R", "trace": "Synth", "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "8R", "trace": "Tool",  "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "8R", "trace": "Conv",  "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "16R", "trace": "Synth", "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "16R", "trace": "Tool",  "arch": "512", "reward": "e2e"},
    {"model": "70b", "replicas": "16R", "trace": "Conv",  "arch": "512", "reward": "e2e"},
]

for i, cfg in enumerate(base_configs):
    TASK_CONFIGS[i + 1] = cfg.copy()

for i, cfg in enumerate(base_configs):
    cfg_ttft = cfg.copy()
    cfg_ttft['reward'] = 'ttft'
    TASK_CONFIGS[i + 13] = cfg_ttft

for i, cfg in enumerate(base_configs):
    cfg_1024 = cfg.copy()
    cfg_1024['arch'] = '1024'
    TASK_CONFIGS[i + 25] = cfg_1024

for i, cfg in enumerate(base_configs):
    cfg_1024_ttft = cfg.copy()
    cfg_1024_ttft['arch'] = '1024'
    cfg_1024_ttft['reward'] = 'ttft'
    TASK_CONFIGS[i + 37] = cfg_1024_ttft


def parse_single_log(task_id):
    """Fetch and parse a single task log."""
    cmd = f"ssh -o ControlMaster=no alpine 'cat /home/kamil@colostate.edu/slurm_logs/vidur_grad_23669902_{task_id}.out'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    
    if result.returncode != 0:
        return []
    
    # Pattern: "Scale X.X Iteration Results: Mean E2E = Y.YYs"
    pattern = re.compile(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s')
    matches = pattern.findall(result.stdout)
    
    scales = []
    for scale, e2e in matches:
        scales.append((float(scale), float(e2e)))
    
    return scales


def collect_all_data():
    """Collect data from all task logs."""
    data = defaultdict(lambda: defaultdict(list))
    metric_pattern = re.compile(r'Scale ([\d.]+) Iteration Results: Mean E2E = ([\d.]+)s')
    
    for task_id in range(1, 49):
        if task_id not in TASK_CONFIGS:
            continue
            
        cmd = f"ssh -o ControlMaster=no alpine 'cat /home/kamil@colostate.edu/slurm_logs/vidur_grad_23669902_{task_id}.out'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            continue
            
        cfg = TASK_CONFIGS[task_id]
        config_key = f"{cfg['model']}-{cfg['replicas']}-{cfg['trace']}-{cfg['reward']}-{cfg['arch']}"
        
        matches = metric_pattern.findall(result.stdout)
        for scale, e2e in matches:
            data[config_key][float(scale)].append(float(e2e))
    
    return data


def create_comparison_plots(data, output_path):
    """Create comparison visualization."""
    
    # Organize by dimension
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
    
    summary_text = "=== Tension Zone (Scale 3.0) ===\n\n"
    
    for dim in ['model', 'trace', 'reward', 'replicas', 'arch']:
        dim_data = comparisons[dim]
        summary_text += f"[{dim.upper()}]\n"
        for label in sorted(dim_data.keys()):
            if 3.0 in dim_data[label]:
                e2es = dim_data[label][3.0]
                summary_text += f"  {label:>6}: {np.mean(e2es):>8.1f}s ± {np.std(e2es):>6.1f}s\n"
        summary_text += "\n"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "/home/kamil/.openclaw/workspace/graduation_comparison.png"
    
    print("Collecting data from 48 task logs...")
    data = collect_all_data()
    
    if data:
        print(f"Parsed {len(data)} configurations")
        fig = create_comparison_plots(data, output_path)
    else:
        print("No data found")
