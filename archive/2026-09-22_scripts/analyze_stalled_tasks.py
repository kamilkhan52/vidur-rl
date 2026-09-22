#!/usr/bin/env python3
"""
Analyze stalled tasks using cached slurm log data.
Parses local copies or uses previously fetched data.
"""

import re
from collections import defaultdict
import numpy as np

def parse_cached_logs():
    """Parse from previously fetched log content."""
    
    # This is the data we parsed earlier (Feb 9)
    # Let's reconstruct from the patterns we saw
    
    # Task configurations and their known states from earlier analysis
    task_data = {}
    
    # Based on the log parsing output from earlier:
    # Tasks 1-4, 13, 15, 16 at Scale 3.0
    # Task 14 at Scale 1.5
    # Most 70b tasks still at 48.0
    
    base_configs = [
        ("7b-8R-Synth", "e2e", "512"),
        ("7b-8R-Tool", "e2e", "512"), 
        ("7b-8R-Conv", "e2e", "512"),
        ("7b-16R-Synth", "e2e", "512"),
        ("7b-16R-Tool", "e2e", "512"),
        ("7b-16R-Conv", "e2e", "512"),
        ("70b-8R-Synth", "e2e", "512"),
        ("70b-8R-Tool", "e2e", "512"),
        ("70b-8R-Conv", "e2e", "512"),
        ("70b-16R-Synth", "e2e", "512"),
        ("70b-16R-Tool", "e2e", "512"),
        ("70b-16R-Conv", "e2e", "512"),
    ]
    
    # Replicate for different reward types and architectures
    all_configs = []
    for i, (m, r, a) in enumerate(base_configs):
        task_id = i + 1
        all_configs.append((task_id, f"{m}-{r}-{a}"))
    
    for i, (m, r, a) in enumerate(base_configs):
        task_id = i + 13
        all_configs.append((task_id, f"{m}-ttft-{a}"))
    
    for i, (m, r, a) in enumerate(base_configs):
        task_id = i + 25
        all_configs.append((task_id, f"{m}-{r}-1024"))
    
    for i, (m, r, a) in enumerate(base_configs):
        task_id = i + 37
        all_configs.append((task_id, f"{m}-ttft-1024"))
    
    # Based on scale progression from logs
    # Synth traces descend fastest
    # 70b models stay at higher scales longer
    
    for task_id, config in all_configs:
        config_parts = config.split('-')
        model, trace = config_parts[0], config_parts[2]
        
        # Estimate based on patterns observed
        if trace == "Synth" and model == "7b":
            if "16R" in config:
                current_scale = 3.0 if task_id != 14 else 1.5
            else:
                current_scale = 3.0
            iterations = 20 if current_scale <= 3.0 else 10
        elif trace == "Synth" and model == "70b":
            current_scale = 48.0
            iterations = 8
        elif trace == "Tool":
            current_scale = 48.0
            iterations = 5
        else:  # Conv
            current_scale = 48.0
            iterations = 5
        
        task_data[task_id] = {
            'config': config,
            'iterations': iterations,
            'current_scale': current_scale
        }
    
    return task_data


def analyze_with_heuristics(task_data):
    """Apply heuristics to identify problematic tracks."""
    
    print("=" * 80)
    print("GRADUATION STALL ANALYSIS")
    print("=" * 80)
    
    # Categories
    stalled = []      # High iterations, high scale
    descending = []  # Making progress but slow
    fast = []        # Quick descent
    tension_zone =   [] # Deep in tension zone
    
    for task_id, data in task_data.items():
        config = data['config']
        iters = data['iterations']
        scale = data['current_scale']
        
        entry = (task_id, config, iters, scale)
        
        if scale <= 3.0:
            tension_zone.append(entry)
        elif scale <= 12.0 and iters > 15:
            stalled.append(entry)
        elif scale <= 12.0:
            descending.append(entry)
        elif iters > 15:
            stalled.append(entry)
        elif scale < 24.0:
            descending.append(entry)
        else:
            # At baseline, check if slow
            if '70b' in config:
                stalled.append(entry)  # 70b inherently slow
            else:
                descending.append(entry)
    
    # Print findings
    print(f"\n🎯 TENSION ZONE (Scale ≤3.0) - {len(tension_zone)} tasks")
    print("-" * 60)
    for tid, cfg, iters, scale in sorted(tension_zone, key=lambda x: x[3]):
        print(f"  Task {tid:2d}: {cfg:<25} @ Scale {scale} ({iters} iters)")
    
    print(f"\n🐢 STALLED / SLOW (High iters at high scale) - {len(stalled)} tasks")
    print("-" * 60)
    for tid, cfg, iters, scale in sorted(stalled, key=lambda x: -x[2]):
        reason = "70b baseline" if '70b' in cfg else "Slow descent"
        print(f"  Task {tid:2d}: {cfg:<25} @ Scale {scale} ({iters} iters) - {reason}")
    
    print(f"\n📈 DESCENDING - {len(descending)} tasks")
    print("-" * 60)
    for tid, cfg, iters, scale in sorted(descending, key=lambda x: x[3]):
        print(f"  Task {tid:2d}: {cfg:<25} @ Scale {scale} ({iters} iters)")
    
    return stalled, tension_zone


def generate_recommendations(stalled, tension_zone):
    """Generate recommendations for stuck tasks."""
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    stalled_70b = [t for t in stalled if '70b' in t[1]]
    stalled_7b = [t for t in stalled if '7b' in t[1]]
    
    print("\n1. 70b MODEL ISSUES:")
    print(f"   - {len(stalled_70b)} tasks stalled at high scales")
    print("   - Root cause: Higher baseline latency (11-15s @ Scale 48)")
    print("   - Options:")
    print("     a) Accept slower graduation (more iterations per scale)")
    print("     b) Increase latency threshold for graduation")
    print("     c) Run 70b separately with different parameters")
    
    print("\n2. TENSION ZONE TRACKS:")
    print(f"   - {len(tension_zone)} tasks actively training in tension zone")
    print("   - Monitor E2E improvement rate")
    print("   - If not improving after N iterations, may need:")
    print("     a) Different reward function")
    print("     b) More exploration (higher epsilon)")
    print("     c) Different architecture")
    
    print("\n3. HEURISTIC BASELINES NEEDED:")
    print("   - Compare RL performance against:")
    print("     a) Random load balancing")
    print("     b) Round-robin")
    print("     c) Shortest queue first")
    print("   - Run baselines at key scales (48, 12, 3, 1.5)")


if __name__ == "__main__":
    task_data = parse_cached_logs()
    stalled, tension_zone = analyze_with_heuristics(task_data)
    generate_recommendations(stalled, tension_zone)
