#!/usr/bin/env python3
"""
RL vs Baseline Comparison: Compare EXP01 RL results against Priority 2 heuristic baselines
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = '/home/kamil/vidur-rl-working/plots/EXP02_RL_vs_Baseline'
os.makedirs(OUTDIR, exist_ok=True)

# Load RL data (EXP01)
rl_df = pd.read_csv('/home/kamil/vidur-rl-working/graduation_summary.csv')
print(f"RL data: {len(rl_df)} rows")

# Load baseline data (Priority 2)
baseline_df = pd.read_csv('/home/kamil/vidur-rl-working/plots/priority2_all.csv')
print(f"Baseline data: {len(baseline_df)} rows")

# Get the BEST RL result per (model, replicas, trace, scale) - lowest E2E
rl_best = rl_df.sort_values(['model', 'replicas', 'trace', 'scale', 'mean_e2e']).groupby(
    ['model', 'replicas', 'trace', 'scale']).first().reset_index()

# Filter to VALID data: exclude Scale 1.5 (buggy runs) and very high E2E (>100s = broken)
rl_filtered = rl_best[(rl_best['scale'] >= 3.0) & (rl_best['mean_e2e'] < 100)].copy()
print(f"RL filtered to valid data (scale>=3.0, e2e<100s): {len(rl_filtered)} rows")

# For comparison, we need matching configs between RL and baseline
# RL has: 7b (Synth graduated), 70b (stuck at 48)
# Baseline has: 7b/70b, 8R/16R, Synth/Tool/Conv, scales 1.5-12.0

# Filter RL to scales that exist in baseline
baseline_scales = baseline_df['scale'].unique()
rl_filtered = rl_best[rl_best['scale'].isin(baseline_scales)].copy()
print(f"RL filtered to baseline scales: {len(rl_filtered)} rows")

# Merge RL with baselines
# Round RL scale to match baseline (baseline uses integer-ish scales)
rl_filtered['scale_round'] = rl_filtered['scale'].round(1)

# Get baseline best (non-RANDOM) for each config/scale
baseline_best = baseline_df[baseline_df['scheduler'] != 'RANDOM'].sort_values(
    ['model', 'replicas', 'trace', 'scale', 'e2e_mean']).groupby(
    ['model', 'replicas', 'trace', 'scale']).first().reset_index()
baseline_best = baseline_best.rename(columns={'e2e_mean': 'baseline_best', 'scheduler': 'best_scheduler'})

# Get RANDOM baseline
baseline_random = baseline_df[baseline_df['scheduler'] == 'RANDOM'][['model', 'replicas', 'trace', 'scale', 'e2e_mean']].copy()
baseline_random = baseline_random.rename(columns={'e2e_mean': 'random_baseline'})

# Merge
comparison = rl_filtered.merge(
    baseline_best[['model', 'replicas', 'trace', 'scale', 'baseline_best', 'best_scheduler']], 
    left_on=['model', 'replicas', 'trace', 'scale_round'],
    right_on=['model', 'replicas', 'trace', 'scale'],
    how='inner'
).merge(
    baseline_random,
    left_on=['model', 'replicas', 'trace', 'scale_round'],
    right_on=['model', 'replicas', 'trace', 'scale'],
    how='left',
    suffixes=('', '_rand')
)

# CRITICAL: Filter out Scale 1.5 (buggy data) and any rows with E2E > 100
comparison = comparison[(comparison['scale'] >= 3.0) & (comparison['mean_e2e'] < 100)]

# Calculate improvements
comparison['rl_vs_best'] = comparison['baseline_best'] - comparison['mean_e2e']  # positive = RL better
comparison['rl_vs_random'] = comparison['random_baseline'] - comparison['mean_e2e']  # positive = RL better

print(f"\n=== COMPARISON: RL vs Best Heuristic ===")
print(comparison[['model', 'replicas', 'trace', 'scale', 'mean_e2e', 'baseline_best', 'best_scheduler', 'rl_vs_best']].to_string())

print(f"\n=== COMPARISON: RL vs RANDOM ===")
print(comparison[['model', 'replicas', 'trace', 'scale', 'mean_e2e', 'random_baseline', 'rl_vs_random']].to_string())

# ============================================================================
# PLOT 1: Bar chart comparison at each config/scale
# ============================================================================
print("\nPlot 1: RL vs Baseline comparison...")

# Get unique configs
configs = comparison[['model', 'replicas', 'trace']].drop_duplicates().values.tolist()

fig, axes = plt.subplots(len(configs), 1, figsize=(14, 4*len(configs)), squeeze=False)
fig.suptitle('RL vs Heuristic Baselines\n(RL = trained agent, Best = best heuristic, RANDOM = unlearned)', 
             fontsize=14, fontweight='bold')

for idx, (model, replicas, trace) in enumerate(configs):
    ax = axes[idx, 0]
    subset = comparison[(comparison['model'] == model) & (comparison['replicas'] == replicas) & (comparison['trace'] == trace)]
    
    if len(subset) == 0:
        ax.text(0.5, 0.5, 'No matching data', transform=ax.transAxes, ha='center')
        ax.set_title(f'{model}-{replicas} / {trace}')
        continue
    
    subset = subset.sort_values('scale', ascending=False)
    
    x = np.arange(len(subset))
    width = 0.25
    
    bars1 = ax.bar(x - width, subset['mean_e2e'], width, label='RL Agent', color='#2196F3')
    bars2 = ax.bar(x, subset['baseline_best'], width, label='Best Heuristic', color='#4CAF50')
    bars3 = ax.bar(x + width, subset['random_baseline'], width, label='RANDOM', color='#F44336')
    
    ax.set_xlabel('Scale (higher load →)')
    ax.set_ylabel('Mean E2E (s)')
    ax.set_title(f'{model}-{replicas} / {trace}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.1f}' for s in subset['scale']])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate with scheduler names
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.annotate(row['best_scheduler'].replace('STICKY_', 'S_').replace('ROUND_ROBIN', 'RR'), 
                   (i, row['baseline_best']), fontsize=6, ha='center', va='bottom', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/1_rl_vs_baseline_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 1_rl_vs_baseline_bars.png")

# ============================================================================
# PLOT 2: Improvement (RL vs Best Heuristic) as percentage
# ============================================================================
print("Plot 2: RL improvement over baselines...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RL Improvement Over Baselines\n(positive = RL is faster)', fontsize=14, fontweight='bold')

# Left: RL vs Best Heuristic
ax = axes[0]
comparison['pct_improvement'] = (comparison['rl_vs_best'] / comparison['baseline_best']) * 100
colors = ['#4CAF50' if x > 0 else '#F44336' for x in comparison['pct_improvement']]
ax.bar(range(len(comparison)), comparison['pct_improvement'], color=colors)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Config/Scale')
ax.set_ylabel('% Improvement (negative = RL worse)')
ax.set_title('RL vs Best Heuristic')

labels = [f"{r['model']}-{r['replicas']}/{r['trace'][:3]}/{r['scale']:.1f}" for _, r in comparison.iterrows()]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Right: RL vs RANDOM
ax = axes[1]
comparison['pct_vs_random'] = (comparison['rl_vs_random'] / comparison['random_baseline']) * 100
colors = ['#4CAF50' if x > 0 else '#F44336' for x in comparison['pct_vs_random']]
ax.bar(range(len(comparison)), comparison['pct_vs_random'], color=colors)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Config/Scale')
ax.set_ylabel('% Improvement')
ax.set_title('RL vs RANDOM (untrained)')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/2_rl_improvement.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 2_rl_improvement.png")

# ============================================================================
# PLOT 3: Direct comparison at Scale 1.5 (tension zone)
# ============================================================================
print("Plot 3: Scale 1.5 deep dive...")

s15 = comparison[comparison['scale'] == 1.5]
if len(s15) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(s15))
    width = 0.25
    
    bars1 = ax.bar(x - width, s15['mean_e2e'], width, label='RL Agent', color='#2196F3')
    bars2 = ax.bar(x, s15['baseline_best'], width, label='Best Heuristic', color='#4CAF50')
    bars3 = ax.bar(x + width, s15['random_baseline'], width, label='RANDOM', color='#F44336')
    
    ax.set_xlabel('Config')
    ax.set_ylabel('Mean E2E (s)')
    ax.set_title('RL vs Baselines at Scale 1.5 (High Load)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    labels = [f"{r['model']}-{r['replicas']}/{r['trace'][:3]}" for _, r in s15.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/3_scale_1.5_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved 3_scale_1.5_comparison.png")
else:
    print("  No Scale 1.5 data in comparison")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n=== SUMMARY ===")
if len(comparison) > 0:
    wins = (comparison['rl_vs_best'] > 0).sum()
    total = len(comparison)
    print(f"RL beats best heuristic: {wins}/{total} ({100*wins/total:.1f}%)")
    
    avg_improvement = comparison['pct_improvement'].mean()
    print(f"Average improvement: {avg_improvement:.1f}%")
    
    beats_random = (comparison['rl_vs_random'] > 0).sum()
    print(f"RL beats RANDOM: {beats_random}/{total} ({100*beats_random/total:.1f}%)")
    
    # Best and worst
    best_idx = comparison['pct_improvement'].idxmax()
    worst_idx = comparison['pct_improvement'].idxmin()
    print(f"\nBest RL improvement: {comparison.loc[best_idx, 'model']}-{comparison.loc[best_idx, 'replicas']} {comparison.loc[best_idx, 'trace']} @ {comparison.loc[best_idx, 'scale']} = {comparison.loc[best_idx, 'pct_improvement']:.1f}%")
    print(f"Worst RL performance: {comparison.loc[worst_idx, 'model']}-{comparison.loc[worst_idx, 'replicas']} {comparison.loc[worst_idx, 'trace']} @ {comparison.loc[worst_idx, 'scale']} = {comparison.loc[worst_idx, 'pct_improvement']:.1f}%")
else:
    print("No matching data between RL and baseline experiments")

print(f"\nAll plots saved to {OUTDIR}/")
