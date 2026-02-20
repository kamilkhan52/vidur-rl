#!/usr/bin/env python3
"""
Priority 2 Analysis: Cross-config baseline comparison
Research questions:
  1. Model size impact (7b vs 70b) on scheduler rankings
  2. Replica count impact (8R vs 16R) on best heuristic
  3. Generalization of Priority 1 findings
  4. RANDOM baseline / RL ceiling per config
  5. Tension zone characterization
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTDIR = '/home/kamil/vidur-rl-working/plots/EXP02_P2'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv('/home/kamil/vidur-rl-working/plots/priority2_all.csv')

# Filter to Priority 2 scales only (exclude 24, 48 from Priority 1)
df = df[df['scale'] <= 12.0].copy()

# Sort scales for plotting (high scale = low load, low scale = high load)
df = df.sort_values(['scheduler', 'model', 'replicas', 'trace', 'scale'])

SCHEDULERS = ['STICKY_LOR', 'ROUND_ROBIN', 'LOR', 'RANDOM']
SCHED_COLORS = {'STICKY_LOR': '#2196F3', 'ROUND_ROBIN': '#4CAF50', 'LOR': '#FF9800', 'RANDOM': '#F44336'}
SCHED_MARKERS = {'STICKY_LOR': 'o', 'ROUND_ROBIN': 's', 'LOR': '^', 'RANDOM': 'x'}
TRACES = ['Synth', 'Tool', 'Conv']
CONFIGS = [('7b', '8R'), ('7b', '16R'), ('70b', '8R'), ('70b', '16R')]

print(f"Total rows after filtering: {len(df)}")
print(f"Scales present: {sorted(df['scale'].unique())}")
print()

# ============================================================================
# PLOT 1: Cross-config E2E comparison (one subplot per trace, lines per scheduler)
# Shows 7b vs 70b and 8R vs 16R side by side
# ============================================================================
print("Plot 1: Cross-config E2E comparison...")

for trace in TRACES:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle(f'E2E Latency vs Scale — {trace} Trace\n(4 configs, 4 schedulers)', fontsize=14, fontweight='bold')
    
    for i, (model, rep) in enumerate(CONFIGS):
        ax = axes[i]
        subset = df[(df['trace'] == trace) & (df['model'] == model) & (df['replicas'] == rep)]
        
        for sched in SCHEDULERS:
            sdata = subset[subset['scheduler'] == sched].sort_values('scale')
            if len(sdata) > 0:
                ax.plot(sdata['scale'], sdata['e2e_mean'], 
                       color=SCHED_COLORS[sched], marker=SCHED_MARKERS[sched],
                       label=sched, linewidth=2, markersize=6)
        
        ax.set_title(f'{model}-{rep}', fontweight='bold')
        ax.set_xlabel('Scale (higher load →)')
        if i == 0:
            ax.set_ylabel('Mean E2E Latency (s)')
        ax.invert_xaxis()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/1_e2e_comparison_{trace}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 1_e2e_comparison_{trace}.png")

# ============================================================================
# PLOT 2: Model size comparison (7b vs 70b) — direct overlay
# ============================================================================
print("Plot 2: Model size comparison (7b vs 70b)...")

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Model Size Impact: 7b vs 70b\n(Solid=7b, Dashed=70b)', fontsize=14, fontweight='bold')

for row, trace in enumerate(TRACES):
    for col, sched in enumerate(SCHEDULERS):
        ax = axes[row, col]
        
        for model, rep in CONFIGS:
            subset = df[(df['trace'] == trace) & (df['scheduler'] == sched) & 
                       (df['model'] == model) & (df['replicas'] == rep)]
            subset = subset.sort_values('scale')
            if len(subset) == 0:
                continue
            
            ls = '-' if model == '7b' else '--'
            color = '#2196F3' if rep == '8R' else '#F44336'
            label = f'{model}-{rep}'
            ax.plot(subset['scale'], subset['e2e_mean'], 
                   linestyle=ls, color=color, marker='o', markersize=4,
                   label=label, linewidth=1.5)
        
        ax.set_title(f'{sched} / {trace}', fontsize=9, fontweight='bold')
        ax.invert_xaxis()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Mean E2E (s)')
        if row == 2:
            ax.set_xlabel('Scale')
        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/2_model_size_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 2_model_size_comparison.png")

# ============================================================================
# PLOT 3: Scheduler ranking heatmaps per config
# Shows which scheduler wins at each scale for each config
# ============================================================================
print("Plot 3: Scheduler ranking heatmaps...")

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Best Scheduler at Each Scale\n(darker = lower E2E = better)', fontsize=14, fontweight='bold')

for row, (model, rep) in enumerate(CONFIGS):
    for col, trace in enumerate(TRACES):
        ax = axes[row, col]
        subset = df[(df['trace'] == trace) & (df['model'] == model) & (df['replicas'] == rep)]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(f'{model}-{rep} / {trace}')
            continue
        
        # Pivot: schedulers as rows, scales as columns
        pivot = subset.pivot_table(index='scheduler', columns='scale', values='e2e_mean')
        pivot = pivot.reindex(SCHEDULERS)
        # Sort columns (scales) descending so high load is on right
        pivot = pivot[sorted(pivot.columns, reverse=True)]
        
        # Normalize per column (scale) — highlight best at each scale
        normed = pivot.copy()
        for c in normed.columns:
            col_vals = normed[c]
            min_val = col_vals.min()
            max_val = col_vals.max()
            if max_val > min_val:
                normed[c] = (col_vals - min_val) / (max_val - min_val)
            else:
                normed[c] = 0
        
        im = ax.imshow(normed.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Annotate with actual values
        for ii in range(len(SCHEDULERS)):
            for jj in range(len(pivot.columns)):
                val = pivot.iloc[ii, jj]
                if pd.notna(val):
                    text = f'{val:.1f}' if val < 100 else f'{val:.0f}'
                    color = 'white' if normed.iloc[ii, jj] > 0.6 else 'black'
                    ax.text(jj, ii, text, ha='center', va='center', fontsize=7, color=color)
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(s) for s in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(SCHEDULERS)))
        ax.set_yticklabels(SCHEDULERS, fontsize=8)
        ax.set_title(f'{model}-{rep} / {trace}', fontsize=10, fontweight='bold')
        if row == 3:
            ax.set_xlabel('Scale (← low load | high load →)')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/3_ranking_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 3_ranking_heatmaps.png")

# ============================================================================
# PLOT 4: RANDOM baseline & RL ceiling
# Gap between RANDOM and best heuristic = potential RL improvement
# ============================================================================
print("Plot 4: RANDOM baseline / RL ceiling...")

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('RL Ceiling: Gap Between RANDOM and Best Heuristic\n(Shaded area = potential RL improvement)', 
             fontsize=14, fontweight='bold')

for row, trace in enumerate(TRACES):
    for col, (model, rep) in enumerate(CONFIGS):
        ax = axes[row, col]
        subset = df[(df['trace'] == trace) & (df['model'] == model) & (df['replicas'] == rep)]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        
        # Get RANDOM and best non-RANDOM at each scale
        scales = sorted(subset['scale'].unique(), reverse=True)
        random_vals = []
        best_vals = []
        best_names = []
        
        for s in scales:
            scale_data = subset[subset['scale'] == s]
            random_row = scale_data[scale_data['scheduler'] == 'RANDOM']
            best_row = scale_data[scale_data['scheduler'] != 'RANDOM'].sort_values('e2e_mean').iloc[0] if len(scale_data[scale_data['scheduler'] != 'RANDOM']) > 0 else None
            
            if len(random_row) > 0 and best_row is not None:
                random_vals.append(random_row['e2e_mean'].iloc[0])
                best_vals.append(best_row['e2e_mean'])
                best_names.append(best_row['scheduler'])
            else:
                random_vals.append(np.nan)
                best_vals.append(np.nan)
                best_names.append('')
        
        ax.plot(scales, random_vals, color='#F44336', marker='x', linewidth=2, label='RANDOM', markersize=8)
        ax.plot(scales, best_vals, color='#2196F3', marker='o', linewidth=2, label='Best Heuristic', markersize=6)
        ax.fill_between(scales, best_vals, random_vals, alpha=0.2, color='#4CAF50', label='RL Ceiling')
        
        # Annotate best scheduler name at each scale
        for j, (s, name) in enumerate(zip(scales, best_names)):
            if name and not np.isnan(best_vals[j]):
                short = name.replace('STICKY_', 'S_').replace('ROUND_ROBIN', 'RR')
                ax.annotate(short, (s, best_vals[j]), fontsize=5, ha='center', va='top',
                           xytext=(0, -8), textcoords='offset points')
        
        ax.set_title(f'{model}-{rep} / {trace}', fontsize=10, fontweight='bold')
        ax.invert_xaxis()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Mean E2E (s)')
        if row == 2:
            ax.set_xlabel('Scale')
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/4_rl_ceiling.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 4_rl_ceiling.png")

# ============================================================================
# PLOT 5: Tension zone — performance spread across schedulers
# ============================================================================
print("Plot 5: Tension zone characterization...")

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Tension Zone: Scheduler Performance Spread\n(Higher spread = more opportunity for RL)', 
             fontsize=14, fontweight='bold')

for row, trace in enumerate(TRACES):
    for col, (model, rep) in enumerate(CONFIGS):
        ax = axes[row, col]
        subset = df[(df['trace'] == trace) & (df['model'] == model) & (df['replicas'] == rep)]
        
        if len(subset) == 0:
            continue
        
        scales = sorted(subset['scale'].unique(), reverse=True)
        spreads = []
        mins = []
        maxs = []
        
        for s in scales:
            scale_data = subset[subset['scale'] == s]
            vals = scale_data['e2e_mean'].values
            spreads.append(max(vals) / min(vals) if min(vals) > 0 else 0)
            mins.append(min(vals))
            maxs.append(max(vals))
        
        # Bar chart of spread ratio
        bars = ax.bar(range(len(scales)), spreads, color=['#F44336' if sp > 5 else '#FF9800' if sp > 2 else '#4CAF50' for sp in spreads])
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([str(s) for s in scales], fontsize=8)
        ax.set_title(f'{model}-{rep} / {trace}', fontsize=10, fontweight='bold')
        ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='2x spread')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5x spread')
        
        # Annotate with actual min-max
        for j, (sp, mn, mx) in enumerate(zip(spreads, mins, maxs)):
            if sp > 1:
                ax.text(j, sp + 0.1, f'{mn:.1f}-{mx:.0f}s' if mx > 10 else f'{mn:.1f}-{mx:.1f}s', 
                       ha='center', fontsize=5, rotation=45)
        
        if col == 0:
            ax.set_ylabel('Worst/Best Ratio')
        if row == 2:
            ax.set_xlabel('Scale (← low load | high load →)')
        if row == 0 and col == 0:
            ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f'{OUTDIR}/5_tension_zone.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 5_tension_zone.png")

# ============================================================================
# PLOT 6: Replica count comparison (8R vs 16R) — direct overlay
# ============================================================================
print("Plot 6: Replica count comparison (8R vs 16R)...")

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Replica Count Impact: 8R vs 16R\n(Solid=8R, Dashed=16R)', fontsize=14, fontweight='bold')

for row, trace in enumerate(TRACES):
    for col, sched in enumerate(SCHEDULERS):
        ax = axes[row, col]
        
        for model in ['7b', '70b']:
            for rep in ['8R', '16R']:
                subset = df[(df['trace'] == trace) & (df['scheduler'] == sched) & 
                           (df['model'] == model) & (df['replicas'] == rep)]
                subset = subset.sort_values('scale')
                if len(subset) == 0:
                    continue
                
                ls = '-' if rep == '8R' else '--'
                color = '#2196F3' if model == '7b' else '#F44336'
                ax.plot(subset['scale'], subset['e2e_mean'],
                       linestyle=ls, color=color, marker='o', markersize=4,
                       label=f'{model}-{rep}', linewidth=1.5)
        
        ax.set_title(f'{sched} / {trace}', fontsize=9, fontweight='bold')
        ax.invert_xaxis()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Mean E2E (s)')
        if row == 2:
            ax.set_xlabel('Scale')
        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/6_replica_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 6_replica_comparison.png")

# ============================================================================
# SUMMARY TABLE: Best scheduler per config per scale
# ============================================================================
print("\n=== SUMMARY: Best scheduler at each scale ===\n")

for trace in TRACES:
    print(f"--- {trace} ---")
    for model, rep in CONFIGS:
        subset = df[(df['trace'] == trace) & (df['model'] == model) & (df['replicas'] == rep)]
        if len(subset) == 0:
            continue
        scales = sorted(subset['scale'].unique())
        winners = []
        for s in scales:
            scale_data = subset[subset['scale'] == s]
            best = scale_data.sort_values('e2e_mean').iloc[0]
            winners.append(f"  Scale {s}: {best['scheduler']} ({best['e2e_mean']:.2f}s)")
        print(f"  {model}-{rep}:")
        for w in winners:
            print(f"    {w}")
    print()

print(f"\nAll plots saved to {OUTDIR}/")
