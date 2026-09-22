#!/usr/bin/env python3
"""
Generate comprehensive baseline sweep table showing ALL configurations.
This creates a visual table for Telegram (since Markdown tables are hard to read).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np

# Configuration matrix
SCHEDULERS = [
    'RANDOM', 'ROUND_ROBIN', 'LOR', 'LOP', 'LOP_BINARY',
    'LOP_BATCH_END', 'LOP_UNCACHED', 'STICKY_RR', 'STICKY_LOR',
    'RANKED_STICKY_LOP', 'TOLERANT_STICKY'
]

CONFIGS = [
    ('7b', '8R', 'Synth'),
    ('7b', '8R', 'Tool'),
    ('7b', '8R', 'Conv'),
    ('7b', '16R', 'Synth'),
    ('7b', '16R', 'Tool'),
    ('7b', '16R', 'Conv'),
    ('70b', '8R', 'Synth'),
    ('70b', '8R', 'Tool'),
    ('70b', '8R', 'Conv'),
    ('70b', '16R', 'Synth'),
    ('70b', '16R', 'Tool'),
    ('70b', '16R', 'Conv'),
]

SCALES = [48.0, 24.0, 12.0, 6.0, 3.0, 1.5]

# Config label mapping
CONFIG_LABELS = [
    '7b-8R-Synth', '7b-8R-Tool', '7b-8R-Conv',
    '7b-16R-Synth', '7b-16R-Tool', '7b-16R-Conv',
    '70b-8R-Synth', '70b-8R-Tool', '70b-8R-Conv',
    '70b-16R-Synth', '70b-16R-Tool', '70b-16R-Conv',
]

def create_comprehensive_table():
    """Create a comprehensive table showing all baseline combinations."""
    
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.axis('off')
    
    # Title
    fig.suptitle('Comprehensive Baseline Sweep Matrix\nAll Schedulers × All Configs × All Scales',
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create table data - 12 configs × 6 scales = 72 cells, each showing 11 schedulers
    # We'll create sub-cells
    
    cell_height = 0.12
    cell_width = 0.08
    start_y = 0.92
    
    # Column headers (Schedulers) - repeated for each config group
    scheduler_headers = SCHEDULERS
    
    # Color mapping for scheduler complexity
    scheduler_colors = {
        'RANDOM': '#E8F5E9',          # Light green
        'ROUND_ROBIN': '#C8E6C9',     # Green
        'LOR': '#BBDEFB',             # Light blue
        'LOP': '#64B5F6',             # Blue
        'LOP_BINARY': '#42A5F5',      # Blue
        'LOP_BATCH_END': '#1E88E5',   # Darker blue
        'LOP_UNCACHED': '#90CAF9',    # Light blue
        'STICKY_RR': '#FFF9C4',      # Light yellow
        'STICKY_LOR': '#FFE082',      # Yellow
        'RANKED_STICKY_LOP': '#FFB74D',   # Orange
        'TOLERANT_STICKY': '#FF8A65',     # Orange-red
    }
    
    # Config label mapping
    config_labels = CONFIG_LABELS
    
    # Create the main table structure
    # Row per config × scale, columns per scheduler
    
    n_configs = len(CONFIGS)
    n_scales = len(SCALES)
    n_schedulers = len(SCHEDULERS)
    
    # Table title
    ax.text(0.5, 0.99, f'Total Combinations: {n_configs} configs × {n_scales} scales × {n_schedulers} schedulers = {n_configs * n_scales * n_schedulers} runs',
            ha='center', va='top', fontsize=12, transform=ax.transAxes)
    
    # Draw scheduler headers
    header_y = 0.95
    col_width = 0.09
    start_x = 0.02
    
    # Config label column
    ax.text(start_x - 0.01, header_y, 'Config', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', rotation=45)
    ax.text(start_x - 0.01, header_y - 0.02, 'Scale', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', rotation=45)
    
    for idx, sched in enumerate(SCHEDULERS):
        x = start_x + idx * col_width + col_width/2
        ax.text(x, header_y, sched, ha='center', va='bottom', 
                fontsize=8, fontweight='bold', rotation=90,
                color=scheduler_colors.get(sched, 'black'))
    
    # Draw config × scale rows
    row_height = 0.13
    current_y = header_y - 0.06
    
    for config_idx, config_label in enumerate(config_labels):
        model, replicas, trace = CONFIGS[config_idx]
        
        # Config label
        ax.text(start_x - 0.01, current_y + row_height/2, config_label,
                ha='center', va='center', fontsize=9, fontweight='bold',
                rotation=0, transform=ax.transAxes)
        
        for scale_idx, scale in enumerate(SCALES):
            x = start_x
            y = current_y - scale_idx * (row_height / n_scales)
            
            # Scale label (only for first config column)
            ax.text(x - 0.01, y + (row_height/n_scales)/2, f'{scale:.1f}',
                    ha='right', va='center', fontsize=7, transform=ax.transAxes)
            
            # Draw scheduler cells for this config × scale
            for sched_idx, sched in enumerate(SCHEDULERS):
                cell_x = start_x + sched_idx * col_width
                
                # Placeholder cell (will be filled with actual data when run)
                rect = plt.Rectangle(
                    (cell_x, y), col_width * 0.9, row_height / n_scales * 0.9,
                    facecolor='white', edgecolor='gray', linewidth=0.5
                )
                ax.add_patch(rect)
                
                # Add placeholder text
                ax.text(cell_x + col_width*0.45, y + row_height/(n_scales*2), 
                        '—', ha='center', va='center', fontsize=6, color='lightgray')
        
        current_y -= row_height * 1.1
    
    # Legend
    legend_y = 0.02
    ax.text(0.02, legend_y + 0.03, 'Scheduler Complexity:', fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    complexity_labels = [
        ('Simple (RANDOM, RR)', '#E8F5E9'),
        ('Lookup (LOR, LOP, etc.)', '#BBDEFB'),
        ('Sticky (STICKY_*)', '#FFF9C4'),
    ]
    
    for idx, (label, color) in enumerate(complexity_labels):
        x = 0.25 + idx * 0.25
        rect = plt.Rectangle((x, legend_y), 0.02, 0.02, facecolor=color, edgecolor='gray')
        ax.add_patch(rect)
        ax.text(x + 0.025, legend_y + 0.01, label, fontsize=9, va='center', transform=ax.transAxes)
    
    # Instructions
    ax.text(0.5, -0.02, 
            'To run: ./scripts/run_baseline_sweep.sh <scheduler> <config> <scale> | Results will fill these cells',
            ha='center', va='bottom', fontsize=10, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def create_priority_table():
    """Create a prioritization table showing run counts."""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    fig.suptitle('Baseline Sweep: Priority Levels and Run Counts', fontsize=16, fontweight='bold')
    
    # Priority levels
    priorities = [
        {
            'name': 'Priority 1: Full Coverage (Essential)',
            'description': 'All schedulers × 7b-16R-Synth × All Scales',
            'configs': ['7b-16R-Synth'],
            'scales': [48.0, 24.0, 12.0, 6.0, 3.0, 1.5],
            'runs': 11 * 1 * 6,
            'color': '#4CAF50'
        },
        {
            'name': 'Priority 2: Model Comparison',
            'description': 'Top 3 schedulers × (7b vs 70b)-8R-Synth × Scales 48, 24, 12',
            'configs': ['7b-8R-Synth', '70b-8R-Synth'],
            'scales': [48.0, 24.0, 12.0],
            'runs': 3 * 2 * 3,
            'color': '#2196F3'
        },
        {
            'name': 'Priority 3: Trace Difficulty',
            'description': 'Top 3 schedulers × All traces × 7b-16R × Scale 48',
            'configs': ['7b-16R-Synth', '7b-16R-Tool', '7b-16R-Conv'],
            'scales': [48.0],
            'runs': 3 * 3 * 1,
            'color': '#FF9800'
        },
        {
            'name': 'Priority 4: Tension Zone Deep Dive',
            'description': 'All schedulers × 7b-8R-Synth × Scale 3.0',
            'configs': ['7b-8R-Synth'],
            'scales': [3.0],
            'runs': 11 * 1 * 1,
            'color': '#F44336'
        },
        {
            'name': 'Priority 5: Full Matrix (If Time Permits)',
            'description': 'All schedulers × All configs × All scales',
            'configs': CONFIG_LABELS,
            'scales': SCALES,
            'runs': 11 * 12 * 6,
            'color': '#9C27B0'
        },
    ]
    
    y = 0.92
    for pri in priorities:
        # Priority header
        rect = plt.Rectangle((0.02, y - 0.08), 0.96, 0.08, 
                            facecolor=pri['color'], edgecolor='black', alpha=0.2)
        ax.add_patch(rect)
        ax.text(0.03, y, pri['name'], fontsize=12, fontweight='bold', 
                va='center', transform=ax.transAxes)
        ax.text(0.85, y, f'{pri["runs"]} runs', fontsize=11, fontweight='bold',
                va='center', transform=ax.transAxes)
        
        # Description
        ax.text(0.03, y - 0.04, pri['description'], fontsize=9, va='center',
                transform=ax.transAxes)
        
        y -= 0.12
    
    # Total runs
    total_runs = sum(p['runs'] for p in priorities)
    ax.text(0.5, 0.05, f'Cumulative Total: {total_runs} runs', 
            fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
    
    return fig

def create_scheduler_reference():
    """Create a reference table for scheduler descriptions."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    fig.suptitle('Scheduler Reference Guide', fontsize=16, fontweight='bold')
    
    scheduler_info = [
        ('RANDOM', 'Simple', 'Random replica selection', 'Baseline for comparison'),
        ('ROUND_ROBIN', 'Simple', 'Cyclic replica selection', 'Simple load balancing'),
        ('LOR', 'Lookup', 'Least Outstanding Requests', 'Prioritizes less busy replicas'),
        ('LOP', 'Lookup', 'Least Outstanding Processing', 'Estimates request time'),
        ('LOP_BINARY', 'Lookup', 'LOP with binary decisions', 'Simplified LOP'),
        ('LOP_BATCH_END', 'Lookup', 'LOP at batch boundaries', 'Batch-aware LOP'),
        ('LOP_UNCACHED', 'Lookup', 'LOP without caching', 'Non-cache-aware variant'),
        ('STICKY_RR', 'Sticky', 'RR with request affinity', 'Keeps requests on same replica'),
        ('STICKY_LOR', 'Sticky', 'LOR with request affinity', 'Combines LOR + stickiness'),
        ('RANKED_STICKY_LOP', 'Sticky', 'Ranked sticky LOP', 'Priority-aware sticky'),
        ('TOLERANT_STICKY', 'Sticky', 'Tolerant sticky variant', 'Stability-focused'),
    ]
    
    # Headers
    y = 0.92
    headers = ['Scheduler', 'Category', 'Description', 'Use Case']
    x_positions = [0.05, 0.28, 0.48, 0.78]
    
    for x, header in zip(x_positions, headers):
        ax.text(x, y, header, fontsize=11, fontweight='bold', va='center')
    
    # Data rows
    y -= 0.06
    colors = ['#E8F5E9', '#BBDEFB', '#FFF9C4']  # Simple, Lookup, Sticky
    
    for idx, (sched, category, desc, use_case) in enumerate(scheduler_info):
        color = '#E8F5E9'  # Simple
        if category == 'Lookup':
            color = '#BBDEFB'
        elif category == 'Sticky':
            color = '#FFF9C4'
        
        rect = plt.Rectangle((0.02, y - 0.04), 0.96, 0.05, facecolor=color, edgecolor='lightgray')
        ax.add_patch(rect)
        
        ax.text(0.05, y, sched, fontsize=10, fontweight='bold', va='center')
        ax.text(0.28, y, category, fontsize=9, va='center')
        ax.text(0.48, y, desc, fontsize=8, va='center')
        ax.text(0.78, y, use_case, fontsize=8, va='center')
        
        y -= 0.06
    
    # Summary
    ax.text(0.5, 0.03, 
            'Key Question: Can RL beat the best heuristic at each scale?',
            fontsize=11, style='italic', ha='center', transform=ax.transAxes)
    
    return fig

if __name__ == "__main__":
    # Generate all tables
    print("Generating comprehensive baseline tables...")
    
    # Table 1: Full matrix
    fig1 = create_comprehensive_table()
    fig1.savefig('/home/kamil/.openclaw/workspace/baseline_sweep_matrix.png', 
                 dpi=100, bbox_inches='tight', facecolor='white')
    print("Saved: baseline_sweep_matrix.png")
    
    # Table 2: Priority levels
    fig2 = create_priority_table()
    fig2.savefig('/home/kamil/.openclaw/workspace/baseline_priority_levels.png',
                 dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: baseline_priority_levels.png")
    
    # Table 3: Scheduler reference
    fig3 = create_scheduler_reference()
    fig3.savefig('/home/kamil/.openclaw/workspace/baseline_scheduler_reference.png',
                 dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: baseline_scheduler_reference.png")
    
    print("\nDone! Tables generated:")
    print("  1. baseline_sweep_matrix.png - Full coverage matrix (placeholder cells)")
    print("  2. baseline_priority_levels.png - Priority levels with run counts")
    print("  3. baseline_scheduler_reference.png - Scheduler descriptions")
