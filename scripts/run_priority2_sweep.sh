#!/bin/bash
#
# Priority 2 Sweep - Full Config × Trace Baseline Evaluation
#
# Runs 4 selected schedulers across all configs and traces to validate:
# 1. Tension zone pattern across model sizes and replica counts
# 2. Scheduler ranking consistency
# 3. Best config for RL training
#
# Usage:
#   ./scripts/run_priority2_sweep.sh [--dry-run] [--resume]
#
# Options:
#   --dry-run    Print commands without executing
#   --resume     Skip already completed runs
#
# Output: 224 total runs (168 new after Priority 1)
#   - 4 schedulers × 4 configs × 3 traces × variable scales
#   - Synth: 6 scales (0.75, 1.0, 1.5, 3.0, 6.0, 12.0)
#   - Tool/Conv: 4 scales (1.5, 3.0, 6.0, 12.0)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Selected schedulers from Priority 1 analysis
SCHEDULERS=("STICKY_LOR" "ROUND_ROBIN" "LOR" "RANDOM")

# All configs
MODELS=("7b" "70b")
REPLICAS=("8R" "16R")

# Traces
TRACES=("Synth" "Tool" "Conv")

# Scale ranges per trace (replacing 48/24 with 1.0/0.75)
declare -A SCALE_RANGES=(
    ["Synth"]="0.75 1.0 1.5 3.0 6.0 12.0"
    ["Tool"]="1.5 3.0 6.0 12.0"
    ["Conv"]="1.5 3.0 6.0 12.0"
)

# Parse options
DRY_RUN=false
RESUME=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_if_completed() {
    local scheduler=$1 model=$2 replicas=$3 trace=$4 scale=$5
    
    # Check if output directory exists with completed run
    local output_dir="/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines"
    output_dir="${output_dir}/${scheduler}/${model}-${replicas}-${trace}/scale${scale}"
    
    if [ -d "$output_dir" ]; then
        # Check for timestamped run directory with request_metrics.csv
        if find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name "202*" -exec test -f {}/request_metrics.csv \; -print -quit | grep -q .; then
            return 0  # Completed
        fi
    fi
    return 1  # Not completed
}

run_config() {
    local scheduler=$1 model=$2 replicas=$3 trace=$4 scale=$5
    
    # Check if already completed
    if $RESUME && check_if_completed "$scheduler" "$model" "$replicas" "$trace" "$scale"; then
        log_info "⏭️  SKIP: $scheduler $model $replicas $trace Scale $scale (already completed)"
        return 0
    fi
    
    local cmd="./scripts/run_baseline_sweep.sh $scheduler $model $replicas $trace $scale"
    
    if $DRY_RUN; then
        echo "  $cmd"
    else
        log_info "▶️  RUN: $scheduler $model $replicas $trace Scale $scale"
        bash -c "$cmd"
    fi
}

# ============================================================================
# MAIN SWEEP LOOP
# ============================================================================

log_info "🚀 Starting Priority 2 Sweep"
log_info "Mode: $(if $DRY_RUN; then echo 'DRY RUN'; else echo 'EXECUTE'; fi)"
log_info "Resume: $(if $RESUME; then echo 'YES'; else echo 'NO'; fi)"
log_info ""

total_runs=0
skipped_runs=0

for scheduler in "${SCHEDULERS[@]}"; do
    for model in "${MODELS[@]}"; do
        for replicas in "${REPLICAS[@]}"; do
            for trace in "${TRACES[@]}"; do
                # Get scales for this trace
                scales=(${SCALE_RANGES[$trace]})
                
                log_info ""
                log_info "════════════════════════════════════════"
                log_info "Config: $scheduler / $model / $replicas / $trace"
                log_info "Scales: ${scales[*]}"
                log_info "════════════════════════════════════════"
                
                for scale in "${scales[@]}"; do
                    ((total_runs++))
                    
                    # Skip if completed Priority 1 run (7b-16R only)
                    if [ "$model" = "7b" ] && [ "$replicas" = "16R" ]; then
                        if $RESUME && check_if_completed "$scheduler" "$model" "$replicas" "$trace" "$scale"; then
                            log_info "⏭️  SKIP: Scale $scale (Priority 1 completed)"
                            ((skipped_runs++))
                            continue
                        fi
                    fi
                    
                    run_config "$scheduler" "$model" "$replicas" "$trace" "$scale"
                done
            done
        done
    done
done

log_info ""
log_info "════════════════════════════════════════"
log_info "✅ Priority 2 Sweep Complete"
log_info "Total runs: $total_runs"
if $RESUME; then
    log_info "Skipped: $skipped_runs (already completed)"
    log_info "Executed: $((total_runs - skipped_runs))"
fi
log_info "════════════════════════════════════════"
