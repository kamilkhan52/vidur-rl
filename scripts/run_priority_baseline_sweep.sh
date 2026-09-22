#!/bin/bash
#
# Priority Baseline Sweep Runner
#
# Runs baseline sweeps by priority level.
# Handles SLURM array job submission or direct execution.
#
# Usage:
#   ./scripts/run_priority_baseline_sweep.sh <priority> [--dry-run]
#
# Priorities:
#   1   Essential: All schedulers × 7b-16R-Synth × All Scales (66 runs)
#   2   Model Comparison: Top 3 schedulers × 7b/70b × Scales 48, 24, 12 (18 runs)
#   3   Trace Difficulty: Top 3 schedulers × All traces × Scale 48 (9 runs)
#   4   Tension Zone: All schedulers × 7b-8R-Synth × Scale 3.0 (11 runs)
#   all  All priorities 1-4 (104 runs)
#   full Full matrix: All schedulers × All configs × All scales (792 runs)
#
# Options:
#   --dry-run    Show commands without executing
#   --test       Run single test config (LOP × 7b × 16R × Synth × 48.0)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT="/projects/kamil@colostate.edu/vidur-rl"
SCRATCH_ROOT="/scratch/alpine/kamil@colostate.edu"

# Priority configurations
declare -A PRIORITY_CONFIGURATIONS=(
    ["1"]="SCHEDULERS=all;MODELS=7b;REPLICAS=16R;TRACES=Synth;SCALES=48.0,24.0,12.0,6.0,3.0,1.5;NOTES='Essential: All schedulers × 7b-16R-Synth × All Scales'"
    ["2"]="SCHEDULERS=top3;MODELS=7b,70b;REPLICAS=8R;TRACES=Synth;SCALES=48.0,24.0,12.0;NOTES='Model Comparison: Top 3 schedulers × 7b/70b × Scales 48-12'"
    ["3"]="SCHEDULERS=top3;MODELS=7b;REPLICAS=16R;TRACES=Synth,Tool,Conv;SCALES=48.0;NOTES='Trace Difficulty: Top 3 schedulers × All traces × Scale 48'"
    ["4"]="SCHEDULERS=all;MODELS=7b;REPLICAS=8R;TRACES=Synth;SCALES=3.0;NOTES='Tension Zone: All schedulers × 7b-8R-Synth × Scale 3.0'"
)

# All schedulers
ALL_SCHEDULERS="RANDOM,ROUND_ROBIN,LOR,LOP,LOP_BINARY,LOP_BATCH_END,LOP_UNCACHED,STICKY_RR,STICKY_LOR,RANKED_STICKY_LOP,TOLERANT_STICKY"

# Top 3 schedulers (placeholder - will be determined from Priority 1 results)
TOP3_SCHEDULERS="LOP,STICKY_LOR,LOR"

# ============================================================================
# FUNCTIONS
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

dry_run() {
    if [ "$DRY_RUN" == "1" ]; then
        echo "[DRY-RUN] $1"
    else
        echo "$1"
    fi
}

count_runs() {
    local schedulers=$1
    local models=$2
    local replicas=$3
    local traces=$4
    local scales=$5
    
    # Count comma-separated values
    local n_sched=$(echo "$schedulers" | tr ',' '\n' | wc -l)
    local n_models=$(echo "$models" | tr ',' '\n' | wc -l)
    local n_replicas=$(echo "$replicas" | tr ',' '\n' | wc -l)
    local n_traces=$(echo "$traces" | tr ',' '\n' | wc -l)
    local n_scales=$(echo "$scales" | tr ',' '\n' | wc -l)
    
    echo $((n_sched * n_models * n_replicas * n_traces * n_scales))
}

run_single_baseline() {
    local scheduler=$1
    local model=$2
    local replicas=$3
    local trace=$4
    local scale=$5
    
    local cmd="./scripts/run_baseline_sweep.sh $scheduler $model $replicas $trace $scale"
    
    dry_run "Would run: $cmd"
    
    if [ "$DRY_RUN" != "1" ]; then
        $cmd
    fi
}

generate_job_array() {
    local priority=$1
    
    # Parse configuration
    eval "${PRIORITY_CONFIGURATIONS[$priority]}"
    
    # Expand scheduler placeholders
    if [ "$SCHEDULERS" == "all" ]; then
        SCHEDULERS=$ALL_SCHEDULERS
    elif [ "$SCHEDULERS" == "top3" ]; then
        SCHEDULERS=$TOP3_SCHEDULERS
    fi
    
    log "Generating job array for Priority $priority"
    log "Configuration: $NOTES"
    
    # Generate all combinations, skipping already completed runs
    local combinations=()
    local skipped=0
    local total=0
    
    IFS=',' read -ra SCHED_ARRAY <<< "$SCHEDULERS"
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    IFS=',' read -ra REPLICA_ARRAY <<< "$REPLICAS"
    IFS=',' read -ra TRACE_ARRAY <<< "$TRACES"
    IFS=',' read -ra SCALE_ARRAY <<< "$SCALES"
    
    for sched in "${SCHED_ARRAY[@]}"; do
        for model in "${MODEL_ARRAY[@]}"; do
            for repl in "${REPLICA_ARRAY[@]}"; do
                for trace in "${TRACE_ARRAY[@]}"; do
                    for scale in "${SCALE_ARRAY[@]}"; do
                        total=$((total + 1))
                        
                        # Check if already completed
                        local search_dir="${SCRATCH_ROOT}/vidur-rl/baselines/${sched}/${model}-${repl}-${trace}/scale${scale}"
                        if [ -d "$search_dir" ]; then
                            local existing_summary=$(find "$search_dir" -name "summary.csv" -type f 2>/dev/null | head -1)
                            if [ -n "$existing_summary" ]; then
                                skipped=$((skipped + 1))
                                dry_run "SKIP: $sched / $model / $repl / $trace / Scale $scale (already completed)"
                                continue
                            fi
                        fi
                        
                        combinations+=("$sched:$model:$repl:$trace:$scale")
                    done
                done
            done
        done
    done
    
    # Output array file
    array_file="${SCRATCH_ROOT}/vidur-rl/baselines/priority_${priority}_array.txt"
    printf '%s\n' "${combinations[@]}" > "$array_file"
    
    local total_new=${#combinations[@]}
    log "Generated $total_new NEW combinations (skipped $skipped already completed)"
    log "Array file: $array_file"
}

submit_sbatch() {
    local priority=$1
    local array_file=$2
    
    log "Submitting SLURM job array for Priority $priority..."
    
    # Get array size before creating script
    local array_size=$(wc -l < "$array_file")
    log "Array size: $array_size tasks"
    
    # Create SLURM submission script with proper escaping
    # Use printf to avoid heredoc variable expansion issues
    cat > "${SCRATCH_ROOT}/vidur-rl/baselines/submit_priority_${priority}.sbatch" << ENDSCRIPT
#!/bin/bash
#SBATCH --job-name=vidur_baseline_p${priority}
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-${array_size}
#SBATCH --output=${SCRATCH_ROOT}/vidur-rl/baselines/slurm/slurm-%A_%a.out
#SBATCH --error=${SCRATCH_ROOT}/vidur-rl/baselines/slurm/slurm-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -e

# Environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd /projects/kamil@colostate.edu/vidur-rl

# Get configuration for this task
IFS=':' read -r scheduler model replicas trace scale < <(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$array_file")

echo "================================================================================"
echo "Priority ${priority} - Task: \${SLURM_ARRAY_TASK_ID}"
echo "Config: \${scheduler} / \${model} / \${replicas} / \${trace} / Scale \${scale}"
echo "================================================================================"

# Run baseline
./scripts/run_baseline_sweep.sh "\${scheduler}" "\${model}" "\${replicas}" "\${trace}" "\${scale}"

echo "Task complete: \$(date)"
ENDSCRIPT
    
    chmod +x "${SCRATCH_ROOT}/vidur-rl/baselines/submit_priority_${priority}.sbatch"
    
    log "Submitting job..."
    sbatch "${SCRATCH_ROOT}/vidur-rl/baselines/submit_priority_${priority}.sbatch"
}

status_check() {
    local priority=$1
    
    # Parse configuration
    eval "${PRIORITY_CONFIGURATIONS[$priority]}"
    
    # Expand scheduler placeholders
    if [ "$SCHEDULERS" == "all" ]; then
        SCHEDULERS=$ALL_SCHEDULERS
    elif [ "$SCHEDULERS" == "top3" ]; then
        SCHEDULERS=$TOP3_SCHEDULERS
    fi
    
    echo ""
    echo "=== Priority $priority Status ==="
    echo "Configuration: $NOTES"
    echo ""
    
    local completed=0
    local pending=0
    
    IFS=',' read -ra SCHED_ARRAY <<< "$SCHEDULERS"
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    IFS=',' read -ra REPLICA_ARRAY <<< "$REPLICAS"
    IFS=',' read -ra TRACE_ARRAY <<< "$TRACES"
    IFS=',' read -ra SCALE_ARRAY <<< "$SCALES"
    
    for sched in "${SCHED_ARRAY[@]}"; do
        for model in "${MODEL_ARRAY[@]}"; do
            for repl in "${REPLICA_ARRAY[@]}"; do
                for trace in "${TRACE_ARRAY[@]}"; do
                    for scale in "${SCALE_ARRAY[@]}"; do
                        local search_dir="${SCRATCH_ROOT}/vidur-rl/baselines/${sched}/${model}-${repl}-${trace}/scale${scale}"
                        if [ -d "$search_dir" ]; then
                            local existing_summary=$(find "$search_dir" -name "summary.csv" -type f 2>/dev/null | head -1)
                            if [ -n "$existing_summary" ]; then
                                completed=$((completed + 1))
                                echo "✓ $sched / $model / $repl / $trace / Scale $scale"
                                continue
                            fi
                        fi
                        pending=$((pending + 1))
                    done
                done
            done
        done
    done
    
    local total=$((completed + pending))
    echo ""
    echo "Completed: $completed / $total"
    echo "Pending: $pending"
}

# ============================================================================
# MAIN
# ============================================================================

usage() {
    echo "Usage: $0 <priority> [--dry-run] [--test] [--status]"
    echo ""
    echo "Priorities:"
    echo "  1   Essential: All schedulers × 7b-16R-Synth × All Scales (66 runs)"
    echo "  2   Model Comparison: Top 3 schedulers × 7b/70b × Scales 48-12 (18 runs)"
    echo "  3   Trace Difficulty: Top 3 schedulers × All traces × Scale 48 (9 runs)"
    echo "  4   Tension Zone: All schedulers × 7b-8R-Synth × Scale 3.0 (11 runs)"
    echo "  all  All priorities 1-4 (104 runs)"
    echo "  full Full matrix: All 792 combinations"
    echo ""
    echo "Options:"
    echo "  --dry-run    Show commands without executing"
    echo "  --test       Run single test config (LOP × 7b × 16R × Synth × 48.0)"
    echo "  --status     Show completed/pending status for priority"
}

# Parse arguments
PRIORITY=""
DRY_RUN=0
TEST_MODE=0
STATUS_MODE=0

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=1
            ;;
        --test)
            TEST_MODE=1
            ;;
        --status)
            STATUS_MODE=1
            ;;
        *)
            PRIORITY=$arg
            ;;
    esac
done

if [ -z "$PRIORITY" ]; then
    usage
    exit 1
fi

log "Starting Priority Baseline Sweep: $PRIORITY"

# Create output directories
mkdir -p "${SCRATCH_ROOT}/vidur-rl/baselines/slurm"

if [ "$STATUS_MODE" == "1" ]; then
    if [ -z "$PRIORITY" ] || [ "$PRIORITY" == "all" ]; then
        for p in 1 2 3 4; do
            status_check $p
            echo ""
        done
    else
        status_check $PRIORITY
    fi
    exit 0
fi

if [ "$TEST_MODE" == "1" ]; then
    log "Running TEST configuration..."
    dry_run "Would run: ./scripts/run_baseline_sweep.sh LOP 7b 16R Synth 48.0"
    if [ "$DRY_RUN" != "1" ]; then
        ./scripts/run_baseline_sweep.sh LOP 7b 16R Synth 48.0
    fi
    exit 0
fi

if [ "$PRIORITY" == "full" ]; then
    log "Running FULL MATRIX (all 792 combinations)"
    
    # Generate array file with all combinations
    ALL_MODELS="7b,70b"
    ALL_REPLICAS="8R,16R"
    ALL_TRACES="Synth,Tool,Conv"
    ALL_SCALES="48.0,24.0,12.0,6.0,3.0,1.5"
    
    generate_job_array_from_lists "$ALL_SCHEDULERS" "$ALL_MODELS" "$ALL_REPLICAS" "$ALL_TRACES" "$ALL_SCALES"
    
elif [ -n "${PRIORITY_CONFIGURATIONS[$PRIORITY]}" ]; then
    # Parse configuration
    eval "${PRIORITY_CONFIGURATIONS[$PRIORITY]}"
    
    # Expand scheduler placeholders
    if [ "$SCHEDULERS" == "all" ]; then
        SCHEDULERS=$ALL_SCHEDULERS
    elif [ "$SCHEDULERS" == "top3" ]; then
        SCHEDULERS=$TOP3_SCHEDULERS
    fi
    
    # Count runs
    total_runs=$(count_runs "$SCHEDULERS" "$MODELS" "$REPLICAS" "$TRACES" "$SCALES")
    log "Priority $PRIORITY: $total_runs runs"
    log "Description: $NOTES"
    
    # Generate array file
    generate_job_array "$PRIORITY"
    array_file="${SCRATCH_ROOT}/vidur-rl/baselines/priority_${PRIORITY}_array.txt"
    
    # Auto-submit (no interactive prompt when running via SSH)
    log "Submitting SLURM job array for Priority $PRIORITY..."
    submit_sbatch "$PRIORITY" "$array_file"
    
elif [ "$PRIORITY" == "all" ]; then
    log "Running all priorities 1-4..."
    
    for p in 1 2 3 4; do
        log "--- Priority $p ---"
        $0 $p --dry-run
    done
    
    echo ""
    read -p "Submit all priorities? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for p in 1 2 3 4; do
            log "Submitting Priority $p..."
            $0 $p
        done
    fi
else
    log_error "Unknown priority: $PRIORITY"
    usage
    exit 1
fi

log "Done!"
