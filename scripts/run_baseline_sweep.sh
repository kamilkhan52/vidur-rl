#!/bin/bash
#
# Baseline Sweep Script - Heuristic Scheduler Evaluation
#
# Comprehensive baseline sweep across all heuristic schedulers for Vidur.
# Incorporates all lessons learned from RL training runs.
#
# Usage:
#   ./scripts/run_baseline_sweep.sh <scheduler> <model> <replicas> <trace> <scale>
#
# Examples:
#   # Test single config (recommended before full sweep)
#   ./scripts/run_baseline_sweep.sh LOP 7b 16R Synth 48.0
#
#   # Run priority 1 sweep (all schedulers × 7b-16R-Synth × all scales)
#   ./scripts/run_priority_baseline_sweep.sh 1
#
#   # Run full comprehensive sweep (all combinations)
#   ./scripts/run_priority_baseline_sweep.sh all
#
# Environment Variables:
#   ENABLE_CHROME_TRACE=0    # Always off to save disk space
#   OUTPUT_DIR_BASE           # Defaults to SCRATCH_DIR
#   NUM_ITERATIONS           # Number of iterations per config (default: 5)
#
# Best Practices Incorporated:
#   ✓ Chrome traces DISABLED (default=False in config)
#   ✓ Outputs to SCRATCH (/scratch/alpine/kamil@colostate.edu), not project
#   ✓ TP=4 for 70b models (required for 80GB A100)
#   ✓ Checkpoint frequency optimized for trace length
#   ✓ Uses existing heuristic schedulers (no RL)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories - detect if we're on Alpine or local
if [ -d "/projects/kamil@colostate.edu/vidur-rl" ]; then
    # On Alpine
    PROJECT_ROOT="/projects/kamil@colostate.edu/vidur-rl"
    SCRATCH_ROOT="/scratch/alpine/kamil@colostate.edu"
else
    # Local (WSL)
    PROJECT_ROOT="/home/kamil/vidur-rl-working"
    SCRATCH_ROOT="/home/kamil/vidur-rl-working/outputs"
fi

CACHE_DIR="${PROJECT_ROOT}/cache"

# Environment
export ENABLE_CHROME_TRACE=${ENABLE_CHROME_TRACE:-0}
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${SCRATCH_ROOT}/vidur-rl/baselines}"
export NUM_RUNS="${NUM_RUNS:-5}"

# Scheduler Type Mapping (config file names)
declare -A SCHEDULER_MAP=(
    ["RANDOM"]="random"
    ["ROUND_ROBIN"]="round_robin"
    ["LOR"]="lor"
    ["LOP"]="lop"
    ["LOP_BINARY"]="lop_binary"
    ["LOP_BATCH_END"]="lop_batch_end"
    ["LOP_UNCACHED"]="lop_uncached"
    ["STICKY_RR"]="sticky_round_robin"
    ["STICKY_LOR"]="sticky_lor"
    ["RANKED_STICKY_LOP"]="ranked_sticky_lop_uncached"
    ["TOLERANT_STICKY"]="tolerant_sticky_lop_uncached"
)

# Trace file mapping
declare -A TRACE_MAP=(
    ["Synth"]="synthetic_trace"
    ["Tool"]="toolagent_trace"
    ["Conv"]="conversation_trace_v2"
)

# ============================================================================
# FUNCTIONS
# ============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

check_requirements() {
    log_info "Checking requirements..."

    # Check project directory exists
    if [ ! -d "$PROJECT_ROOT" ]; then
        log_error "Project root not found: $PROJECT_ROOT"
        exit 1
    fi

    # Check cache directory exists (for predictors)
    if [ ! -d "$CACHE_DIR" ]; then
        log_error "Cache directory not found: $CACHE_DIR"
        log_error "Run: rsync -avz /projects/kamil@colostate.edu/vidur-rl/cache/ $CACHE_DIR/"
        exit 1
    fi

    log_info "All requirements met"
}

setup_environment() {
    log_info "Setting up environment..."

    # Create output directory
    mkdir -p "$OUTPUT_DIR_BASE"

    log_info "Environment ready"
}

get_model_name() {
    local model=$1
    if [ "$model" == "70b" ]; then
        echo "meta-llama/Llama-2-70b-hf"
    else
        echo "meta-llama/Llama-2-7b-hf"
    fi
}

get_tensor_parallel() {
    local model=$1
    if [ "$model" == "70b" ]; then
        echo "4"  # 70b needs TP=4 for 80GB A100
    else
        echo "1"
    fi
}

check_existing_run() {
    local scheduler=$1
    local model=$2
    local replicas=$3
    local trace=$4
    local scale=$5

    # Check if any summary.csv exists for this config
    local search_dir="${OUTPUT_DIR_BASE}/${scheduler}/${model}-${replicas}-${trace}/scale${scale}"

    if [ -d "$search_dir" ]; then
        # Find the most recent summary.csv (non-empty file)
        local latest_summary=$(find "$search_dir" -name "summary.csv" -type f 2>/dev/null | sort -r | head -1)

        if [ -n "$latest_summary" ] && [ -f "$latest_summary" ] && [ -s "$latest_summary" ]; then
            echo "$latest_summary"
            return 0  # Found - should skip
        fi
    fi

    return 1  # Not found - should run
}

resume_or_run() {
    local scheduler=$1
    local model=$2
    local replicas=$3
    local trace=$4
    local scale=$5

    # Check if config already completed
    if summary_file=$(check_existing_run "$scheduler" "$model" "$replicas" "$trace" "$scale"); then
        log_info "SKIPPING - Already completed: $scheduler / $model / $replicas / $trace / Scale $scale"
        log_info "Existing results: $summary_file"
        return 1  # Skip
    else
        return 0  # Run
    fi
}

run_baseline_single() {
    local scheduler=$1
    local model=$2
    local replicas=$3
    local trace=$4
    local scale=$5
    
    # Strip 'R' suffix from replicas for vidur command (16R -> 16)
    local replicas_num="${replicas%%R}"
    
    # Check for existing run first
    if ! resume_or_run "$scheduler" "$model" "$replicas" "$trace" "$scale"; then
        return 0  # Skipped
    fi
    
    log_info "Running baseline: scheduler=$scheduler model=$model replicas=$replicas trace=$trace scale=$scale"
    
    # Derive parameters
    local scheduler_type="${SCHEDULER_MAP[$scheduler]}"
    local trace_name="${TRACE_MAP[$trace]}"
    local model_name=$(get_model_name $model)
    local tp_size=$(get_tensor_parallel $model)
    
    # Output directory for this run
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_dir="${OUTPUT_DIR_BASE}/${scheduler}/${model}-${replicas}-${trace}/scale${scale}/${timestamp}"
    mkdir -p "$output_dir"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run baseline evaluation
    log_info "Starting evaluation..."
    log_info "Output: $output_dir"

    # Run multiple runs and aggregate
    local all_metrics=""
    for i in $(seq 1 $NUM_RUNS); do
        local run_dir="${output_dir}/run${i}"
        mkdir -p "$run_dir"
        
        log_info "Run $i/$NUM_RUNS"
        
        # EXACT PARAMETERS MATCHING RL SWEEP (job 23714214)
        python3 -m vidur.main \
            --replica_config_device a100 \
            --replica_config_model_name "$model_name" \
            --replica_config_tensor_parallel_size "$tp_size" \
            --cluster_config_num_replicas "$replicas_num" \
            --replica_scheduler_config_type vllm_v1 \
            --global_scheduler_config_type "$scheduler_type" \
            --trace_request_generator_config_time_scale_factor "$scale" \
            --trace_request_generator_config_trace_file "data/processed_traces/mooncake_${trace_name}.csv" \
            --metrics_config_output_dir "$run_dir" \
            --no-metrics_config_write_json_trace \
            --request_generator_config_type trace \
            --cache_config_enable_prefix_caching \
            --no-metrics_config_enable_chrome_trace 2>&1 | tee "${run_dir}/output.log"
        
        # Check if simulation succeeded (pipefail doesn't work with tee, so check explicitly)
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "Simulation failed for run $i (see ${run_dir}/output.log)"
            exit 1
        fi

        # Collect metrics (vidur creates timestamped subdirectory)
        local metrics_file=$(find "$run_dir" -name "request_metrics.csv" -type f 2>/dev/null | head -1)
        if [ -f "$metrics_file" ]; then
            local mean_e2e=$(python3 -c "import pandas as pd; print(pd.read_csv('$metrics_file')['request_e2e_time'].mean())")
            local p99_e2e=$(python3 -c "import pandas as pd; print(pd.read_csv('$metrics_file')['request_e2e_time'].quantile(0.99))")
            local throughput=$(python3 -c "import pandas as pd; df = pd.read_csv('$metrics_file'); print(len(df) / (df['request_e2e_time'].max() - df['request_arrival_time'].min()))")

            echo "$scheduler,$model,$replicas,$trace,$scale,$i,$mean_e2e,$p99_e2e,$throughput" >> "${output_dir}/metrics.csv"
        else
            log_error "Metrics file not found: $metrics_file (simulation may have failed silently)"
            exit 1
        fi
    done

    # Aggregate results
    if [ -f "${output_dir}/metrics.csv" ]; then
        log_info "Aggregating results..."
        python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('${output_dir}/metrics.csv', header=None,
                 names=['scheduler', 'model', 'replicas', 'trace', 'scale', 'run', 'e2e_mean', 'e2e_p99', 'throughput'])

summary = df.groupby(['scheduler', 'model', 'replicas', 'trace', 'scale']).agg({
    'e2e_mean': ['mean', 'std', 'min', 'max'],
    'e2e_p99': ['mean', 'std'],
    'throughput': 'mean'
}).round(4)

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.to_csv('${output_dir}/summary.csv')
print(summary.to_string())
"
    fi

    log_info "Baseline run complete: $output_dir"
    echo "$output_dir"
}

# ============================================================================
# MAIN
# ============================================================================

usage() {
    echo "Usage: $0 <scheduler> <model> <replicas> <trace> <scale>"
    echo ""
    echo "Arguments:"
    echo "  scheduler   One of: RANDOM, ROUND_ROBIN, LOR, LOP, LOP_BINARY,"
    echo "              LOP_BATCH_END, LOP_UNCACHED, STICKY_RR, STICKY_LOR,"
    echo "              RANKED_STICKY_LOP, TOLERANT_STICKY"
    echo "  model       Model size: 7b or 70b"
    echo "  replicas   Number of replicas: 8R or 16R"
    echo "  trace      Trace type: Synth, Tool, or Conv"
    echo "  scale      Request scale factor: 48.0, 24.0, 12.0, 6.0, 3.0, or 1.5"
    echo ""
    echo "Examples:"
    echo "  $0 LOP 7b 16R Synth 48.0"
    echo "  $0 ROUND_ROBIN 70b 8R Conv 24.0"
    echo ""
    echo "Environment Variables:"
    echo "  ENABLE_CHROME_TRACE=0    # Always disabled (default)"
    echo "  OUTPUT_DIR_BASE         # Default: SCRATCH/baselines"
    echo "  NUM_ITERATIONS=5        # Iterations per config"
}

# Parse arguments
if [ $# -lt 5 ]; then
    usage
    exit 1
fi

SCHEDULER=$1
MODEL=$2
REPLICAS=$3
TRACE=$4
SCALE=$5

# Validate inputs
if [ -z "${SCHEDULER_MAP[$SCHEDULER]}" ]; then
    log_error "Unknown scheduler: $SCHEDULER"
    log_error "Valid schedulers: ${!SCHEDULER_MAP[*]}"
    exit 1
fi

if [ "$MODEL" != "7b" ] && [ "$MODEL" != "70b" ]; then
    log_error "Unknown model: $MODEL (must be 7b or 70b)"
    exit 1
fi

if [ "$REPLICAS" != "8R" ] && [ "$REPLICAS" != "16R" ]; then
    log_error "Unknown replicas: $REPLICAS (must be 8R or 16R)"
    exit 1
fi

if [ -z "${TRACE_MAP[$TRACE]}" ]; then
    log_error "Unknown trace: $TRACE (must be Synth, Tool, or Conv)"
    exit 1
fi

# Run
check_requirements
setup_environment
run_baseline_single "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"
