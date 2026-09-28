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
#   NUM_RUNS                 # Runs per config (default: 1; identical unless SEED_BASE set)
#   SEED_BASE                # If set, run i uses seed SEED_BASE+i (simulator, trace, scheduler RNG)
#   MAX_TOKENS_7B/70B        # Trace filter = per-replica KV capacity (122752 / 525920)
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
# NOTE (2026-09-21): the simulator is deterministic for a given seed, so repeated
# runs with the same seed are bit-identical. NUM_RUNS>1 only adds information when
# SEED_BASE is set (run i uses seed SEED_BASE+i for the simulator, the request
# generator and the scheduler's own tie-break RNG).
export NUM_RUNS="${NUM_RUNS:-1}"
export SEED_BASE="${SEED_BASE:-}"
# Optional: replay only the first NUM_REQUESTS requests of the trace (smoke tests,
# cheap evaluation stages). Unset = full trace.
export NUM_REQUESTS="${NUM_REQUESTS:-}"
# Optional arrival jitter in trace seconds (see TraceRequestGeneratorConfig.arrival_jitter).
# Mooncake conv/tool timestamps are 3 s buckets; 3.0 spreads each bucket. Unset/0 = exact replay.
export ARRIVAL_JITTER="${ARRIVAL_JITTER:-}"
# Requests larger than a replica's KV cache can never be scheduled and used to
# wedge the replica silently (see vidur/simulator.py::_validate_request_capacity).
# Per-replica capacity on an 80 GB A100: 7b/TP1 = 122752 tokens, 70b/TP4 = 525920.
MAX_TOKENS_7B="${MAX_TOKENS_7B:-122752}"
MAX_TOKENS_70B="${MAX_TOKENS_70B:-525920}"

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
    ["Conv"]="conversation_trace"
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

get_max_tokens() {
    local model=$1
    if [ "$model" == "70b" ]; then
        echo "$MAX_TOKENS_70B"
    else
        echo "$MAX_TOKENS_7B"
    fi
}

# Flag name of the per-scheduler RNG seed (Vidur derives it from the config class name).
declare -A SEED_FLAG_MAP=(
    ["random"]="random_global_scheduler_config_seed"
    ["round_robin"]="round_robin_global_scheduler_config_seed"
    ["lor"]="l_o_r_global_scheduler_config_seed"
    ["lop"]="l_o_p_global_scheduler_config_seed"
    ["lop_binary"]="l_o_p_binary_global_scheduler_config_seed"
    ["lop_batch_end"]="l_o_p_batch_end_global_scheduler_config_seed"
    ["lop_uncached"]="l_o_p_uncached_global_scheduler_config_seed"
    ["sticky_round_robin"]="sticky_round_robin_global_scheduler_config_seed"
    ["sticky_lor"]="sticky_lor_global_scheduler_config_seed"
    ["ranked_sticky_lop_uncached"]="ranked_sticky_lop_uncached_global_scheduler_config_seed"
    ["tolerant_sticky_lop_uncached"]="tolerant_sticky_lop_uncached_global_scheduler_config_seed"
)

get_tensor_parallel() {
    local model=$1
    if [ "$model" == "70b" ]; then
        echo "4"  # 70b needs TP=4 for 80GB A100
    else
        echo "1"
    fi
}

resolve_trace_file() {
    local trace_name=$1
    local v2="data/processed_traces/mooncake_${trace_name}_v2.csv"
    local base="data/processed_traces/mooncake_${trace_name}.csv"

    if [ -f "$v2" ]; then
        echo "$v2"
    elif [ -f "$base" ]; then
        echo "$base"
    else
        # Return default path for clear downstream error message
        echo "$base"
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
    local trace_file=$(resolve_trace_file "$trace_name")
    local model_name=$(get_model_name $model)
    local tp_size=$(get_tensor_parallel $model)
    local max_tokens=$(get_max_tokens $model)
    
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

        local seed_args=()
        if [ -n "$SEED_BASE" ]; then
            local run_seed=$((SEED_BASE + i))
            seed_args=(--seed "$run_seed" --trace_request_generator_config_seed "$run_seed"
                       "--${SEED_FLAG_MAP[$scheduler_type]}" "$run_seed")
            log_info "Seed: $run_seed"
        fi
        local limit_args=()
        if [ -n "$NUM_REQUESTS" ]; then
            limit_args=(--trace_request_generator_config_num_requests "$NUM_REQUESTS")
            log_info "Request limit: $NUM_REQUESTS"
        fi
        if [ -n "$ARRIVAL_JITTER" ]; then
            limit_args+=(--trace_request_generator_config_arrival_jitter "$ARRIVAL_JITTER")
            log_info "Arrival jitter: $ARRIVAL_JITTER s"
        fi

        # EXACT PARAMETERS MATCHING RL SWEEP (job 23714214), plus the KV-capacity filter
        python3 -m vidur.main \
            "${seed_args[@]}" \
            "${limit_args[@]}" \
            --trace_request_generator_config_max_tokens "$max_tokens" \
            --replica_config_device a100 \
            --replica_config_model_name "$model_name" \
            --replica_config_tensor_parallel_size "$tp_size" \
            --cluster_config_num_replicas "$replicas_num" \
            --replica_scheduler_config_type vllm_v1 \
            --global_scheduler_config_type "$scheduler_type" \
            --trace_request_generator_config_time_scale_factor "$scale" \
            --trace_request_generator_config_trace_file "$trace_file" \
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
            local throughput=$(python3 -c "import pandas as pd, numpy as np; df=pd.read_csv('$metrics_file'); n=len(df); arr='request_arrival_time' if 'request_arrival_time' in df.columns else ('request_arrived_at' if 'request_arrived_at' in df.columns else None); dur=(df['request_e2e_time'].max()-df[arr].min()) if arr else np.nan; print((n/dur) if (pd.notna(dur) and dur>0) else np.nan)")

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
    echo "  NUM_RUNS=1              # Runs per config (identical unless SEED_BASE is set)"
    echo "  SEED_BASE=              # If set, run i uses seed SEED_BASE+i"
    echo "  MAX_TOKENS_7B/70B       # KV-capacity trace filter (defaults 122752 / 525920)"
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
