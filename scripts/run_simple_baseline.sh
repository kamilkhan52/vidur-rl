#!/bin/bash
#
# Simple Baseline Runner - Runs baseline configs one at a time
#
# Usage: ./scripts/run_simple_baseline.sh <scheduler> <model> <replicas> <trace> <scale>
#
# This script bypasses the array job complexity and runs configs directly.
# Use this if the array job approach fails.

set -e

# Arguments
SCHEDULER=$1
MODEL=$2
REPLICAS=$3
TRACE=$4
SCALE=$5

# Validate
if [ -z "$SCHEDULER" ] || [ -z "$MODEL" ] || [ -z "$REPLICAS" ] || [ -z "$TRACE" ] || [ -z "$SCALE" ]; then
    echo "Usage: $0 <scheduler> <model> <replicas> <trace> <scale>"
    echo "Example: $0 LOP 7b 16R Synth 48.0"
    exit 1
fi

# Setup
SCRATCH_ROOT="/scratch/alpine/kamil@colostate.edu"
OUTPUT_DIR="${SCRATCH_ROOT}/vidur-rl/baselines/${SCHEDULER}/${MODEL}-${REPLICAS}-${TRACE}/scale${SCALE}"

echo "Running baseline: $SCHEDULER / $MODEL / $REPLICAS / $TRACE / Scale $SCALE"
echo "Output: $OUTPUT_DIR"

# Check if already completed
if [ -f "${OUTPUT_DIR}/summary.csv" ]; then
    echo "SKIPPING - Already completed"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd /projects/kamil@colostate.edu/vidur-rl

# Derive parameters
if [ "$MODEL" == "70b" ]; then
    MODEL_NAME="meta-llama/Llama-2-70b-hf"
    TP_SIZE="4"
else
    MODEL_NAME="meta-llama/Llama-2-7b-hf"
    TP_SIZE="1"
fi

REPLICAS_NUM="${REPLICAS%%R}"

# Map trace name
case "$TRACE" in
    Synth) TRACE_FILE="synthetic_trace" ;;
    Tool) TRACE_FILE="toolagent_trace" ;;
    Conv) TRACE_FILE="conversation_trace" ;;  # canonical v1
    *) TRACE_FILE="$TRACE" ;;
esac

# Run single baseline
python3 -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name "$MODEL_NAME" \
    --replica_config_tensor_parallel_size "$TP_SIZE" \
    --cluster_config_num_replicas "$REPLICAS_NUM" \
    --replica_scheduler_config_type vllm_v1 \
    --global_scheduler_config_type "$SCHEDULER" \
    --trace_request_generator_config_time_scale_factor "$SCALE" \
    --trace_request_generator_config_trace_file "data/processed_traces/mooncake_${TRACE_FILE}.csv" \
    --trace_request_generator_config_max_tokens "$([[ "$MODEL_NAME" == *70b* ]] && echo 525920 || echo 122752)" \
    --metrics_config_output_dir "$OUTPUT_DIR" \
    --no-metrics_config_write_json_trace \
    --request_generator_config_type trace \
    --cache_config_enable_prefix_caching \
    --no-metrics_config_enable_chrome_trace

echo "Complete: $OUTPUT_DIR"
