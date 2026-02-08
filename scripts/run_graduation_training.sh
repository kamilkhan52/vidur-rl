#!/bin/bash
#
# Phase 3 Graduation Pipeline
#
# This script manages the "graduation" of RL models from low QPS to high QPS.
# It handles the model naming, architecture sweeps, and scale transitions.
#
# Usage:
#   ./scripts/run_graduation_training.sh <config_name> <initial_scale> [hidden_dims] [reward_type]

set -e

CONFIG_NAME=$1
CURRENT_SCALE=$2
HIDDEN_DIMS=${3:-"512,512,256,128"}
REWARD_TYPE=${4:-"e2e_time"}

if [ -z "$CONFIG_NAME" ] || [ -z "$CURRENT_SCALE" ]; then
    echo "Usage: $0 <config_name> <initial_scale> [hidden_dims] [reward_type]"
    echo "Example: $0 7b-8R-Synth 64.0 '512,512' ttft"
    exit 1
fi

# Derive training parameters from config_name
# Format: model-replicas-trace (e.g., 7b-8R-Synth)
MODEL_NAME="meta-llama/Llama-2-7b-hf"
[[ "$CONFIG_NAME" == *"70b"* ]] && MODEL_NAME="meta-llama/Llama-2-70b-hf"

NUM_REPLICAS=8
[[ "$CONFIG_NAME" == *"16R"* ]] && NUM_REPLICAS=16
[[ "$CONFIG_NAME" == *"4R"* ]] && NUM_REPLICAS=4

TRACE_NAME="synthetic_trace"
[[ "$CONFIG_NAME" == *"Tool"* ]] && TRACE_NAME="toolagent_trace"
[[ "$CONFIG_NAME" == *"Conv"* ]] && TRACE_NAME="conversation_trace_v2"

# Stable unique identifier for this "class" of model
MODEL_CLASS="grad_${CONFIG_NAME}_h${HIDDEN_DIMS//,/_}_r${REWARD_TYPE}"
CHECKPOINT_DIR="rl_models/graduation/${MODEL_CLASS}"
mkdir -p "$CHECKPOINT_DIR"

echo "================================================================================"
echo "🎓 Starting Graduation Pipeline for Class: $MODEL_CLASS"
echo "  Scale:      $CURRENT_SCALE"
echo "  Architecture: $HIDDEN_DIMS"
echo "  Reward:     $REWARD_TYPE"
echo "================================================================================"

# Graduation Loop
while true; do
    LATEST_MODEL="${CHECKPOINT_DIR}/scale${CURRENT_SCALE}_latest.pt"
    
    # Run the continuous trainer for a set number of iterations (e.g., 10)
    # We use env vars to override settings in the existing continuous trainer
    export TIME_SCALE_FACTOR=$CURRENT_SCALE
    export MODEL_NAME=$MODEL_NAME
    export NUM_REPLICAS=$NUM_REPLICAS
    export TRACE_NAME=$TRACE_NAME
    
    echo "--- Launching Training at Scale $CURRENT_SCALE ---"
    
    # We pass the hidden_dims and reward_type directly to the python command via custom logic or config override
    # For now, we'll wrap the existing trainer and hope it respects our env or we'll call vidur.main directly.
    
    # Custom training call for maximum control
    OUTPUT_DIR="/scratch/alpine/kamil@colostate.edu/vidur-rl/graduation/${MODEL_CLASS}/scale${CURRENT_SCALE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    # Determine if we have a model to load from the PREVIOUS scale
    LOAD_ARG=""
    if [ -f "$LATEST_MODEL" ]; then
        LOAD_ARG="--r_l_cache_aware_global_scheduler_config_model_checkpoint_path $LATEST_MODEL"
    fi

    # Execute Vidur with the current graduation parameters
    # Note: We run 10 iterations per scale "level"
    source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
    conda activate vidur

    # Create a sub-directory for this scale's iterations
    SCALE_DIR="${OUTPUT_DIR}"
    mkdir -p "$SCALE_DIR"

    for i in {1..20}; do
        echo "   - Starting Iteration $i..."
        ITER_DIR="${SCALE_DIR}/iter${i}"
        
        python3 -m vidur.main \
            --replica_config_device a100 \
            --replica_config_model_name "$MODEL_NAME" \
            --cluster_config_num_replicas "$NUM_REPLICAS" \
            --replica_scheduler_config_type vllm_v1 \
            --global_scheduler_config_type rl_cache_aware \
            --r_l_cache_aware_global_scheduler_config_mode training \
            --r_l_cache_aware_global_scheduler_config_reward_type "$REWARD_TYPE" \
            --r_l_cache_aware_global_scheduler_config_hidden_dims "$HIDDEN_DIMS" \
            --r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$LATEST_MODEL" \
            --trace_request_generator_config_time_scale_factor "$CURRENT_SCALE" \
            --trace_request_generator_config_trace_file "data/processed_traces/mooncake_${TRACE_NAME}.csv" \
            --metrics_config_output_dir "$ITER_DIR" \
            --no-metrics_config_write_json_trace \
            --request_generator_config_type trace \
            --cache_config_enable_prefix_caching \
            --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps 5000
    done

    # Evaluate for Graduation using the metrics from the LAST iteration
    METRICS_FILE=$(find "$SCALE_DIR/iter20" -name "request_metrics.csv" | head -1)
    if [ -f "$METRICS_FILE" ]; then
        MEAN_E2E=$(python3 -c "import pandas as pd; df = pd.read_csv('$METRICS_FILE'); print(df['request_e2e_time'].mean())")
    else
        echo "Error: Could not find metrics file for iteration 20"
        exit 1
    fi
    
    echo "Scale $CURRENT_SCALE Iteration Results: Mean E2E = ${MEAN_E2E}s"
    
    # Graduation Condition: If E2E is reasonable, lower the scale (increase load)
    # Target: If E2E < 10.0s (arbitrary baseline for stability), move to next stage
    if (( $(echo "$MEAN_E2E < 10.0" | bc -l) )); then
        NEW_SCALE=$(python3 -c "print($CURRENT_SCALE / 2.0)")
        echo "🎉 GRADUATED! Moving from Scale $CURRENT_SCALE to $NEW_SCALE"
        
        # Copy the best model to be the seed for the next scale
        NEXT_MODEL="${CHECKPOINT_DIR}/scale${NEW_SCALE}_latest.pt"
        cp "$LATEST_MODEL" "$NEXT_MODEL"
        
        CURRENT_SCALE=$NEW_SCALE
    else
        echo "⚖️ Scale $CURRENT_SCALE not yet stable. Retrying Stage..."
    fi

    # Exit if we hit the limit
    if (( $(echo "$CURRENT_SCALE < 1.0" | bc -l) )); then
        echo "🏆 Pipeline Complete! Target Scale 1.0 reached."
        break
    fi
done
