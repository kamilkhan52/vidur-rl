#!/bin/bash
#
# Continuous Mooncake RL Training with Prefix Caching
#
# Continuously trains on Mooncake trace with caching enabled.
# Uses 68-dimensional state (includes cache affinity features).
# Loops until Ctrl+C is pressed.
#
# Usage:
#   ./scripts/train_mooncake_continuous.sh [checkpoint_path]
#
# Environment variables:
#   EPSILON_START=1.0          # Initial epsilon (default: 1.0)
#   EPSILON_END=0.05           # Final epsilon (default: 0.05)
#   EPSILON_DECAY_STEPS=2500    # Decay over N steps (default: 2500)
#   LOG_TRANSITIONS=0          # Set to 1 to log (state, action, reward) to CSV
#   TRACE_NAME=conversation_v2 # conversation|conversation_v2|toolagent|synthetic
#   TRACE_MAX_TOKENS=          # Optional override for max tokens (KV cache cap)
#   MAX_TOKENS_CAP=122752      # Default KV cache cap for auto max token detection
#   MODEL_NAME=meta-llama/Llama-2-7b-hf  # Model name override

set -e

# Epsilon configuration
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.005}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-5000}"

# Transition logging (set LOG_TRANSITIONS=1 to enable)
LOG_TRANSITIONS="${LOG_TRANSITIONS:-0}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"

# Trace selection
TRACE_NAME="${TRACE_NAME:-conversation_v2}"
MAX_TOKENS_CAP="${MAX_TOKENS_CAP:-122752}"

# Time scale factor for trace replay (to match 4 GPU capacity vs 128 GPU original)
# Original Mooncake trace: 3.4 QPS on 128 A800 GPUs (16 nodes × 8 GPUs)
# Scaling: 20x slower for just 4 GPUs
TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR:-20.0}"

echo "================================================================================"
echo "  Continuous Mooncake RL Training - Press Ctrl+C to stop"
echo "================================================================================"
echo ""
echo "Epsilon Configuration:"
echo "  Start: $EPSILON_START"
echo "  End: $EPSILON_END"
echo "  Decay steps: $EPSILON_DECAY_STEPS"
echo ""
echo "Transition Logging:"
if [ "$LOG_TRANSITIONS" = "1" ]; then
    echo "  ENABLED - CSV logs will be saved per iteration"
else
    echo "  Disabled (set LOG_TRANSITIONS=1 to enable)"
fi
echo ""
echo "Trace Scaling:"
echo "  Time scale factor: ${TIME_SCALE_FACTOR}x (slows arrival rate)"
echo "  Original: 3.4 QPS → Scaled: $(python3 -c "print(f'{3.4 / ${TIME_SCALE_FACTOR}:.2f}')") QPS"
echo ""

# Configuration
TRACE_FILE=""
TRACE_LABEL=""
case "$TRACE_NAME" in
    conversation|conversation_v1)
        TRACE_FILE="data/processed_traces/mooncake_conversation_trace.csv"
        TRACE_LABEL="Conversation (v1)"
        ;;
    conversation_v2)
        TRACE_FILE="data/processed_traces/mooncake_conversation_trace_v2.csv"
        TRACE_LABEL="Conversation (v2)"
        ;;
    toolagent)
        TRACE_FILE="data/processed_traces/mooncake_toolagent_trace.csv"
        TRACE_LABEL="Tool&Agent"
        ;;
    synthetic)
        TRACE_FILE="data/processed_traces/mooncake_synthetic_trace.csv"
        TRACE_LABEL="Synthetic"
        ;;
    *)
        echo "Error: Unknown TRACE_NAME '$TRACE_NAME' (use conversation, conversation_v2, toolagent, synthetic)"
        exit 1
        ;;
esac

TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-}"
if [ -z "$TRACE_MAX_TOKENS" ]; then
    TRACE_MAX_TOKENS=$(TRACE_FILE="$TRACE_FILE" MAX_TOKENS_CAP="$MAX_TOKENS_CAP" python3 - <<'PY' || echo "$MAX_TOKENS_CAP"
import os
import pandas as pd

trace_file = os.environ["TRACE_FILE"]
cap = int(os.environ["MAX_TOKENS_CAP"])

df = pd.read_csv(trace_file)
if "num_prefill_tokens" in df.columns and "num_decode_tokens" in df.columns:
    max_tokens = (df["num_prefill_tokens"] + df["num_decode_tokens"]).max()
else:
    max_tokens = (df["input_length"] + df["output_length"]).max()

print(min(int(max_tokens), cap))
PY
    )
fi
RUN_TAG="${TRACE_NAME}_scale${TIME_SCALE_FACTOR}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="/tmp/mooncake_training_logs/${RUN_TAG}"
CHECKPOINT_DIR="rl_models"
CHECKPOINT_NAME="mooncake_continuous_${TRACE_NAME}_scale${TIME_SCALE_FACTOR}"

# Optional: Load existing model
LOAD_MODEL_PATH="${1:-}"

# Verify trace exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Mooncake trace not found: $TRACE_FILE"
    exit 1
fi

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Check if continuing from checkpoint
if [ -n "$LOAD_MODEL_PATH" ]; then
    if [ ! -f "$LOAD_MODEL_PATH" ]; then
        echo "Error: Checkpoint not found: $LOAD_MODEL_PATH"
        exit 1
    fi
    echo "Mode: Continue from checkpoint"
    echo "  Loading: $LOAD_MODEL_PATH"
else
    echo "Mode: Train from scratch"
fi

echo ""
echo "Configuration:"
echo "  Trace:       $TRACE_LABEL ($TRACE_FILE)"
echo "  Max tokens:  $TRACE_MAX_TOKENS (cap: $MAX_TOKENS_CAP)"
echo "  Caching:     ENABLED (68-dim state with cache features)"
echo "  Model:       $MODEL_NAME"
echo "  Checkpoint:  $CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
echo "  Log:         $LOG_DIR/training.log"
echo "  History:     $LOG_DIR/training_history.txt"
echo ""
echo "How it works:"
echo "  • Each iteration runs full Mooncake trace (12K requests)"
echo "  • Model is loaded → trained → saved after each iteration"
echo "  • Next iteration continues from updated model"
echo "  • Training continues until you press Ctrl+C"
echo ""
echo "Monitor training:"
echo "  tail -f $LOG_DIR/training.log           # Live training progress"
echo "  cat $LOG_DIR/training_history.txt        # Performance over time"
echo ""
echo "================================================================================"
echo ""

# Clear logs
rm -f "$LOG_DIR/training.log"
rm -f "$LOG_DIR/training_history.txt"

# Training loop
ITERATION=1
TOTAL_TRAINING_STEPS=0
BEST_E2E=999999.0

echo "Training Progress History:" | tee "$LOG_DIR/training_history.txt"
echo "======================================================" | tee -a "$LOG_DIR/training_history.txt"

while true; do
    echo "================================================================================"
    echo "Iteration $ITERATION (Total steps: $TOTAL_TRAINING_STEPS)"
    echo "================================================================================"
    echo ""

    # Create output directory
    OUTPUT_DIR="results/mooncake_continuous_iter${ITERATION}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    # Build command
    CMD=(
        python -m vidur.main
        --replica_config_device a100
        --replica_config_model_name "$MODEL_NAME"
        --cluster_config_num_replicas "$NUM_REPLICAS"
        --replica_scheduler_config_type vllm_v1
        --global_scheduler_config_type rl_cache_aware
        --r_l_cache_aware_global_scheduler_config_mode training
        --r_l_cache_aware_global_scheduler_config_epsilon_start "$EPSILON_START"
        --r_l_cache_aware_global_scheduler_config_epsilon_end "$EPSILON_END"
        --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps "$EPSILON_DECAY_STEPS"
        --r_l_cache_aware_global_scheduler_config_learning_rate 0.0001
        --r_l_cache_aware_global_scheduler_config_batch_size 128
        --r_l_cache_aware_global_scheduler_config_update_frequency 50
        --r_l_cache_aware_global_scheduler_config_replay_buffer_size 10000
        --request_generator_config_type trace
        --trace_request_generator_config_trace_file "$TRACE_FILE"
        --trace_request_generator_config_time_scale_factor "$TIME_SCALE_FACTOR"
        --trace_request_generator_config_max_tokens "$TRACE_MAX_TOKENS"
        --cache_config_enable_prefix_caching
        --no-metrics_config_write_json_trace
        --no-metrics_config_store_operation_metrics
        --no-metrics_config_store_batch_metrics
        --no-metrics_config_enable_chrome_trace
        --metrics_config_output_dir "$OUTPUT_DIR"
    )

    # Add checkpoint path
    CMD+=(--r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt")

    # Add transition logging if enabled
    if [ "$LOG_TRANSITIONS" = "1" ]; then
        TRANSITION_LOG="$OUTPUT_DIR/transitions.csv"
        CMD+=(--r_l_cache_aware_global_scheduler_config_transition_log_path "$TRANSITION_LOG")
        echo "  Transition log: $TRANSITION_LOG"
    fi

    # If first iteration with user-provided model, copy it
    if [ $ITERATION -eq 1 ] && [ -n "$LOAD_MODEL_PATH" ]; then
        cp "$LOAD_MODEL_PATH" "$CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
        echo "Copied initial checkpoint"
    fi

    # Run training iteration
    echo "Starting iteration $ITERATION..."
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

    # Extract final training step count
    FINAL_STEP=$(grep -oP 'Steps: \K\d+' "$OUTPUT_DIR/simulation.log" | tail -1)
    if [ -n "$FINAL_STEP" ]; then
        TOTAL_TRAINING_STEPS=$((TOTAL_TRAINING_STEPS + FINAL_STEP))
    fi

    # Extract metrics
    METRICS_FILE=$(find "$OUTPUT_DIR" -name "request_metrics.csv" | head -1)
    if [ -f "$METRICS_FILE" ]; then
        # Calculate mean E2E
        MEAN_E2E=$(python3 -c "import pandas as pd; df = pd.read_csv('$METRICS_FILE'); print(f'{df[\"request_e2e_time\"].mean():.3f}')")

        # Calculate load distribution
        LOAD_DIST=$(python3 -c "
import pandas as pd
df = pd.read_csv('$METRICS_FILE')
counts = df['replica'].value_counts().sort_index()
for replica, count in counts.items():
    pct = 100.0 * count / len(df)
    print(f'    Replica {replica}: {count:5d} ({pct:5.1f}%)')
max_load = counts.max()
min_load = counts.min()
imbalance = (max_load - min_load) / len(df)
print(f'    Load imbalance: {imbalance:.4f}')
")

        echo ""
        echo "================================================================================"
        echo "Iteration $ITERATION Complete"
        echo "  Total training steps: $TOTAL_TRAINING_STEPS"
        echo "  Mean E2E latency: ${MEAN_E2E}s"
        echo ""
        echo "  Load Distribution:"
        echo "$LOAD_DIST"
        echo ""

        # Check if this is best so far
        if (( $(echo "$MEAN_E2E < $BEST_E2E" | bc -l) )); then
            BEST_E2E=$MEAN_E2E
            echo "  ✨ NEW BEST E2E: ${BEST_E2E}s"
            cp "$CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt" "$CHECKPOINT_DIR/${CHECKPOINT_NAME}_best.pt"
            echo "  Saved best checkpoint: ${CHECKPOINT_NAME}_best.pt"
        fi

        echo "  Checkpoint: $CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
        echo "================================================================================"
        echo ""

        # Save to history
        echo "Iter $ITERATION | Steps: $TOTAL_TRAINING_STEPS | E2E: ${MEAN_E2E}s" >> "$LOG_DIR/training_history.txt"

        # Generate training progress plot
        PLOT_PATH="$LOG_DIR/training_progress.png"
        if python3 scripts/plot_training_progress.py "$PLOT_PATH" 2>/dev/null; then
            echo ""
            echo "  📊 Training plot: $PLOT_PATH"
        fi
    fi

    ITERATION=$((ITERATION + 1))

    # Small delay before next iteration
    sleep 2
done
