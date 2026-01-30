#!/bin/bash
#
# Train RL Agent on 16-GPU System with Toolagent Trace
#
# Configuration aligned with eval_16gpu_toolagent.sh:
#   - 16 replicas (16 GPUs total, TP=1, PP=1)
#   - Time scale: 3.0x (matches eval)
#   - Trace: toolagent
#
# Usage:
#   ./scripts/train_16gpu_toolagent.sh [checkpoint_path]
#
# Environment variables:
#   EPSILON_START=1.0          # Initial epsilon (default: 1.0)
#   EPSILON_END=0.005          # Final epsilon (default: 0.005)
#   EPSILON_DECAY_STEPS=5000   # Decay over N steps (default: 5000)
#   RESET_EPSILON_ON_LOAD=0    # Set to 1 to ignore loaded epsilon state and start fresh
#   LOG_TRANSITIONS=0          # Set to 1 to log (state, action, reward) to CSV

set -e

# Configuration matching eval_16gpu_toolagent.sh
NUM_REPLICAS=16
TIME_SCALE_FACTOR=3.0
TRACE_FILE="data/processed_traces/mooncake_toolagent_trace.csv"
MAX_TOKENS=122752
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"

# Epsilon configuration
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.005}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-5000}"

# Transition logging (set LOG_TRANSITIONS=1 to enable)
LOG_TRANSITIONS="${LOG_TRANSITIONS:-0}"

# Reset epsilon on load (set RESET_EPSILON_ON_LOAD=1 to ignore saved epsilon state)
RESET_EPSILON_ON_LOAD="${RESET_EPSILON_ON_LOAD:-0}"

echo "================================================================================"
echo "  16-GPU Toolagent RL Training - Press Ctrl+C to stop"
echo "================================================================================"
echo ""
echo "System Configuration (matching eval_16gpu_toolagent.sh):"
echo "  Replicas:    $NUM_REPLICAS"
echo "  Time scale:  ${TIME_SCALE_FACTOR}x"
echo "  Trace:       toolagent ($TRACE_FILE)"
echo "  Max tokens:  $MAX_TOKENS"
echo "  Model:       $MODEL_NAME"
echo ""
echo "Epsilon Configuration:"
echo "  Start: $EPSILON_START"
echo "  End: $EPSILON_END"
echo "  Decay steps: $EPSILON_DECAY_STEPS"
if [ "$RESET_EPSILON_ON_LOAD" = "1" ]; then
    echo "  Reset on load: ENABLED (will ignore saved epsilon state)"
else
    echo "  Reset on load: Disabled (will use saved epsilon state if available)"
fi
echo ""
echo "Transition Logging:"
if [ "$LOG_TRANSITIONS" = "1" ]; then
    echo "  ENABLED - CSV logs will be saved per iteration"
else
    echo "  Disabled (set LOG_TRANSITIONS=1 to enable)"
fi
echo ""

# Directories
RUN_TAG="16gpu_toolagent_scale${TIME_SCALE_FACTOR}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="/tmp/mooncake_training_logs/${RUN_TAG}"
CHECKPOINT_DIR="rl_models"
CHECKPOINT_NAME="16gpu_toolagent_scale${TIME_SCALE_FACTOR}"

# Optional: Load existing model
LOAD_MODEL_PATH="${1:-}"

# Verify trace exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Trace file not found: $TRACE_FILE"
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
echo "Checkpoint:  $CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
echo "Log:         $LOG_DIR/training.log"
echo "History:     $LOG_DIR/training_history.txt"
echo ""
echo "How it works:"
echo "  - Each iteration runs full toolagent trace"
echo "  - Model is loaded -> trained -> saved after each iteration"
echo "  - Next iteration continues from updated model"
echo "  - Training continues until you press Ctrl+C"
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
    OUTPUT_DIR="results/16gpu_toolagent_training_iter${ITERATION}_$(date +%Y%m%d_%H%M%S)"
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
        --trace_request_generator_config_max_tokens "$MAX_TOKENS"
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

    # Only reset epsilon on first iteration (if requested)
    if [ $ITERATION -eq 1 ] && [ "$RESET_EPSILON_ON_LOAD" = "1" ]; then
        CMD+=(--r_l_cache_aware_global_scheduler_config_reset_epsilon_on_load)
        echo "  Epsilon reset: ENABLED (first iteration only)"
    fi

    # If first iteration with user-provided model, copy it
    if [ $ITERATION -eq 1 ] && [ -n "$LOAD_MODEL_PATH" ]; then
        cp "$LOAD_MODEL_PATH" "$CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
        echo "Copied initial checkpoint"
    fi

    # Run training iteration
    echo "Starting iteration $ITERATION..."
    echo "Command: ${CMD[*]}"
    echo ""
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

    # Extract final training step count
    FINAL_STEP=$(grep -oP 'Steps: \K\d+' "$OUTPUT_DIR/simulation.log" | tail -1)
    if [ -n "$FINAL_STEP" ]; then
        TOTAL_TRAINING_STEPS=$((TOTAL_TRAINING_STEPS + FINAL_STEP))
    fi

    # Extract metrics
    METRICS_FILE=$(find "$OUTPUT_DIR" -name "request_metrics.csv" | head -1)
    if [ -f "$METRICS_FILE" ]; then
        # Calculate metrics
        METRICS=$(python3 -c "
import pandas as pd

df = pd.read_csv('$METRICS_FILE')

mean_e2e = df['request_e2e_time'].mean()
median_e2e = df['request_e2e_time'].median()
p99_e2e = df['request_e2e_time'].quantile(0.99)

print(f'Mean E2E: {mean_e2e:.3f}s')
print(f'Median E2E: {median_e2e:.3f}s')
print(f'P99 E2E: {p99_e2e:.3f}s')

# Cache hit rate
if 'request_num_prefill_tokens_cached' in df.columns:
    cached = df['request_num_prefill_tokens_cached']
    total = df['request_num_prefill_tokens']
    cache_rate = 100 * cached.sum() / total.sum() if total.sum() > 0 else 0
    print(f'Cache hit rate: {cache_rate:.1f}%')

# Load distribution
print()
print('Load Distribution:')
counts = df['replica'].value_counts().sort_index()
for replica, count in counts.items():
    pct = 100.0 * count / len(df)
    print(f'  Replica {replica:2d}: {count:5d} ({pct:5.1f}%)')

imbalance = (counts.max() - counts.min()) / len(df)
print(f'  Imbalance: {imbalance:.4f}')

# Output summary values
print(f'MEAN_E2E:{mean_e2e:.6f}')
")

        MEAN_E2E=$(echo "$METRICS" | grep "MEAN_E2E:" | cut -d: -f2)

        echo ""
        echo "================================================================================"
        echo "Iteration $ITERATION Complete"
        echo "  Total training steps: $TOTAL_TRAINING_STEPS"
        echo ""
        echo "$METRICS" | grep -v "MEAN_E2E:"
        echo ""

        # Check if this is best so far
        if (( $(echo "$MEAN_E2E < $BEST_E2E" | bc -l) )); then
            BEST_E2E=$MEAN_E2E
            echo "  NEW BEST E2E: ${BEST_E2E}s"
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
            echo "  Training plot: $PLOT_PATH"
        fi
    fi

    ITERATION=$((ITERATION + 1))

    # Small delay before next iteration
    sleep 2
done
