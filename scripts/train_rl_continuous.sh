#!/bin/bash
#
# Continuous RL Training with Live Monitoring
#
# This script continuously trains the RL model, saving checkpoints periodically.
# Training continues until the user presses Ctrl+C.
#
# Usage:
#   ./scripts/train_rl_continuous.sh [model_path]
#
# Environment variables (optional):
#   EPSILON_START=1.0              # Initial epsilon (default: 1.0)
#   EPSILON_END=0.0                # Final epsilon (default: 0.0)
#   EPSILON_DECAY_STEPS=500        # Steps to decay (default: 500)
#
# Examples:
#   # Start fresh with default epsilon decay
#   ./scripts/train_rl_continuous.sh
#
#   # Continue from checkpoint (epsilon state is preserved)
#   ./scripts/train_rl_continuous.sh rl_models/continuous_training_latest.pt
#
#   # Start fresh with custom epsilon decay
#   EPSILON_DECAY_STEPS=1000 ./scripts/train_rl_continuous.sh
#
#   # Start fresh with slower decay (more exploration)
#   EPSILON_START=1.0 EPSILON_END=0.1 EPSILON_DECAY_STEPS=2000 ./scripts/train_rl_continuous.sh

set -e

# Default epsilon parameters (can be overridden)
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.05}"  # Changed: keep 5% exploration (was 0.0)
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-500}"  # Changed: decay slower (was 500)

echo "================================================================================"
echo "  Continuous RL Training - Press Ctrl+C to stop"
echo "================================================================================"
echo ""
echo "Epsilon Configuration:"
echo "  Start: $EPSILON_START"
echo "  End: $EPSILON_END"
echo "  Decay steps: $EPSILON_DECAY_STEPS"
echo ""

# Configuration
TRACE="/home/kamil/vidur/data/processed_traces/splitwise_conv.csv"
LOG_DIR="/tmp/rl_training_logs"
CHECKPOINT_DIR="rl_models"

# Optional: Load existing model to continue training
LOAD_MODEL_PATH="${1:-}"

# Check trace exists
if [ ! -f "$TRACE" ]; then
    echo "Error: Trace file not found: $TRACE"
    exit 1
fi

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Check if continuing from existing model
if [ -n "$LOAD_MODEL_PATH" ]; then
    if [ ! -f "$LOAD_MODEL_PATH" ]; then
        echo "Error: Model checkpoint not found: $LOAD_MODEL_PATH"
        exit 1
    fi
    echo "Mode: Continue training from existing model"
    echo "  Loading from: $LOAD_MODEL_PATH"
else
    echo "Mode: Train from scratch (random initialization)"
fi

echo ""
echo "Configuration:"
echo "  Trace:      $TRACE"
echo "  Requests:   $(wc -l < "$TRACE") requests per iteration"
echo "  Save to:    $CHECKPOINT_DIR/continuous_training_latest.pt"
echo "  Log:        $LOG_DIR/training.log"
echo "  History:    $LOG_DIR/training_history.txt"
echo ""
echo "How it works:"
echo "  • Each iteration runs the full trace (19K requests)"
echo "  • Model is loaded → trained → saved after each iteration"
echo "  • Next iteration continues from updated model"
echo "  • Training continues until you press Ctrl+C"
echo ""
echo "Training will run continuously. Monitor with:"
echo "  tail -f $LOG_DIR/training.log         # Live training progress"
echo "  cat $LOG_DIR/training_history.txt      # Performance over time"
echo ""
echo "Press Ctrl+C to stop training and save final checkpoint"
echo ""
echo "================================================================================"
echo ""

# Clear old training log
rm -f "$LOG_DIR/training.log"
rm -f "$LOG_DIR/training_history.txt"

# Training loop - runs until interrupted
ITERATION=1
TOTAL_TRAINING_STEPS=0
BEST_E2E=999999.0

echo "Training Progress History:" | tee "$LOG_DIR/training_history.txt"
echo "======================================================" | tee -a "$LOG_DIR/training_history.txt"

while true; do
    echo "================================================================================"
    echo "Training Iteration $ITERATION (Total steps: $TOTAL_TRAINING_STEPS)"
    echo "================================================================================"
    echo ""

    # Create output directory for this iteration
    OUTPUT_DIR="results/rl_continuous_training_iter${ITERATION}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    # Build command
    CMD=(
        python -m vidur.main
        --replica_config_device a100
        --replica_config_model_name meta-llama/Llama-2-7b-hf
        --cluster_config_num_replicas 4
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
        --trace_request_generator_config_trace_file "$TRACE"
        --no-cache_config_enable_prefix_caching
        --no-metrics_config_write_json_trace
        --no-metrics_config_store_operation_metrics
        --no-metrics_config_store_batch_metrics
        --no-metrics_config_enable_chrome_trace
        --metrics_config_output_dir "$OUTPUT_DIR"
    )

    # Add checkpoint path for saving
    CMD+=(--r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_DIR/continuous_training_latest.pt")

    # If this is not the first iteration or we're loading a model, ensure checkpoint exists
    if [ $ITERATION -gt 1 ]; then
        # After first iteration, the checkpoint should exist
        if [ ! -f "$CHECKPOINT_DIR/continuous_training_latest.pt" ]; then
            echo "Warning: Expected checkpoint not found, will start fresh"
        fi
    elif [ -n "$LOAD_MODEL_PATH" ]; then
        # First iteration with user-provided model - copy it to our checkpoint location
        cp "$LOAD_MODEL_PATH" "$CHECKPOINT_DIR/continuous_training_latest.pt"
        echo "Copied initial model to $CHECKPOINT_DIR/continuous_training_latest.pt"
    fi

    # Run training iteration
    echo "Starting training iteration $ITERATION..."
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

    # Extract final training step count from log
    FINAL_STEP=$(grep -oP 'Steps: \K\d+' "$OUTPUT_DIR/simulation.log" | tail -1)
    if [ -n "$FINAL_STEP" ]; then
        TOTAL_TRAINING_STEPS=$((TOTAL_TRAINING_STEPS + FINAL_STEP))
    fi

    # Extract mean E2E from metrics
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
        echo "  Model checkpoint: $CHECKPOINT_DIR/continuous_training_latest.pt"
        echo "================================================================================"
        echo ""
        
        # Save to history file
        echo "Iter $ITERATION | Steps: $TOTAL_TRAINING_STEPS | E2E: ${MEAN_E2E}s" >> "$LOG_DIR/training_history.txt"
    fi

    ITERATION=$((ITERATION + 1))

    # Small delay before next iteration
    sleep 2
done
