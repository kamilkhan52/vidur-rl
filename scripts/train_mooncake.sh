#!/bin/bash
#
# Mooncake RL Training with Prefix Caching
#
# This script trains the RL model on Mooncake trace with prefix caching ENABLED.
# Uses 68-dimensional state (includes cache affinity features).
#
# Usage:
#   ./scripts/train_mooncake.sh [checkpoint_path]
#
# Examples:
#   ./scripts/train_mooncake.sh                                    # Train from scratch
#   ./scripts/train_mooncake.sh rl_models/mooncake_latest.pt      # Continue training
#
# Environment variables:
#   EPSILON_START=1.0          # Initial epsilon (default: 1.0)
#   EPSILON_END=0.05           # Final epsilon (default: 0.05)
#   EPSILON_DECAY_STEPS=2000   # Decay over N steps (default: 2000)

set -e

# Epsilon configuration
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.05}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-2000}"

# Time scale factor for trace replay (to match 4 GPU capacity vs 128 GPU original)
# Original Mooncake trace: 3.4 QPS on 128 A800 GPUs (16 nodes × 8 GPUs)
# Scaling: 16x slower → 0.21 QPS for 4 A100 GPUs
TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR:-16.0}"

echo "================================================================================"
echo "  Mooncake RL Training - Cache-Aware (68-dim state)"
echo "================================================================================"
echo ""
echo "Epsilon Configuration:"
echo "  Start: $EPSILON_START"
echo "  End: $EPSILON_END"
echo "  Decay steps: $EPSILON_DECAY_STEPS"
echo ""
echo "Trace Scaling:"
echo "  Time scale factor: ${TIME_SCALE_FACTOR}x (slows arrival rate)"
echo "  Original: 3.4 QPS → Scaled: $(python3 -c "print(f'{3.4 / ${TIME_SCALE_FACTOR}:.2f}')") QPS"
echo ""

# Configuration
TRACE="data/processed_traces/mooncake_conversation_trace.csv"
OUTPUT_DIR="results/mooncake_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/rl_training_logs"
CHECKPOINT_DIR="rl_models"
CHECKPOINT_NAME="mooncake_training"

# Optional: Load existing model
LOAD_MODEL_PATH="${1:-}"

# Verify trace exists
if [ ! -f "$TRACE" ]; then
    echo "Error: Mooncake trace not found: $TRACE"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
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
echo "  Trace:       $TRACE"
echo "  Requests:    12,031 (with multi-turn sessions)"
echo "  Caching:     ENABLED (cache-aware features)"
echo "  State dims:  68 (includes 20 cache features)"
echo "  Output:      $OUTPUT_DIR"
echo "  Checkpoint:  $CHECKPOINT_DIR/${CHECKPOINT_NAME}.pt"
echo "  Log:         $LOG_DIR/training.log"
echo ""
echo "================================================================================"
echo ""

# Clear old log
rm -f "$LOG_DIR/training.log"

echo "Starting training..."
echo "Monitor: tail -f $LOG_DIR/training.log"
echo ""

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
    --trace_request_generator_config_time_scale_factor "$TIME_SCALE_FACTOR"
    --trace_request_generator_config_max_tokens 122752
    --cache_config_enable_prefix_caching
    --no-metrics_config_write_json_trace
    --no-metrics_config_store_operation_metrics
    --no-metrics_config_store_batch_metrics
    --no-metrics_config_enable_chrome_trace
    --metrics_config_output_dir "$OUTPUT_DIR"
)

# Add checkpoint path for saving
CMD+=(--r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_DIR/${CHECKPOINT_NAME}.pt")

# If loading checkpoint, copy it first
if [ -n "$LOAD_MODEL_PATH" ]; then
    cp "$LOAD_MODEL_PATH" "$CHECKPOINT_DIR/${CHECKPOINT_NAME}.pt"
    echo "Copied checkpoint to training location"
fi

# Run training
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

echo ""
echo "================================================================================"
echo "  Training Complete!"
echo "================================================================================"
echo ""
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT_DIR/${CHECKPOINT_NAME}.pt"
echo ""

# Find metrics file
METRICS_FILE=$(find "$OUTPUT_DIR" -name "request_metrics.csv" | head -1)

if [ -f "$METRICS_FILE" ]; then
    echo "Analyzing results..."
    python3 -c "
import pandas as pd
df = pd.read_csv('$METRICS_FILE')

print('Performance Metrics:')
print(f'  Mean E2E: {df[\"request_e2e_time\"].mean():.3f}s')
print(f'  Median E2E: {df[\"request_e2e_time\"].median():.3f}s')
print(f'  P99 E2E: {df[\"request_e2e_time\"].quantile(0.99):.3f}s')
print()

print('Load Distribution:')
for replica in sorted(df['replica'].unique()):
    count = (df['replica'] == replica).sum()
    pct = 100.0 * count / len(df)
    print(f'  Replica {replica}: {count:5d} ({pct:5.1f}%)')
"
    echo ""
fi

echo "View training progress:"
echo "  cat $LOG_DIR/training.log | grep 'Steps:'"
echo ""
echo "Checkpoint ready for evaluation:"
echo "  python scripts/evaluate_schedulers.py \\"
echo "    --schedulers rl_cache_aware round_robin lor lop_uncached \\"
echo "    --workloads mooncake \\"
echo "    --rl_model_path $CHECKPOINT_DIR/${CHECKPOINT_NAME}.pt"
echo ""
