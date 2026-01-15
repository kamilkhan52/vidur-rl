#!/bin/bash
#
# Long RL Training Run for Learning Analysis
#
# This script runs a long training session on the longest trace (arxiv)
# to analyze whether the RL agent learns anything over time.
#
# Expected runtime: ~1 hour
# Output: Training logs + metrics for analysis
#
# Usage:
#   ./scripts/train_rl_long.sh                           # Train from scratch
#   ./scripts/train_rl_long.sh rl_models/model_latest.pt # Continue from checkpoint

set -e

echo "================================================================================"
echo "  Long RL Training Run - Learning Analysis"
echo "================================================================================"
echo ""

# Configuration
TRACE="data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"
OUTPUT_DIR="results/rl_long_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/rl_training_logs"
CHECKPOINT_DIR="rl_models"

# Optional: Load existing model to continue training
LOAD_MODEL_PATH="${1:-}"  # First argument, empty if not provided

# Check trace exists
if [ ! -f "$TRACE" ]; then
    echo "Error: Trace file not found: $TRACE"
    echo "Run: python scripts/fix_trace_formats.py --check-all"
    exit 1
fi

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
echo "  Requests:   $(wc -l < "$TRACE")"
echo "  Output:     $OUTPUT_DIR"
echo "  Logs:       $LOG_DIR/training.log"
echo "  Save to:    $CHECKPOINT_DIR/long_training.pt"
echo ""
echo "================================================================================"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Clear old training log
rm -f "$LOG_DIR/training.log"

echo "Starting training... (this will take ~1 hour)"
echo "Monitor progress: tail -f $LOG_DIR/training.log"
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
  --r_l_cache_aware_global_scheduler_config_learning_rate 0.00003
  --r_l_cache_aware_global_scheduler_config_batch_size 64
  --r_l_cache_aware_global_scheduler_config_update_frequency 100
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
CMD+=(--r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_DIR/long_training.pt")

# If loading from existing model, copy it to the checkpoint path first
# (since model_checkpoint_path is used for both loading and saving)
if [ -n "$LOAD_MODEL_PATH" ]; then
  echo "Copying initial model to checkpoint location..."
  cp "$LOAD_MODEL_PATH" "$CHECKPOINT_DIR/long_training.pt"
fi

# Run training
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

echo ""
echo "================================================================================"
echo "  Training Complete!"
echo "================================================================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""

# Find the actual output directory (has timestamp)
ACTUAL_DIR=$(find "$OUTPUT_DIR" -type d -name "2026-*" | head -n 1)

if [ -z "$ACTUAL_DIR" ]; then
    echo "Warning: Could not find output directory"
    ACTUAL_DIR="$OUTPUT_DIR"
fi

echo "Analyzing training progress..."
python scripts/analyze_training_progress.py "$ACTUAL_DIR" \
  --log "$LOG_DIR/training.log" \
  --output "$OUTPUT_DIR/analysis"

echo ""
echo "================================================================================"
echo "  Analysis Complete!"
echo "================================================================================"
echo ""
echo "View results:"
echo "  • Training plots:   $OUTPUT_DIR/analysis/training_progress.png"
echo "  • Latency plots:    $OUTPUT_DIR/analysis/latency_over_time.png"
echo "  • Training log:     $LOG_DIR/training.log"
echo "  • Simulation log:   $OUTPUT_DIR/simulation.log"
echo "  • Metrics CSV:      $ACTUAL_DIR/request_metrics.csv"
echo ""
echo "Check learning:"
echo "  python scripts/analyze_training_progress.py $ACTUAL_DIR --log $LOG_DIR/training.log"
echo ""
echo "================================================================================"
