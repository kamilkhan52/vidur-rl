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
#   MAX_ITERS=                 # Optional max training iterations (default: infinite)
#   CHECKPOINT_TAG=            # Optional suffix for checkpoint/log namespace
#   REWARD_TYPE=e2e_time|ttft  # Reward signal (default: e2e_time)
#   LOOKAHEAD_K=5              # State lookahead depth (default: 5)
#   HIDDEN_DIMS=1024,1024,512  # RL network architecture override
#   LEARNING_RATE=0.0001       # Optimizer LR override
#   BATCH_SIZE=128             # Batch size override
#   UPDATE_FREQUENCY=50        # Train every N completed requests
#   REPLAY_BUFFER_SIZE=10000   # Replay capacity override
#   LOG_TRANSITIONS=0          # Set to 1 to log (state, action, reward) to CSV
#   TRACE_NAME=conversation_v2 # conversation|conversation_v2|toolagent|synthetic
#   TRACE_MAX_TOKENS=          # Optional override for max tokens (KV cache cap)
#   MAX_TOKENS_CAP=122752      # Default KV cache cap for auto max token detection
#   MODEL_NAME=meta-llama/Llama-2-7b-hf  # Model name override
#   LOG_BASE_DIR=./tmp/mooncake_training_logs # Base directory for run logs
#   RESULTS_BASE_DIR=./results              # Base directory for iteration outputs
#   CHECKPOINT_BASE_DIR=./rl_models         # Base directory for checkpoints

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
PYTHON_BIN="$(command -v python3)"
if [ -x "$VENV_PYTHON" ]; then
    PYTHON_BIN="$VENV_PYTHON"
fi

# Epsilon configuration
EPSILON_START="${EPSILON_START:-1.0}"
EPSILON_END="${EPSILON_END:-0.005}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-5000}"
MAX_ITERS="${MAX_ITERS:-}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-}"
REWARD_TYPE="${REWARD_TYPE:-e2e_time}"
LOOKAHEAD_K="${LOOKAHEAD_K:-5}"
HIDDEN_DIMS="${HIDDEN_DIMS:-1024,1024,512}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UPDATE_FREQUENCY="${UPDATE_FREQUENCY:-50}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-10000}"
LOG_BASE_DIR="${LOG_BASE_DIR:-./tmp/mooncake_training_logs}"
RESULTS_BASE_DIR="${RESULTS_BASE_DIR:-./results}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./rl_models}"

# Transition logging (set LOG_TRANSITIONS=1 to enable)
LOG_TRANSITIONS="${LOG_TRANSITIONS:-0}"

# Model configuration
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
NUM_REPLICAS="${NUM_REPLICAS:-4}"
TP_SIZE="${TP_SIZE:-1}"

# Trace selection
TRACE_NAME="${TRACE_NAME:-conversation}"  # canonical v1; v2 exists only on the Mac
# KV-capacity cap for the trace filter: 7b/TP1 = 122752 tokens, 70b/TP4 = 525920.
# (Before 2026-09-21 the cap was 122752 for every model, over-filtering 70b runs.)
if [ -z "${MAX_TOKENS_CAP:-}" ]; then
    case "$MODEL_NAME" in
        *70b*) MAX_TOKENS_CAP=525920 ;;
        *)     MAX_TOKENS_CAP=122752 ;;
    esac
fi

# Time scale factor for trace replay (to match 4 GPU capacity vs 128 GPU original)
# Original Mooncake trace: 3.4 QPS on 128 A800 GPUs (16 nodes × 8 GPUs)
# Scaling: 20x slower for just 4 GPUs
TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR:-20.0}"

echo "================================================================================"
echo "  Continuous Mooncake RL Training - Press Ctrl+C to stop"
echo "================================================================================"
echo ""
echo "Python: $PYTHON_BIN"
echo ""
echo "Epsilon Configuration:"
echo "  Start: $EPSILON_START"
echo "  End: $EPSILON_END"
echo "  Decay steps: $EPSILON_DECAY_STEPS"
if [ -n "$MAX_ITERS" ]; then
    echo "  Max iterations: $MAX_ITERS"
else
    echo "  Max iterations: infinite"
fi
    echo "  Reward type: $REWARD_TYPE"
    echo "  Lookahead K: $LOOKAHEAD_K"
    echo "  Hidden dims: $HIDDEN_DIMS"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Update frequency: $UPDATE_FREQUENCY"
    echo "  Replay buffer size: $REPLAY_BUFFER_SIZE"
    echo "  Log base dir: $LOG_BASE_DIR"
    echo "  Results base dir: $RESULTS_BASE_DIR"
    echo "  Checkpoint base dir: $CHECKPOINT_BASE_DIR"
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
    TRACE_MAX_TOKENS=$(TRACE_FILE="$TRACE_FILE" MAX_TOKENS_CAP="$MAX_TOKENS_CAP" "$PYTHON_BIN" - <<'PY' || echo "$MAX_TOKENS_CAP"
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
NAME_SUFFIX=""
if [ -n "$CHECKPOINT_TAG" ]; then
    NAME_SUFFIX="_${CHECKPOINT_TAG}"
fi

RUN_TAG="${TRACE_NAME}_scale${TIME_SCALE_FACTOR}${NAME_SUFFIX}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="${LOG_BASE_DIR%/}/${RUN_TAG}"
CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR%/}"
CHECKPOINT_NAME="mooncake_continuous_${TRACE_NAME}_scale${TIME_SCALE_FACTOR}${NAME_SUFFIX}"

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
echo "  Reward:      $REWARD_TYPE"
echo "  Lookahead:   K=$LOOKAHEAD_K"
echo "  Hidden dims: $HIDDEN_DIMS"
echo "  Checkpoint:  $CHECKPOINT_DIR/${CHECKPOINT_NAME}_latest.pt"
echo "  Log:         $LOG_DIR/training.log"
echo "  History:     $LOG_DIR/training_history.txt"
echo "  Summary CSV: $LOG_DIR/iteration_summary.csv"
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
rm -f "$LOG_DIR/iteration_summary.csv"

# Training loop
ITERATION=1
TOTAL_TRAINING_STEPS=0
BEST_E2E=999999.0

echo "Training Progress History:" | tee "$LOG_DIR/training_history.txt"
echo "======================================================" | tee -a "$LOG_DIR/training_history.txt"
echo "iteration,total_training_steps,updates_this_iter,final_epsilon,avg_loss,avg_reward,avg_current_q,avg_target_q,avg_td_abs,avg_rand_frac,avg_q_gap,completed_requests,mean_e2e,median_e2e,p95_e2e,p99_e2e,load_imbalance,output_dir" > "$LOG_DIR/iteration_summary.csv"

while true; do
    if [ -n "$MAX_ITERS" ] && [ "$ITERATION" -gt "$MAX_ITERS" ]; then
        echo "Reached MAX_ITERS=$MAX_ITERS, stopping."
        break
    fi

    echo "================================================================================"
    echo "Iteration $ITERATION (Total steps: $TOTAL_TRAINING_STEPS)"
    echo "================================================================================"
    echo ""

    # Create output directory
    OUTPUT_DIR="${RESULTS_BASE_DIR%/}/mooncake_continuous_iter${ITERATION}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    # Build command
    CMD=(
        "$PYTHON_BIN" -m vidur.main
        --replica_config_device a100
        --replica_config_model_name "$MODEL_NAME"
        --replica_config_tensor_parallel_size "$TP_SIZE"
        --cluster_config_num_replicas "$NUM_REPLICAS"
        --replica_scheduler_config_type vllm_v1
        --global_scheduler_config_type rl_cache_aware
        --r_l_cache_aware_global_scheduler_config_mode training
        --r_l_cache_aware_global_scheduler_config_epsilon_start "$EPSILON_START"
        --r_l_cache_aware_global_scheduler_config_epsilon_end "$EPSILON_END"
        --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps "$EPSILON_DECAY_STEPS"
        --r_l_cache_aware_global_scheduler_config_learning_rate "$LEARNING_RATE"
        --r_l_cache_aware_global_scheduler_config_batch_size "$BATCH_SIZE"
        --r_l_cache_aware_global_scheduler_config_update_frequency "$UPDATE_FREQUENCY"
        --r_l_cache_aware_global_scheduler_config_replay_buffer_size "$REPLAY_BUFFER_SIZE"
        --r_l_cache_aware_global_scheduler_config_reward_type "$REWARD_TYPE"
        --r_l_cache_aware_global_scheduler_config_lookahead_k "$LOOKAHEAD_K"
        --r_l_cache_aware_global_scheduler_config_hidden_dims "$HIDDEN_DIMS"
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
    else
        FINAL_STEP=0
    fi

    FINAL_EPSILON=$(grep -oP 'Epsilon: \K[0-9.]+' "$OUTPUT_DIR/simulation.log" | tail -1)
    AVG_LOSS=$(grep -oP 'Loss: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_REWARD=$(grep -oP 'Reward: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_CURRENT_Q=$(grep -oP 'CurrQ: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_TARGET_Q=$(grep -oP 'TgtQ: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_TD_ABS=$(grep -oP 'TDAbs: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_RAND_FRAC=$(grep -oP 'RandFrac: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')
    AVG_Q_GAP=$(grep -oP 'QGap: \K[-0-9.]+' "$OUTPUT_DIR/simulation.log" | awk '{sum+=$1; n+=1} END {if (n>0) printf "%.6f", sum/n; else print ""}')

    # Extract metrics
    METRICS_FILE=$(find "$OUTPUT_DIR" -name "request_metrics.csv" | head -1)
    if [ -f "$METRICS_FILE" ]; then
        # Calculate latency summary
        read -r COMPLETED_REQS MEAN_E2E MEDIAN_E2E P95_E2E P99_E2E <<< "$("$PYTHON_BIN" - <<PY
import pandas as pd
df = pd.read_csv('$METRICS_FILE')
e2e = df['request_e2e_time']
print(len(df), f'{e2e.mean():.3f}', f'{e2e.median():.3f}', f'{e2e.quantile(0.95):.3f}', f'{e2e.quantile(0.99):.3f}')
PY
)"

        # Calculate load distribution
        LOAD_INFO=$("$PYTHON_BIN" -c "
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
        LOAD_DIST=$(printf "%s\n" "$LOAD_INFO" | sed '$d')
        LOAD_IMBALANCE=$(printf "%s\n" "$LOAD_INFO" | tail -1 | awk '{print $3}')

        echo ""
        echo "================================================================================"
        echo "Iteration $ITERATION Complete"
        echo "  Total training steps: $TOTAL_TRAINING_STEPS"
        echo "  Updates this iteration: $FINAL_STEP"
        echo "  Final epsilon: ${FINAL_EPSILON:-N/A}"
        echo "  Avg loss: ${AVG_LOSS:-N/A}"
        echo "  Avg reward: ${AVG_REWARD:-N/A}"
        echo "  Avg current/target Q: ${AVG_CURRENT_Q:-N/A} / ${AVG_TARGET_Q:-N/A}"
        echo "  Avg TD abs: ${AVG_TD_ABS:-N/A}"
        echo "  Avg random fraction: ${AVG_RAND_FRAC:-N/A}"
        echo "  Avg Q gap: ${AVG_Q_GAP:-N/A}"
        echo "  Completed requests: ${COMPLETED_REQS}"
        echo "  Mean/Median E2E latency: ${MEAN_E2E}s / ${MEDIAN_E2E}s"
        echo "  P95/P99 E2E latency: ${P95_E2E}s / ${P99_E2E}s"
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
        echo "Iter $ITERATION | Steps: $TOTAL_TRAINING_STEPS | E2E: ${MEAN_E2E}s | Median: ${MEDIAN_E2E}s | P99: ${P99_E2E}s | Eps: ${FINAL_EPSILON:-N/A}" >> "$LOG_DIR/training_history.txt"
        echo "$ITERATION,$TOTAL_TRAINING_STEPS,$FINAL_STEP,${FINAL_EPSILON:-},${AVG_LOSS:-},${AVG_REWARD:-},${AVG_CURRENT_Q:-},${AVG_TARGET_Q:-},${AVG_TD_ABS:-},${AVG_RAND_FRAC:-},${AVG_Q_GAP:-},$COMPLETED_REQS,$MEAN_E2E,$MEDIAN_E2E,$P95_E2E,$P99_E2E,${LOAD_IMBALANCE:-},$OUTPUT_DIR" >> "$LOG_DIR/iteration_summary.csv"

        # Generate training progress plot
        PLOT_PATH="$LOG_DIR/training_progress.png"
        if "$PYTHON_BIN" scripts/plot_training_progress.py "$PLOT_PATH" 2>/dev/null; then
            echo ""
            echo "  📊 Training plot: $PLOT_PATH"
        fi
    else
        echo "Iteration $ITERATION failed: request_metrics.csv not found. Stopping."
        exit 1
    fi

    ITERATION=$((ITERATION + 1))

    # Small delay before next iteration
    sleep 2
done
