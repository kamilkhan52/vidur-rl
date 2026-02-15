#!/bin/bash
#
# Minimal test to reproduce checkpoint bug
# Run 2 iterations and verify checkpoint is updated between them
#

set -e

# Test parameters
MODEL_NAME="meta-llama/Llama-2-7b-hf"
TRACE_FILE="data/processed_traces/mooncake_synthetic_trace.csv"
CHECKPOINT_DIR="test_ckpt_bug"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/test_latest.pt"
SCALE=48.0

# Clean up any previous test
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "=========================================="
echo "Testing checkpoint save/load bug"
echo "=========================================="
echo ""

# Run iteration 1
echo "=== Iteration 1 ==="
python3 -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name "$MODEL_NAME" \
    --replica_config_tensor_parallel_size 1 \
    --cluster_config_num_replicas 8 \
    --replica_scheduler_config_type vllm_v1 \
    --global_scheduler_config_type rl_cache_aware \
    --r_l_cache_aware_global_scheduler_config_mode training \
    --r_l_cache_aware_global_scheduler_config_reward_type e2e_time \
    --r_l_cache_aware_global_scheduler_config_hidden_dims "512,512,256,128" \
    --r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_PATH" \
    --trace_request_generator_config_time_scale_factor "$SCALE" \
    --trace_request_generator_config_trace_file "$TRACE_FILE" \
    --metrics_config_output_dir "${CHECKPOINT_DIR}/iter1" \
    --no-metrics_config_write_json_trace \
    --request_generator_config_type trace \
    --cache_config_enable_prefix_caching \
    --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps 5000 \
    2>&1 | tee "${CHECKPOINT_DIR}/iter1.log"

echo ""
echo "Iteration 1 complete. Checking checkpoint..."
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "✓ Checkpoint exists: $CHECKPOINT_PATH"
    python3 -c "import torch; ckpt=torch.load('$CHECKPOINT_PATH', map_location='cpu'); print(f\"  State dict keys: {len(ckpt.keys())}\")"
else
    echo "✗ ERROR: Checkpoint not saved!"
    exit 1
fi

echo ""
echo "=== Iteration 2 ==="
python3 -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name "$MODEL_NAME" \
    --replica_config_tensor_parallel_size 1 \
    --cluster_config_num_replicas 8 \
    --replica_scheduler_config_type vllm_v1 \
    --global_scheduler_config_type rl_cache_aware \
    --r_l_cache_aware_global_scheduler_config_mode training \
    --r_l_cache_aware_global_scheduler_config_reward_type e2e_time \
    --r_l_cache_aware_global_scheduler_config_hidden_dims "512,512,256,128" \
    --r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_PATH" \
    --trace_request_generator_config_time_scale_factor "$SCALE" \
    --trace_request_generator_config_trace_file "$TRACE_FILE" \
    --metrics_config_output_dir "${CHECKPOINT_DIR}/iter2" \
    --no-metrics_config_write_json_trace \
    --request_generator_config_type trace \
    --cache_config_enable_prefix_caching \
    --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps 5000 \
    2>&1 | tee "${CHECKPOINT_DIR}/iter2.log"

echo ""
echo "=========================================="
echo "Analyzing results..."
echo "=========================================="
echo ""

# Check if "Resuming from update" changed between iterations
UPDATE1=$(grep "Resuming from update" "${CHECKPOINT_DIR}/iter1.log" | head -1)
UPDATE2=$(grep "Resuming from update" "${CHECKPOINT_DIR}/iter2.log" | head -1)

echo "Iteration 1: $UPDATE1"
echo "Iteration 2: $UPDATE2"
echo ""

if [ "$UPDATE1" = "$UPDATE2" ]; then
    echo "❌ BUG CONFIRMED: Both iterations resumed from the same update number!"
    echo "   The checkpoint was NOT updated between iterations."
else
    echo "✓ SUCCESS: Iterations resumed from different update numbers."
    echo "   The checkpoint WAS updated correctly."
fi

# Check for "Saving final checkpoint" messages
echo ""
echo "Checking for 'Saving final checkpoint' messages..."
SAVE1=$(grep -c "Saving final checkpoint" "${CHECKPOINT_DIR}/iter1.log" || echo "0")
SAVE2=$(grep -c "Saving final checkpoint" "${CHECKPOINT_DIR}/iter2.log" || echo "0")

echo "Iteration 1: $SAVE1 saves"
echo "Iteration 2: $SAVE2 saves"

if [ "$SAVE1" -eq "0" ] || [ "$SAVE2" -eq "0" ]; then
    echo "❌ atexit handler NOT firing! Final checkpoint not being saved."
else
    echo "✓ atexit handler IS firing."
fi
