#!/bin/bash
#
# Test the simulator fix for low-scale stalling
#

set -e

TEST_DIR="/tmp/test_simulator_fix"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "=========================================="
echo "Testing Simulator Fix at Low Scale"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Scheduler: LOP (batch-end)"
echo "  Model: Llama-2-7b"
echo "  Replicas: 16"
echo "  Scale: 1.5 (low request rate)"
echo "  Trace: Synthetic"
echo ""

cd /home/kamil/vidur-rl-working

# Activate venv
source .venv/bin/activate

# Run a SHORT simulation at Scale 1.5 with limited requests
echo "Running simulation (this should complete in ~30 seconds)..."
echo ""

timeout 120 python3 -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name meta-llama/Llama-2-7b-hf \
    --replica_config_tensor_parallel_size 1 \
    --cluster_config_num_replicas 16 \
    --replica_scheduler_config_type vllm_v1 \
    --global_scheduler_config_type lop_batch_end \
    --trace_request_generator_config_time_scale_factor 1.5 \
    --trace_request_generator_config_trace_file "data/processed_traces/mooncake_synthetic_trace.csv" \
    --trace_request_generator_config_num_requests 300 \
    --metrics_config_output_dir "$TEST_DIR" \
    --no-metrics_config_write_json_trace \
    --no-metrics_config_enable_chrome_trace \
    --request_generator_config_type trace \
    --cache_config_enable_prefix_caching \
    --log_level debug 2>&1 | tee "${TEST_DIR}/test.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Results"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    # Check if summary was written
    if [ -f "${TEST_DIR}/request_metrics.csv" ]; then
        NUM_REQUESTS=$(wc -l < "${TEST_DIR}/request_metrics.csv")
        NUM_REQUESTS=$((NUM_REQUESTS - 1))  # Subtract header
        
        echo "✅ SUCCESS: Simulation completed"
        echo "   Requests processed: $NUM_REQUESTS"
        echo ""
        echo "Checking for stall indicators in log..."
        
        # Check for warning about pending requests
        if grep -q "pending requests in scheduler" "${TEST_DIR}/test.log"; then
            echo "⚠️  WARNING: Log contains 'pending requests' message"
            echo "   The fix may not be working correctly"
            exit 1
        fi
        
        # Check for manual injection messages
        if grep -q "Injecting GlobalScheduleEvent" "${TEST_DIR}/test.log"; then
            echo "ℹ️  INFO: Manual GlobalScheduleEvent injection occurred"
            echo "   This is expected at low scales when the event queue drains"
        fi
        
        # Check for infinite loop breaking
        if grep -q "Breaking to avoid infinite loop" "${TEST_DIR}/test.log"; then
            echo "❌ FAILURE: Infinite loop detected!"
            echo "   The scheduler couldn't make progress"
            exit 1
        fi
        
        echo ""
        echo "✅ All checks passed! The fix is working correctly."
        exit 0
    else
        echo "❌ FAILURE: No metrics file generated"
        echo "   Simulation may have crashed"
        exit 1
    fi
elif [ $EXIT_CODE -eq 124 ]; then
    echo "❌ FAILURE: Simulation timed out (120 seconds)"
    echo "   This likely indicates the stall bug is still present"
    exit 1
else
    echo "❌ FAILURE: Simulation exited with code $EXIT_CODE"
    exit 1
fi
