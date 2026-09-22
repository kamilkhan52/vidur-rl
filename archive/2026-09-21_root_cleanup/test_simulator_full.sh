#!/bin/bash
#
# Test the simulator fix with FULL trace (all 3993 requests)
#

set -e

TEST_DIR="/tmp/test_simulator_full"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "=========================================="
echo "Testing Simulator Fix - FULL TRACE"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Scheduler: LOP (batch-end)"
echo "  Model: Llama-2-7b"
echo "  Replicas: 16"
echo "  Scale: 1.5 (low request rate)"
echo "  Trace: Synthetic (ALL 3993 requests)"
echo ""

cd /home/kamil/vidur-rl-working

# Activate venv
source .venv/bin/activate

# Run FULL simulation at Scale 1.5 with ALL requests
echo "Running FULL simulation (this will take several minutes)..."
echo "Expected duration: ~2660 seconds of simulated time"
echo ""

timeout 600 python3 -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name meta-llama/Llama-2-7b-hf \
    --replica_config_tensor_parallel_size 1 \
    --cluster_config_num_replicas 16 \
    --replica_scheduler_config_type vllm_v1 \
    --global_scheduler_config_type lop_batch_end \
    --trace_request_generator_config_time_scale_factor 1.5 \
    --trace_request_generator_config_trace_file "data/processed_traces/mooncake_synthetic_trace.csv" \
    --metrics_config_output_dir "$TEST_DIR" \
    --no-metrics_config_write_json_trace \
    --no-metrics_config_enable_chrome_trace \
    --request_generator_config_type trace \
    --cache_config_enable_prefix_caching \
    --log_level info 2>&1 | tee "${TEST_DIR}/test_full.log"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Results - FULL TRACE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    # Find the output directory (timestamped)
    OUTPUT_DIR=$(find "$TEST_DIR" -type d -name "2026-*" | head -1)
    
    if [ -f "${OUTPUT_DIR}/request_metrics.csv" ]; then
        NUM_REQUESTS=$(wc -l < "${OUTPUT_DIR}/request_metrics.csv")
        NUM_REQUESTS=$((NUM_REQUESTS - 1))  # Subtract header
        
        echo "✅ SUCCESS: Simulation completed"
        echo "   Requests processed: $NUM_REQUESTS / 3993"
        echo ""
        
        if [ $NUM_REQUESTS -eq 3993 ]; then
            echo "✅✅✅ PERFECT: All 3993 requests completed!"
        else
            echo "⚠️  WARNING: Only $NUM_REQUESTS / 3993 requests completed"
        fi
        
        # Get final simulated time
        FINAL_TIME=$(grep "Simulation ended at:" "${TEST_DIR}/test_full.log" | awk '{print $5}' | sed 's/s$//')
        echo "   Simulated time: ${FINAL_TIME}s"
        
        # Check for warnings
        echo ""
        echo "Checking for issues..."
        
        if grep -q "pending requests in scheduler" "${TEST_DIR}/test_full.log"; then
            echo "⚠️  Found 'pending requests' warning"
            grep "pending requests" "${TEST_DIR}/test_full.log"
        fi
        
        if grep -q "Breaking to avoid infinite loop" "${TEST_DIR}/test_full.log"; then
            echo "⚠️  Infinite loop break triggered"
        fi
        
        if grep -q "Injecting GlobalScheduleEvent" "${TEST_DIR}/test_full.log"; then
            INJECT_COUNT=$(grep -c "Injecting GlobalScheduleEvent" "${TEST_DIR}/test_full.log")
            echo "ℹ️  Manual GlobalScheduleEvent injections: $INJECT_COUNT"
        fi
        
        echo ""
        if [ $NUM_REQUESTS -eq 3993 ]; then
            echo "✅ VERIFICATION PASSED: Fix works with full trace!"
            exit 0
        else
            echo "❌ VERIFICATION FAILED: Not all requests completed"
            exit 1
        fi
    else
        echo "❌ FAILURE: No metrics file generated"
        echo "   Output dir: $OUTPUT_DIR"
        ls -la "$OUTPUT_DIR" 2>/dev/null || echo "   Directory not found"
        exit 1
    fi
elif [ $EXIT_CODE -eq 124 ]; then
    echo "❌ FAILURE: Simulation timed out (600 seconds)"
    echo "   This likely indicates the stall bug is still present"
    exit 1
else
    echo "❌ FAILURE: Simulation exited with code $EXIT_CODE"
    tail -50 "${TEST_DIR}/test_full.log"
    exit 1
fi
