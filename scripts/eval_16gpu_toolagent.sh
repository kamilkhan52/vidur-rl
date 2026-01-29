#!/bin/bash
#
# Evaluate cache-aware schedulers on 16-GPU system
#
# Configuration:
#   - 16 replicas (16 GPUs total, TP=1, PP=1)
#   - Time scale: 8x (closer to original trace collection)
#   - Trace: toolagent
#
# Usage:
#   ./scripts/eval_16gpu_toolagent.sh [scheduler1] [scheduler2] ...
#   ./scripts/eval_16gpu_toolagent.sh random round_robin sticky_lor

set -e

# Configuration
NUM_REPLICAS=16
TIME_SCALE=3.0
TRACE_FILE="data/processed_traces/mooncake_toolagent_trace.csv"
PAPER_CACHE="59%"

# Default schedulers if none specified
if [ $# -eq 0 ]; then
    SCHEDULERS=(random round_robin lor lop_uncached sticky_lor)
else
    SCHEDULERS=("$@")
fi

echo "================================================================================"
echo "  16-GPU Cache-Aware Scheduler Evaluation"
echo "================================================================================"
echo ""
echo "  Configuration:"
echo "    Replicas:   $NUM_REPLICAS"
echo "    Time scale: ${TIME_SCALE}x"
echo "    Trace:      toolagent ($TRACE_FILE)"
echo "    Schedulers: ${SCHEDULERS[*]}"
echo "    Paper cache ratio: $PAPER_CACHE"
echo ""
echo "================================================================================"

# Verify trace exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Trace file not found: $TRACE_FILE"
    exit 1
fi

# Store results for final summary
declare -A RESULTS
declare -a LABELS

# Run each scheduler
for SCHEDULER in "${SCHEDULERS[@]}"; do
    echo ""
    echo "================================================================================"
    echo "  Running: $SCHEDULER"
    echo "================================================================================"
    echo ""

    # Output directory
    OUTPUT_DIR="results/16gpu_eval/toolagent_${SCHEDULER}_r${NUM_REPLICAS}_scale${TIME_SCALE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    # Build command
    CMD=(
        python -m vidur.main
        --replica_config_device a100
        --replica_config_model_name meta-llama/Llama-2-7b-hf
        --cluster_config_num_replicas "$NUM_REPLICAS"
        --replica_scheduler_config_type vllm_v1
        --global_scheduler_config_type "$SCHEDULER"
        --request_generator_config_type trace
        --trace_request_generator_config_trace_file "$TRACE_FILE"
        --trace_request_generator_config_time_scale_factor "$TIME_SCALE"
        --trace_request_generator_config_max_tokens 122752
        --cache_config_enable_prefix_caching
        --no-metrics_config_write_json_trace
        --no-metrics_config_store_operation_metrics
        --no-metrics_config_store_batch_metrics
        --no-metrics_config_enable_chrome_trace
        --metrics_config_output_dir "$OUTPUT_DIR"
    )

    # Run simulation
    echo "Command: ${CMD[*]}"
    echo ""
    "${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

    echo ""
    echo "  RESULTS for $SCHEDULER:"
    echo "  ----------------------------------------"

    # Find and analyze metrics
    METRICS_FILE=$(find "$OUTPUT_DIR" -name "request_metrics.csv" | head -1)

    if [ -f "$METRICS_FILE" ]; then
        RESULT=$(python3 -c "
import pandas as pd

df = pd.read_csv('$METRICS_FILE')

mean_e2e = df['request_e2e_time'].mean()
median_e2e = df['request_e2e_time'].median()
p99_e2e = df['request_e2e_time'].quantile(0.99)

print(f'  Completed: {len(df):,} requests')
print()
print('  E2E Latency:')
print(f'    Mean:   {mean_e2e:.2f}s')
print(f'    Median: {median_e2e:.2f}s')
print(f\"    P95:    {df['request_e2e_time'].quantile(0.95):.2f}s\")
print(f'    P99:    {p99_e2e:.2f}s')
print()
print('  Scheduling Delay:')
print(f\"    Mean:   {df['request_scheduling_delay'].mean():.2f}s\")
print(f\"    Max:    {df['request_scheduling_delay'].max():.2f}s\")
print()

# Cache hit rate (token-weighted and request-weighted)
cache_rate_token = 0
cache_rate_request = 0
if 'request_num_prefill_tokens_cached' in df.columns:
    cached = df['request_num_prefill_tokens_cached']
    total = df['request_num_prefill_tokens']
    cache_rate_token = 100 * cached.sum() / total.sum() if total.sum() > 0 else 0
    hit_ratios = cached / total
    cache_rate_request = 100 * hit_ratios.mean()
    pct_with_hits = 100 * (cached > 0).sum() / len(df)
    print(f'  Cache Hit Rate (token-weighted):   {cache_rate_token:.1f}%')
    print(f'  Cache Hit Rate (request-weighted): {cache_rate_request:.1f}%')
    print(f'  Requests with cache hits:          {pct_with_hits:.1f}%')
print()

# Load distribution
print('  Load Distribution:')
for replica in sorted(df['replica'].unique()):
    count = (df['replica'] == replica).sum()
    pct = 100 * count / len(df)
    print(f'    Replica {replica:2d}: {count:>5,} ({pct:5.1f}%)')

counts = df['replica'].value_counts()
imbalance = (counts.max() - counts.min()) / len(df)
print(f'    Imbalance: {imbalance:.4f}')

# Output summary line for final table
print(f'\nSUMMARY_LINE:{mean_e2e:.2f},{median_e2e:.2f},{p99_e2e:.2f},{cache_rate_token:.1f},{cache_rate_request:.1f},{imbalance:.4f}')
")
        echo "$RESULT" | grep -v "SUMMARY_LINE"
        SUMMARY=$(echo "$RESULT" | grep "SUMMARY_LINE" | cut -d: -f2)
        RESULTS["$SCHEDULER"]="$SUMMARY"
        LABELS+=("$SCHEDULER")
    else
        echo "  No metrics file found!"
        RESULTS["$SCHEDULER"]="N/A"
        LABELS+=("$SCHEDULER")
    fi

    echo ""
    echo "  Output: $OUTPUT_DIR"
done

# Final summary table
echo ""
echo "================================================================================"
echo "  FINAL SUMMARY - 16-GPU toolagent (${NUM_REPLICAS} replicas, ${TIME_SCALE}x scale)"
echo "================================================================================"
echo ""
printf "  %-20s %10s %10s %10s %12s %12s %10s\n" "Scheduler" "Mean E2E" "Median" "P99" "Cache(token)" "Cache(req)" "Imbalance"
echo "  ------------------------------------------------------------------------------------------"
for LABEL in "${LABELS[@]}"; do
    if [ "${RESULTS[$LABEL]}" != "N/A" ]; then
        IFS=',' read -r mean median p99 cache_token cache_req imbal <<< "${RESULTS[$LABEL]}"
        printf "  %-20s %10s %10s %10s %12s %12s %10s\n" "$LABEL" "${mean}s" "${median}s" "${p99}s" "${cache_token}%" "${cache_req}%" "$imbal"
    else
        printf "  %-20s %10s\n" "$LABEL" "FAILED"
    fi
done
echo ""
echo "================================================================================"
