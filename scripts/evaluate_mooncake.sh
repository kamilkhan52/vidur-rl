#!/bin/bash
#
# Evaluate Schedulers on Mooncake with Prefix Caching
#
# Compares RL scheduler against baselines on Mooncake trace
# with prefix caching ENABLED for fair comparison.
#
# Usage:
#   ./scripts/evaluate_mooncake.sh [rl_model_path] [num_runs]
#
# Examples:
#   ./scripts/evaluate_mooncake.sh rl_models/mooncake_training.pt 3
#   ./scripts/evaluate_mooncake.sh                                  # Use default model, 1 run

set -e

# Configuration
RL_MODEL_PATH="${1:-rl_models/mooncake_continuous_latest.pt}"
NUM_RUNS="${2:-1}"
OUTPUT_DIR="results/mooncake_evaluation_$(date +%Y%m%d_%H%M%S)"

# Time scale factor for trace replay (to match 4 GPU capacity vs 128 GPU original)
# Original Mooncake trace: 3.4 QPS on 128 A800 GPUs (16 nodes × 8 GPUs)
# Scaling: 16x slower → 0.21 QPS for 4 A100 GPUs
TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR:-16.0}"

echo "================================================================================"
echo "  Mooncake Scheduler Evaluation - Cache-Aware"
echo "================================================================================"
echo ""

# Check if RL model exists
if [ ! -f "$RL_MODEL_PATH" ]; then
    echo "Warning: RL model not found: $RL_MODEL_PATH"
    echo "Will evaluate only baseline schedulers"
    RL_MODEL_PATH=""
fi

echo "Configuration:"
echo "  Trace:       Mooncake (12,031 requests, multi-turn sessions)"
echo "  Caching:     ENABLED (cache-aware comparison)"
echo "  Trace scaling: ${TIME_SCALE_FACTOR}x (scaled arrival rate: $(python3 -c "print(f'{3.4 / ${TIME_SCALE_FACTOR}:.2f}')") QPS)"
echo "  RL model:    ${RL_MODEL_PATH:-None - evaluating baselines only}"
echo "  Runs:        $NUM_RUNS per scheduler"
echo "  Output:      $OUTPUT_DIR"
echo ""
echo "NOTE: Currently evaluate_schedulers.py does not support time_scale_factor."
echo "      For baseline testing with scaled trace, use direct vidur.main commands."
echo ""
echo "Schedulers to evaluate:"
echo "  • random          - Random assignment"
echo "  • round_robin     - Cyclic assignment"
echo "  • lor             - Least Outstanding Requests"
echo "  • lop_uncached    - Cache-aware LOP (uncached tokens)"
echo "  • tolerant_sticky - Session-sticky with tolerance"

if [ -n "$RL_MODEL_PATH" ]; then
    echo "  • rl_cache_aware  - RL with cache-aware state (68 dims)"
fi

echo ""
echo "================================================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Define schedulers
BASELINE_SCHEDULERS="random round_robin lor lop_uncached tolerant_sticky"

if [ -n "$RL_MODEL_PATH" ]; then
    ALL_SCHEDULERS="$BASELINE_SCHEDULERS rl_cache_aware"
else
    ALL_SCHEDULERS="$BASELINE_SCHEDULERS"
fi

# Run evaluation
echo "Starting evaluation..."
echo ""

EVAL_CMD=(
    python scripts/evaluate_schedulers.py
    --schedulers $ALL_SCHEDULERS
    --workloads mooncake
    --num_runs "$NUM_RUNS"
    --output_dir "$OUTPUT_DIR"
)

# Add RL model path if available
if [ -n "$RL_MODEL_PATH" ]; then
    EVAL_CMD+=(--rl_model_path "$RL_MODEL_PATH")
fi

# Run evaluation
"${EVAL_CMD[@]}"

echo ""
echo "================================================================================"
echo "  Evaluation Complete!"
echo "================================================================================"
echo ""
echo "Results: $OUTPUT_DIR"
echo ""

# Display summary
if [ -f "$OUTPUT_DIR/results.csv" ]; then
    echo "Performance Summary:"
    echo "-------------------"
    python3 -c "
import pandas as pd

df = pd.read_csv('$OUTPUT_DIR/results.csv')

# Group by scheduler and compute mean metrics
summary = df.groupby('scheduler').agg({
    'mean_e2e': 'mean',
    'median_e2e': 'mean',
    'p99_e2e': 'mean'
}).round(3)

# Sort by mean_e2e (best first)
summary = summary.sort_values('mean_e2e')

print()
print(f'{'Scheduler':<20} {'Mean E2E':>12} {'Median E2E':>12} {'P99 E2E':>12}')
print('='*60)

for scheduler, row in summary.iterrows():
    print(f'{scheduler:<20} {row[\"mean_e2e\"]:>10.3f}s {row[\"median_e2e\"]:>10.3f}s {row[\"p99_e2e\"]:>10.3f}s')

print()

# Highlight RL performance
if 'rl_cache_aware' in summary.index:
    rl_rank = list(summary.index).index('rl_cache_aware') + 1
    total = len(summary)
    print(f'RL Ranking: {rl_rank}/{total}')

    best_scheduler = summary.index[0]
    best_e2e = summary.loc[best_scheduler, 'mean_e2e']
    rl_e2e = summary.loc['rl_cache_aware', 'mean_e2e']

    if rl_rank == 1:
        print('✨ RL is BEST performer!')
    elif rl_rank <= 3:
        ratio = rl_e2e / best_e2e
        print(f'⚠️  RL is {ratio:.2f}x slower than {best_scheduler}')
    else:
        ratio = rl_e2e / best_e2e
        print(f'❌ RL is {ratio:.2f}x slower than {best_scheduler}')
"
    echo ""
fi

echo "View plots:"
echo "  ls $OUTPUT_DIR/plots/*.png"
echo ""
echo "Detailed results:"
echo "  cat $OUTPUT_DIR/results.csv"
echo ""
