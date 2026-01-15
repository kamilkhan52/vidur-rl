#!/bin/bash
#
# Quick Evaluation Script
#
# Usage:
#   ./scripts/quick_eval.sh                    # Run default configuration
#   ./scripts/quick_eval.sh custom             # Run custom configuration (edit script first)
#
# Edit the variables below to customize your evaluation

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Edit these variables to customize your evaluation
# ============================================================================

# Schedulers to evaluate (space-separated)
# Options: random round_robin lor lop_uncached tolerant_sticky rl_cache_aware
SCHEDULERS="round_robin random lor lop_uncached"

# Workloads to test (space-separated)
# Options: splitwise_conv arxiv splitwise_code
WORKLOADS="splitwise_conv"

# Number of runs per configuration (for statistical significance)
NUM_RUNS=1

# Output directory (will be created if doesn't exist)
OUTPUT_DIR="results/eval_$(date +%Y%m%d_%H%M%S)"

# Quick mode (disables expensive metrics for faster evaluation)
QUICK_MODE="--quick"

# ============================================================================
# ADVANCED CONFIGURATION (optional)
# ============================================================================

# If evaluating RL scheduler with trained model, uncomment and set path:
# RL_CHECKPOINT="rl_models/checkpoint_latest.pt"

# Custom Python executable (if not using default)
# PYTHON_CMD="/path/to/python"
PYTHON_CMD="python"

# ============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================

echo "================================================================================"
echo "  Vidur Scheduler Evaluation"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Schedulers:  $SCHEDULERS"
echo "  Workloads:   $WORKLOADS"
echo "  Runs:        $NUM_RUNS"
echo "  Output:      $OUTPUT_DIR"
echo "  Quick mode:  $([ -n "$QUICK_MODE" ] && echo 'Enabled' || echo 'Disabled')"
echo ""
echo "================================================================================"
echo ""

# Build command
CMD="$PYTHON_CMD scripts/evaluate_schedulers.py"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --schedulers $SCHEDULERS"
CMD="$CMD --workloads $WORKLOADS"
CMD="$CMD --num_runs $NUM_RUNS"

if [ -n "$QUICK_MODE" ]; then
    CMD="$CMD $QUICK_MODE"
fi

# Show command
echo "Running: $CMD"
echo ""

# Run evaluation
eval $CMD

# Show results
echo ""
echo "================================================================================"
echo "  Evaluation Complete!"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  • Plots:        ls $OUTPUT_DIR/plots/"
echo "  • Summary CSV:  cat $OUTPUT_DIR/results.csv"
echo "  • Full log:     cat $OUTPUT_DIR/evaluation.log"
echo ""

# Open results directory if possible
if command -v xdg-open &> /dev/null; then
    echo "Opening results directory..."
    xdg-open "$OUTPUT_DIR/plots/" 2>/dev/null || true
elif command -v open &> /dev/null; then
    echo "Opening results directory..."
    open "$OUTPUT_DIR/plots/" 2>/dev/null || true
fi

echo "================================================================================"
