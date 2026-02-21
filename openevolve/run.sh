#!/bin/bash
# Launch OpenEvolve RL formulation search.
#
# Usage:
#   export OPENAI_API_KEY="your-gemini-api-key"
#   bash openevolve/run.sh [--iterations N] [--resume]
#
# Options:
#   --iterations N   Number of evolution iterations (default: 100)
#   --resume         Resume from existing evolution_output if it exists

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
OPENEVOLVE_DIR="$REPO_ROOT/openevolve"
OUTPUT_DIR="$OPENEVOLVE_DIR/evolution_output"
ITERATIONS=100

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set."
    echo "  export OPENAI_API_KEY='your-gemini-api-key'"
    exit 1
fi

# Check venv
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    exit 1
fi

# Precompute baseline if not done yet
BASELINE_FILE="$OPENEVOLVE_DIR/baselines/sticky_lor_e2e.json"
if [ ! -f "$BASELINE_FILE" ]; then
    echo "=== Precomputing baselines ==="
    "$VENV_PYTHON" "$OPENEVOLVE_DIR/precompute_baseline.py"
fi

echo "=== Starting OpenEvolve: $ITERATIONS iterations ==="
echo "Output: $OUTPUT_DIR"
echo ""

cd "$REPO_ROOT"
"$VENV_PYTHON" - <<EOF
import openevolve, os, json
from pathlib import Path

result = openevolve.run_evolution(
    initial_program='openevolve/initial_program.py',
    evaluator='openevolve/evaluator.py',
    config='openevolve/config.yaml',
    iterations=$ITERATIONS,
    output_dir='$OUTPUT_DIR',
)

print()
print('=== Evolution complete ===')
print(f'Best score: {result.best_score:.4f}')
print(f'Best program: {result.best_program_path if hasattr(result, "best_program_path") else "$OUTPUT_DIR/best/best_program.py"}')

# Print best program
best_path = Path('$OUTPUT_DIR/best/best_program.py')
if best_path.exists():
    print()
    print('--- Best program ---')
    print(best_path.read_text())
EOF
