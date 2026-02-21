#!/bin/bash
# Launch OpenEvolve on Alpine (login node, inside a tmux session).
#
# Run from the repo root on Alpine:
#   export OPENAI_API_KEY="your-gemini-api-key"
#   tmux new-session -d -s openevolve \
#     "cd /projects/kamil@colostate.edu/vidur-rl && bash openevolve/run_alpine.sh --iterations 50"
#   tmux attach -t openevolve
#
# Options:
#   --iterations N      Number of evolution iterations (default: 50)
#   --output-dir DIR    Where to save results (default: SCRATCH/vidur-rl/openevolve/TIMESTAMP)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ITERATIONS=50
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT="/scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve/run_${TIMESTAMP}"
OUTPUT_DIR="$DEFAULT_OUTPUT"

while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "  export OPENAI_API_KEY='your-gemini-api-key'"
    exit 1
fi

# ── Conda setup ───────────────────────────────────────────────────────────────
CONDA_INIT="/curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_INIT" ]; then
    echo "ERROR: Conda not found at $CONDA_INIT"
    exit 1
fi
source "$CONDA_INIT"
conda activate vidur
CONDA_PYTHON="$(which python3)"
echo "Python: $CONDA_PYTHON"

# ── Install openevolve if missing ─────────────────────────────────────────────
if ! "$CONDA_PYTHON" -c "import openevolve" 2>/dev/null; then
    echo "Installing openevolve in conda vidur env..."
    pip install openevolve --quiet
fi

# ── Output directory on SCRATCH ───────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"

# Copy baseline if it exists locally in the repo (avoids re-running baseline)
BASELINE_SRC="$REPO_ROOT/openevolve/baselines/sticky_lor_e2e.json"
BASELINE_DST="$OUTPUT_DIR/baselines/sticky_lor_e2e.json"
if [ -f "$BASELINE_SRC" ] && [ ! -f "$BASELINE_DST" ]; then
    mkdir -p "$(dirname "$BASELINE_DST")"
    cp "$BASELINE_SRC" "$BASELINE_DST"
    echo "Copied baseline from repo: STICKY_LOR=$(python3 -c "import json; print(json.load(open('$BASELINE_DST'))['sticky_lor_e2e'])")"
fi

echo ""
echo "=== Starting OpenEvolve: $ITERATIONS iterations ==="
echo ""

cd "$REPO_ROOT"
"$CONDA_PYTHON" openevolve/run_evolution.py \
    --iterations "$ITERATIONS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Done. Results in: $OUTPUT_DIR ==="
echo "Plot: $OUTPUT_DIR/progress.png"
