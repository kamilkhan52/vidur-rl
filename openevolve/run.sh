#!/bin/bash
# Launch OpenEvolve RL formulation search (local / WSL).
#
# Usage:
#   export OPENAI_API_KEY="your-gemini-api-key"
#   bash openevolve/run.sh [--iterations N]

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
ITERATIONS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    exit 1
fi

cd "$REPO_ROOT"
"$VENV_PYTHON" openevolve/run_evolution.py --iterations "$ITERATIONS"
