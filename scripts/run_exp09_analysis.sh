#!/bin/bash
set -euo pipefail

# One-command postprocessing for EXP09 queue diagnostics:
# 1) collect/merge queue_diag_events.csv files
# 2) run EXP09 queue analysis on merged CSV
#
# Usage:
#   scripts/run_exp09_analysis.sh \
#     --input-dir /scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_pilot/runs \
#     --analysis-dir /scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_pilot/analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR=""
ANALYSIS_DIR=""
PYTHON_BIN=""
PATTERN="queue_diag_events.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --analysis-dir)
      ANALYSIS_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$ANALYSIS_DIR" ]]; then
  echo "ERROR: --input-dir and --analysis-dir are required"
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$ANALYSIS_DIR"
MERGED_CSV="${ANALYSIS_DIR}/exp09_queue_events_merged.csv"

echo "[EXP09] Input runs dir: $INPUT_DIR"
echo "[EXP09] Analysis dir:   $ANALYSIS_DIR"
echo "[EXP09] Python:         $PYTHON_BIN"
echo "[EXP09] Pattern:        $PATTERN"

echo "[EXP09] Step 1/2: Collect + merge queue diagnostics CSVs"
"$PYTHON_BIN" "${SCRIPT_DIR}/collect_exp09_queue_events.py" \
  --input-dir "$INPUT_DIR" \
  --pattern "$PATTERN" \
  --output-csv "$MERGED_CSV"

echo "[EXP09] Step 2/2: Analyze merged queue diagnostics"
"$PYTHON_BIN" "${SCRIPT_DIR}/analyze_exp09_queue_diagnostics.py" \
  --events-csv "$MERGED_CSV" \
  --output-dir "$ANALYSIS_DIR"

echo "[EXP09] Done"
echo "[EXP09] Merged CSV: $MERGED_CSV"
echo "[EXP09] Outputs:"
echo "  - ${ANALYSIS_DIR}/exp09_queue_events_summary.csv"
echo "  - ${ANALYSIS_DIR}/exp09_hol_metrics.csv"
echo "  - ${ANALYSIS_DIR}/exp09_lookahead_opportunity.csv"