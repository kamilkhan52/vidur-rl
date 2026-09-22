#!/bin/bash
#SBATCH --job-name=exp09_queue_diag_pilot
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_pilot/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_pilot/slurm/slurm-%A_%a.err
#SBATCH --array=1-144
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

PROJECTS_DIR="/projects/kamil@colostate.edu"
SCRATCH_DIR="/scratch/alpine/kamil@colostate.edu"
REPO_DIR="${PROJECTS_DIR}/vidur-rl"
MATRIX_FILE="${REPO_DIR}/scripts/exp09_queue_diag_pilot_matrix.tsv"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${SCRATCH_DIR}/vidur-rl/exp09_queue_diag_pilot/runs}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRATCH_DIR}/vidur-rl/exp09_queue_diag_pilot/slurm}"
mkdir -p "${OUTPUT_DIR_BASE}" "${SLURM_LOG_DIR}"

if [[ ! -f "$MATRIX_FILE" ]]; then
  echo "ERROR: matrix file missing: $MATRIX_FILE"
  exit 1
fi

NUM_ROWS=$(python - <<'PY'
import csv
from pathlib import Path
p=Path('/projects/kamil@colostate.edu/vidur-rl/scripts/exp09_queue_diag_pilot_matrix.tsv')
with p.open() as f:
    r=csv.reader(f, delimiter='\t')
    next(r, None)
    n=sum(1 for _ in r)
print(n)
PY
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID))
if [[ "$TASK_INDEX" -lt 1 || "$TASK_INDEX" -gt "$NUM_ROWS" ]]; then
  echo "ERROR: array task ${TASK_INDEX} out of range 1-${NUM_ROWS}"
  exit 1
fi

LINE=$(sed -n "$((TASK_INDEX+1))p" "$MATRIX_FILE")
IFS=$'\t' read -r SCHEDULER MODEL REPLICAS TRACE SCALE REPEAT_ID <<< "$LINE"

if [[ -z "${SCHEDULER:-}" || -z "${MODEL:-}" || -z "${REPLICAS:-}" || -z "${TRACE:-}" || -z "${SCALE:-}" || -z "${REPEAT_ID:-}" ]]; then
  echo "ERROR: failed to parse matrix line for task ${TASK_INDEX}: $LINE"
  exit 1
fi

echo "============================================================"
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Task: $SCHEDULER / $MODEL / $REPLICAS / $TRACE / Scale $SCALE / Repeat $REPEAT_ID"
echo "Output root: $OUTPUT_DIR_BASE"
echo "============================================================"

cd "$REPO_DIR"

export ENABLE_QUEUE_DIAGNOSTICS=1

# Repeat is encoded in matrix; run one realization per task.
NUM_RUNS=1 OUTPUT_DIR_BASE="$OUTPUT_DIR_BASE" ./scripts/run_baseline_sweep.sh \
  "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo "Done: $SCHEDULER $MODEL $REPLICAS $TRACE scale $SCALE repeat $REPEAT_ID"
