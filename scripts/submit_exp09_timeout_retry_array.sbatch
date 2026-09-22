#!/bin/bash
#SBATCH --job-name=exp09_timeout_retry
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_timeout_retry_20260426/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp09_queue_diag_timeout_retry_20260426/slurm/slurm-%A_%a.err
#SBATCH --array=5,77,113,114
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

PROJECTS_DIR="/projects/kamil@colostate.edu"
SCRATCH_DIR="/scratch/alpine/kamil@colostate.edu"
REPO_DIR="${PROJECTS_DIR}/vidur-rl"
MATRIX_FILE="${REPO_DIR}/scripts/exp09_queue_diag_pilot_matrix.tsv"
RETRY_ROOT="${SCRATCH_DIR}/vidur-rl/exp09_queue_diag_timeout_retry_20260426"
OUTPUT_DIR_BASE="${RETRY_ROOT}/runs/task_${SLURM_ARRAY_TASK_ID}"
SLURM_LOG_DIR="${RETRY_ROOT}/slurm"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

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
    print(sum(1 for _ in r))
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

# Strip possible CR from TSV line endings.
REPEAT_ID="${REPEAT_ID//$'\r'/}"
SCALE="${SCALE//$'\r'/}"
TRACE="${TRACE//$'\r'/}"
REPLICAS="${REPLICAS//$'\r'/}"
MODEL="${MODEL//$'\r'/}"
SCHEDULER="${SCHEDULER//$'\r'/}"

cat <<EOF
============================================================
SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
Node: $(hostname)
Retry task: $SCHEDULER / $MODEL / $REPLICAS / $TRACE / Scale $SCALE / Repeat $REPEAT_ID
Output root: $OUTPUT_DIR_BASE
Reason: retry EXP09 tasks that hit 6h wall time in job 25943224
============================================================
EOF

cd "$REPO_DIR"

export ENABLE_QUEUE_DIAGNOSTICS=1
export NUM_RUNS=1
export OUTPUT_DIR_BASE

./scripts/run_baseline_sweep.sh "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo "Done retry: task ${TASK_INDEX}: $SCHEDULER $MODEL $REPLICAS $TRACE scale $SCALE repeat $REPEAT_ID"
