#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/scripts/alpine_exp03_postfix_experiments.csv"
SBATCH_SCRIPT="$REPO_ROOT/scripts/submit_exp03_postfix_array.sbatch"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Missing config file: $CONFIG_FILE"
    exit 1
fi

TASK_COUNT=$(python3 - <<PY
import csv
with open("$CONFIG_FILE", newline="") as f:
    reader = csv.DictReader(f)
    print(sum(1 for row in reader if any(str(v).strip() for v in row.values())))
PY
)

if [[ "$TASK_COUNT" -le 0 ]]; then
    echo "No tasks found in $CONFIG_FILE"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm_logs

echo "Submitting EXP03 post-fix rerun with $TASK_COUNT tasks"
sbatch --array="1-${TASK_COUNT}" "$SBATCH_SCRIPT"
