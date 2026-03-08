#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$REPO_ROOT/scripts/alpine_exp06_rl_experiments.csv"
SBATCH_SCRIPT="$REPO_ROOT/scripts/submit_exp06_rl_alpine_array.sbatch"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Missing config file: $CONFIG_FILE"
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
    echo "Error: No experiment rows found in $CONFIG_FILE"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm_logs

echo "Submitting EXP06 Alpine RL tasks ($TASK_COUNT rows) from $CONFIG_FILE"
sbatch --array="1-${TASK_COUNT}" "$SBATCH_SCRIPT"
