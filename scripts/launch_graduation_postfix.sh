#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACK_FILE="$REPO_ROOT/scripts/graduation_tracks_postfix.map"
SBATCH_SCRIPT="$REPO_ROOT/scripts/submit_graduation_postfix.sbatch"

if [[ ! -f "$TRACK_FILE" ]]; then
    echo "Missing track file: $TRACK_FILE"
    exit 1
fi

TASK_COUNT=$(grep -cve '^\s*$' "$TRACK_FILE")
if [[ "$TASK_COUNT" -le 0 ]]; then
    echo "No tasks found in $TRACK_FILE"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm_logs

echo "Submitting EXP01 post-fix graduation rerun with $TASK_COUNT tracks"
sbatch --array="1-${TASK_COUNT}" "$SBATCH_SCRIPT"
