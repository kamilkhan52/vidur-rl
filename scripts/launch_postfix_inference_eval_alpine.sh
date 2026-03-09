#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/scripts/submit_postfix_inference_eval_array.sbatch"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/alpine/kamil@colostate.edu/vidur-rl}"
RUN_TAG="${RUN_TAG:-postfix_inference_eval_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$SCRATCH_ROOT/evals/$RUN_TAG"
TASK_FILE="$RUN_ROOT/tasks.tsv"

mkdir -p "$RUN_ROOT"

python3 - <<PY
from pathlib import Path
import csv

scratch_root = Path("$SCRATCH_ROOT")
run_root = Path("$RUN_ROOT")
task_file = Path("$TASK_FILE")

patterns = [
    scratch_root / "exp05",
    scratch_root / "exp06",
]

rows = []
for base in patterns:
    if not base.exists():
        continue
    for checkpoint in sorted(base.glob("exp*/checkpoints/*_best.pt")):
        run_label = checkpoint.parent.parent.name
        exp_family = checkpoint.parts[-4]
        summary_csv = checkpoint.parent.parent / "logs"
        summary_match = sorted(summary_csv.glob("*/iteration_summary.csv"))
        completed_iters = ""
        if summary_match:
            try:
                import csv as _csv
                with open(summary_match[-1], newline="") as f:
                    completed_iters = str(sum(1 for _ in _csv.DictReader(f)))
            except Exception:
                completed_iters = ""
        rows.append({
            "label": run_label,
            "exp_family": exp_family,
            "checkpoint_path": str(checkpoint),
            "completed_iters": completed_iters,
        })

if not rows:
    raise SystemExit("No EXP05/EXP06 best checkpoints found under scratch")

task_file.parent.mkdir(parents=True, exist_ok=True)
with open(task_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["task_id", "label", "exp_family", "checkpoint_path", "completed_iters"],
        delimiter="\t",
    )
    writer.writeheader()
    for idx, row in enumerate(rows, start=1):
        writer.writerow({"task_id": idx, **row})

print(task_file)
print(len(rows))
PY

TASK_COUNT=$(python3 - <<PY
import csv
with open("$TASK_FILE", newline="") as f:
    reader = csv.DictReader(f, delimiter='\t')
    print(sum(1 for _ in reader))
PY
)

echo "Discovered $TASK_COUNT checkpoints for postfix inference evaluation"
echo "Run root: $RUN_ROOT"
echo "Task file: $TASK_FILE"

cd "$REPO_ROOT"
mkdir -p slurm_logs

sbatch --array="1-${TASK_COUNT}" "$SBATCH_SCRIPT" "$TASK_FILE" "$RUN_ROOT"
