#!/bin/bash
# One job on Cashew (CPU). Usage: sbatch -J <name> scripts/cashew_job.sbatch <command> [args...]
#   e.g. sbatch -J orc_s3 scripts/cashew_job.sbatch bash evolve/rl_plan/rerun_lin4_oracle_s3.sh
# Pass the command as positional arguments (never through --export: it splits on commas).
#SBATCH --partition=coe_cpu,all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=60G
#SBATCH --time=13:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) start $(date -Is) cpus $SLURM_CPUS_PER_TASK"
echo "cmd: $*"
echo "git $(git rev-parse --short HEAD) $(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ') modified"
ls cache/prediction_bundle_*.pkl
python -c "from evolve import evaluate as ev; print('code_version', ev.code_version())"
nice -n 5 "$@"
echo "end $(date -Is)"
