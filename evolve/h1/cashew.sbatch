#!/bin/bash
# One h1 job on Cashew (CPU). Usage: sbatch -J <name> evolve/h1/cashew.sbatch <command> [args...]
#   e.g. sbatch -J h1_s40 evolve/h1/cashew.sbatch bash evolve/h1/es_seed.sh 40
#        sbatch -J h1_val evolve/h1/cashew.sbatch python -m evolve.h1.validate val
# Pass the command as positional arguments (never through --export: it splits on commas).
#SBATCH --partition=coe_cpu,all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=18:00:00
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
