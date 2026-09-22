#!/bin/bash
#SBATCH --job-name=openevolve
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve/slurm_%j.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

# ── Config ────────────────────────────────────────────────────────────────────
ITERATIONS=100
# ─────────────────────────────────────────────────────────────────────────────

set -e

VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"

# Load API key from .env (not hardcoded in sbatch)
ENV_FILE="$VIDUR_ROOT/openevolve/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

mkdir -p /scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve

echo "========================================================"
echo "OpenEvolve RL Formulation Search"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Iterations: $ITERATIONS"
echo "Started: $(date)"
echo "========================================================"

bash openevolve/run_alpine.sh --iterations "$ITERATIONS"

echo "========================================================"
echo "Finished: $(date)"
echo "========================================================"
