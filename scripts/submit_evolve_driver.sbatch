#!/bin/bash
# Evolved-router search (EXP11) on one node: the driver proposes policies and runs every
# simulation as a local subprocess (up to --workers at a time). Resumable: rerun with the
# same RUN_DIR and the archive continues (seeds and baselines are not recomputed).
#   sbatch scripts/submit_evolve_driver.sbatch                      # DeepSeek via OpenRouter
#   PROPOSER=claude-code sbatch ...   (only where the claude CLI is logged in; not Alpine)
# The OpenRouter key is read from ~/.config/vidur-evolve/openrouter.env (mode 600).
#SBATCH --job-name=vidur_evolve
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve/slurm/evolve-%j.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve/slurm/evolve-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail
REPO_DIR="/projects/kamil@colostate.edu/vidur-rl"
SCRATCH="/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve"
RUN_DIR="${RUN_DIR:-$SCRATCH/runs/deepseek_r1}"
PROPOSER="${PROPOSER:-openrouter}"
ITERATIONS="${ITERATIONS:-60}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-16}"
MAX_COST_USD="${MAX_COST_USD:-8}"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur
cd "$REPO_DIR"
export EVOLVE_CACHE_DIR="$SCRATCH/cache"
mkdir -p "$RUN_DIR"
echo "commit $(git rev-parse --short HEAD) proposer $PROPOSER run_dir $RUN_DIR $(date)"
python -m evolve.driver --run-dir "$RUN_DIR" --proposer "$PROPOSER" --scenarios stage1 \
    --iterations "$ITERATIONS" --batch "$BATCH" --workers "$WORKERS" --max-cost-usd "$MAX_COST_USD"
