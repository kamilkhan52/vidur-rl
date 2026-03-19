#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/scripts/submit_priority2_sweep.sh"

cd "$REPO_ROOT"
mkdir -p /scratch/alpine/kamil@colostate.edu/vidur-rl/baselines_postfix/slurm

echo "Submitting EXP02 post-fix baseline rerun (Priority2 matrix)"
sbatch \
  --job-name=exp02_pf_baseline \
  --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines_postfix/slurm/slurm-%A_%a.out \
  --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines_postfix/slurm/slurm-%A_%a.err \
  --export=ALL,OUTPUT_DIR_BASE=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines_postfix/priority2,NUM_RUNS=5 \
  "$SBATCH_SCRIPT"
