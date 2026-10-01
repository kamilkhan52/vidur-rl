#!/bin/bash
# Generic CPU job for evolve/ analyses on Alpine: runs "$CMD" from the repo checkout $REPO_DIR.
#   sbatch --cpus-per-task=16 --mem=96G --time=12:00:00 --job-name=vidur_ablate \
#     --export=ALL,REPO_DIR=/scratch/.../vidur-rl-exp12,CMD="python -m evolve.analysis.ablate ..." \
#     scripts/submit_python_job.sbatch
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.log
set -euo pipefail
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur
cd "$REPO_DIR"
echo "host $(hostname) start $(date -Is) cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "cmd: $CMD"
ls cache | grep prediction_bundle || echo "WARNING: no prediction bundle; results will not match other machines"
eval "$CMD"
echo "end $(date -Is)"
