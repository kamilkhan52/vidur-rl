#!/bin/bash
# Knee calibration (2026-09-28). Frontier v1 (job 33083463) found two regimes: light load
# (all 16R, all Synth, 8R at 3.0; LOR queueing <= 7 %) and collapse (8R Tool/Conv at 1.5/2.0;
# queueing 75-99 %, P99 ~3e4 s). At 8R Tool/Conv scale 2.0 LOR is metastable: P99 E2E
# 16-74 s in some perturbations and ~2.8e4 s in others. This sweep brackets each
# scenario's knee (see scripts/load_calibration_matrix.tsv) with LOR, STICKY_LOR,
# ROUND_ROBIN, LOP_UNCACHED x perturbations 1-3 (seeds 101/201/301, Conv/Tool 3 s
# in-bucket jitter) = 432 tasks. Same code path as frontier v1.
#SBATCH --job-name=vidur_load_cal
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=1-432
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/load_cal/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/load_cal/slurm/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

REPO_DIR="/projects/kamil@colostate.edu/vidur-rl"
RUN_ROOT="/scratch/alpine/kamil@colostate.edu/vidur-rl/load_cal"
MATRIX_FILE="${REPO_DIR}/scripts/load_calibration_matrix.tsv"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur
cd "$REPO_DIR"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATRIX_FILE")
IFS=$'\t' read -r SCHEDULER MODEL REPLICAS TRACE SCALE PERT <<< "$LINE"
if [[ -z "${PERT:-}" ]]; then
  echo "ERROR: cannot parse matrix line $((SLURM_ARRAY_TASK_ID + 1)): $LINE"; exit 1
fi

JITTER=""
if [[ "$PERT" != "0" && "$TRACE" != "Synth" ]]; then JITTER="3.0"; fi
echo "task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): $SCHEDULER $MODEL $REPLICAS $TRACE scale $SCALE perturbation $PERT jitter ${JITTER:-0}"
echo "repo commit: $(git rev-parse --short HEAD)"

export ENABLE_QUEUE_DIAGNOSTICS=0
NUM_RUNS=1 SEED_BASE=$((100 * PERT)) ARRIVAL_JITTER="$JITTER" \
OUTPUT_DIR_BASE="${RUN_ROOT}/runs/p${PERT}" \
  ./scripts/run_baseline_sweep.sh "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo "done: $SCHEDULER $MODEL $REPLICAS $TRACE scale $SCALE perturbation $PERT"
