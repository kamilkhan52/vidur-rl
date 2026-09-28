#!/bin/bash
# Frontier re-measurement v1 (2026-09-27): the EXP09 heuristic grid rerun on the
# audited code. 4 schedulers x 18 scenarios (7b; 8R/16R; Tool/Conv/Synth; scale
# 1.5/2.0/3.0) x 4 workload perturbations = 288 tasks. Differences from EXP09:
# KV-capacity trace filter on (no wedged replicas), restart-safe metrics, queue
# telemetry off, and paired workload perturbations:
#   p0      exact trace replay, seed 1 (comparable with EXP09)
#   p1..p3  seed 100*p+1 (101/201/301) for simulator, trace generator and the
#           scheduler RNG; Conv/Tool arrivals spread uniformly within their 3 s
#           Mooncake timestamp buckets (ARRIVAL_JITTER=3.0). Synth has ms
#           timestamps and is not jittered, so its p1..p3 vary only the scheduler RNG.
# All schedulers share the same perturbation for a given p (paired comparison).
#SBATCH --job-name=vidur_frontier_v1
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=1-288
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v1/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v1/slurm/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

REPO_DIR="/projects/kamil@colostate.edu/vidur-rl"
RUN_ROOT="/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v1"
MATRIX_FILE="${REPO_DIR}/scripts/frontier_v1_matrix.tsv"

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
