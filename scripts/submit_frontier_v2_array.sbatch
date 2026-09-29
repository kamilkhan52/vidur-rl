#!/bin/bash
# Frontier re-measurement v2 (2026-09-28), on code with the mid-prefill restart fix.
# Frontier v1's overload results (8R Tool/Conv collapse, the "metastable knee" at 8R/2.0)
# were largely an artifact: a request preempted mid-prefill came back with its unprocessed
# prompt tokens turned into output tokens decoded one per iteration (audit finding 25;
# on an overloaded window LOR P99 E2E 816 s pre-fix vs 31 s fixed). Wide load range because
# the real knee is unknown: LOR, STICKY_LOR, ROUND_ROBIN, RANDOM, LOP_UNCACHED (double-count
# fix) x {8R,16R} x {Tool,Conv,Synth} x scale {3.0,2.0,1.5,1.0,0.75,0.5} x perturbations
# 0-3 (p0 exact replay; p1-3 seeds 101/201/301 with Conv/Tool 3 s in-bucket jitter)
# = 720 tasks. Conv = canonical v1 trace; each run writes provenance.txt.
#SBATCH --job-name=vidur_frontier_v2
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=1-720
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v2/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v2/slurm/slurm-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

REPO_DIR="/projects/kamil@colostate.edu/vidur-rl"
RUN_ROOT="/scratch/alpine/kamil@colostate.edu/vidur-rl/frontier_v2"
MATRIX_FILE="${REPO_DIR}/scripts/frontier_v2_matrix.tsv"

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
