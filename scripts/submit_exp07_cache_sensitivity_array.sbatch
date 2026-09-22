#!/bin/bash
#SBATCH --job-name=exp07_cache_sens
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp07_cache_sensitivity/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/exp07_cache_sensitivity/slurm/slurm-%A_%a.err
#SBATCH --array=1-24
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

PROJECTS_DIR="/projects/kamil@colostate.edu"
SCRATCH_DIR="/scratch/alpine/kamil@colostate.edu"
REPO_DIR="${PROJECTS_DIR}/vidur-rl"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

OUTPUT_DIR_BASE="${SCRATCH_DIR}/vidur-rl/exp07_cache_sensitivity/runs"
mkdir -p "${OUTPUT_DIR_BASE}" "${SCRATCH_DIR}/vidur-rl/exp07_cache_sensitivity/slurm"

TASKS=(
  "STICKY_LOR 70b 16R Synth 1.5"
  "STICKY_LOR 70b 16R Synth 3.0"
  "STICKY_LOR 70b 16R Synth 6.0"
  "STICKY_LOR 70b 16R Tool 1.5"
  "STICKY_LOR 70b 16R Tool 3.0"
  "STICKY_LOR 70b 16R Tool 6.0"

  "LOR 70b 16R Synth 1.5"
  "LOR 70b 16R Synth 3.0"
  "LOR 70b 16R Synth 6.0"
  "LOR 70b 16R Tool 1.5"
  "LOR 70b 16R Tool 3.0"
  "LOR 70b 16R Tool 6.0"

  "ROUND_ROBIN 70b 16R Synth 1.5"
  "ROUND_ROBIN 70b 16R Synth 3.0"
  "ROUND_ROBIN 70b 16R Synth 6.0"
  "ROUND_ROBIN 70b 16R Tool 1.5"
  "ROUND_ROBIN 70b 16R Tool 3.0"
  "ROUND_ROBIN 70b 16R Tool 6.0"

  "RANDOM 70b 16R Synth 1.5"
  "RANDOM 70b 16R Synth 3.0"
  "RANDOM 70b 16R Synth 6.0"
  "RANDOM 70b 16R Tool 1.5"
  "RANDOM 70b 16R Tool 3.0"
  "RANDOM 70b 16R Tool 6.0"
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$TASK_INDEX" -ge "${#TASKS[@]}" ]; then
  echo "ERROR: Task index $TASK_INDEX out of range (max ${#TASKS[@]})"
  exit 1
fi

read -r SCHEDULER MODEL REPLICAS TRACE SCALE <<< "${TASKS[$TASK_INDEX]}"

echo "============================================================"
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Task: $SCHEDULER / $MODEL / $REPLICAS / $TRACE / Scale $SCALE"
echo "Output root: $OUTPUT_DIR_BASE"
echo "============================================================"

cd "$REPO_DIR"

# Keep this intentionally lower than EXP02 to limit core-hours for the targeted sensitivity study.
NUM_RUNS=3 OUTPUT_DIR_BASE="$OUTPUT_DIR_BASE" ./scripts/run_baseline_sweep.sh \
  "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo "Done: $SCHEDULER $MODEL $REPLICAS $TRACE scale $SCALE"
