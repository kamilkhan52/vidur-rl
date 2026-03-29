#!/bin/bash
#SBATCH --job-name=vidur_grad_pf
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-12
#SBATCH --output=slurm_logs/vidur_grad_pf_%A_%a.out
#SBATCH --error=slurm_logs/vidur_grad_pf_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"
TRACK_FILE="scripts/graduation_tracks_postfix.map"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"

CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRACK_FILE")
if [[ -z "$CONFIG_LINE" ]]; then
    echo "No config for task ${SLURM_ARRAY_TASK_ID} in $TRACK_FILE"
    exit 1
fi

read -r CONFIG_NAME START_SCALE HIDDEN_DIMS REWARD_TYPE <<< "$CONFIG_LINE"

echo "========================================================"
echo "Vidur Graduation Post-Fix Track: $SLURM_ARRAY_TASK_ID"
echo "Config:  $CONFIG_NAME"
echo "Scale:   $START_SCALE"
echo "RL Size: $HIDDEN_DIMS"
echo "Reward:  $REWARD_TYPE"
echo "========================================================"

./scripts/run_graduation_training.sh "$CONFIG_NAME" "$START_SCALE" "$HIDDEN_DIMS" "$REWARD_TYPE"
