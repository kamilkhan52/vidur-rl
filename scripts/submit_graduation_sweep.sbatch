#!/bin/bash
#SBATCH --job-name=vidur_grad
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/vidur_grad_%A_%a.out
#SBATCH --error=slurm_logs/vidur_grad_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

# Phase 3 Graduation Sweep - SLURM Wrapper
# This script is called as a job array (indices 1-48)

set -e

# Project paths
VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

# Load environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"

# Get the specific track configuration from our generator script logic
# We use a temporary mapping file to translate array index to track params
TRACK_FILE="scripts/graduation_tracks.map"

# Read the line corresponding to the SLURM_ARRAY_TASK_ID
CONFIG_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRACK_FILE")

# Parse params: config_name scale hidden_dims reward
read -r CONFIG_NAME START_SCALE HIDDEN_DIMS REWARD_TYPE <<< "$CONFIG_LINE"

echo "========================================================"
echo "Vidur Graduation Track: $SLURM_ARRAY_TASK_ID"
echo "Config:  $CONFIG_NAME"
echo "RL Size: $HIDDEN_DIMS"
echo "Reward:  $REWARD_TYPE"
echo "Scale:   $START_SCALE"
echo "========================================================"

# Run the graduation script
# This script will now run in the foreground of this SLURM job
./scripts/run_graduation_training.sh "$CONFIG_NAME" "$START_SCALE" "$HIDDEN_DIMS" "$REWARD_TYPE"
