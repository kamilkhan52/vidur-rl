#!/bin/bash
#SBATCH --job-name=rl_train_array
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-4
#SBATCH --output=slurm_logs/rl_train_%A_%a.out
#SBATCH --error=slurm_logs/rl_train_%A_%a.err

# Load config for this array task
CONFIG_FILE="scripts/rl_training_configs.csv"
LINE=$(grep "^${SLURM_ARRAY_TASK_ID}," $CONFIG_FILE)

if [ -z "$LINE" ]; then
    echo "Error: No config found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Parse CSV line
IFS=',' read -r task_id model replicas trace scale <<< "$LINE"

echo "========================================================"
echo "RL Training Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model:    $model"
echo "Replicas: $replicas"
echo "Trace:    $trace"
echo "Scale:    $scale"
echo "Job ID:   $SLURM_ARRAY_JOB_ID ($SLURM_ARRAY_TASK_ID)"
echo "========================================================"

# Setup environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

# Set environment variables for the training script
export MODEL_NAME="$model"
export NUM_REPLICAS="$replicas"
export TRACE_NAME="$trace"
export TIME_SCALE_FACTOR="$scale"

# Start continuous training
./scripts/train_mooncake_continuous.sh
