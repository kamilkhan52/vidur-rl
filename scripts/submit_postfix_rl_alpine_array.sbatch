#!/bin/bash
#SBATCH --job-name=exp05_rl
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-1
#SBATCH --output=slurm_logs/exp05_rl_%A_%a.out
#SBATCH --error=slurm_logs/exp05_rl_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"
SCRATCH_ROOT="/scratch/alpine/kamil@colostate.edu/vidur-rl/exp05"
CONFIG_FILE="scripts/alpine_postfix_rl_experiments.csv"

echo "========================================================"
echo "EXP05 RL Alpine array task"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "========================================================"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"
mkdir -p slurm_logs

CONFIG_TSV=$(python3 - <<PY
import csv
import sys

task_id = str(${SLURM_ARRAY_TASK_ID})
with open("${CONFIG_FILE}", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["task_id"] == task_id:
            print("\t".join([
                row["task_id"],
                row["model"],
                row["replicas"],
                row["trace"],
                row["scale"],
                row["tp"],
                row["reward_type"],
                row["lookahead_k"],
                row["hidden_dims"],
                row["epsilon_start"],
                row["epsilon_end"],
                row["epsilon_decay_steps"],
                row["learning_rate"],
                row["batch_size"],
                row["update_frequency"],
                row["replay_buffer_size"],
                row["max_iters"],
                row["checkpoint_tag"],
            ]))
            break
    else:
        sys.exit(1)
PY
)

if [[ -z "$CONFIG_TSV" ]]; then
    echo "Error: No config found for task $SLURM_ARRAY_TASK_ID in $CONFIG_FILE"
    exit 1
fi

IFS=$'\t' read -r task_id model replicas trace scale tp reward_type lookahead_k hidden_dims epsilon_start epsilon_end epsilon_decay_steps learning_rate batch_size update_frequency replay_buffer_size max_iters checkpoint_tag <<< "$CONFIG_TSV"

RUN_ROOT="$SCRATCH_ROOT/${checkpoint_tag}"
LOG_BASE_DIR="$RUN_ROOT/logs"
RESULTS_BASE_DIR="$RUN_ROOT/results"
CHECKPOINT_BASE_DIR="$RUN_ROOT/checkpoints"

mkdir -p "$LOG_BASE_DIR" "$RESULTS_BASE_DIR" "$CHECKPOINT_BASE_DIR"

echo "Configuration:"
echo "  Model: $model"
echo "  Replicas: $replicas"
echo "  Trace: $trace"
echo "  Scale: $scale"
echo "  TP: $tp"
echo "  Reward: $reward_type"
echo "  Lookahead K: $lookahead_k"
echo "  Hidden dims: $hidden_dims"
echo "  Epsilon: $epsilon_start -> $epsilon_end over $epsilon_decay_steps requests"
echo "  LR: $learning_rate"
echo "  Batch size: $batch_size"
echo "  Update frequency: $update_frequency"
echo "  Replay buffer size: $replay_buffer_size"
echo "  Max iterations: $max_iters"
echo "  Checkpoint tag: $checkpoint_tag"
echo "  Scratch root: $RUN_ROOT"
echo ""

export MODEL_NAME="$model"
export NUM_REPLICAS="$replicas"
export TRACE_NAME="$trace"
export TIME_SCALE_FACTOR="$scale"
export TP_SIZE="$tp"
export REWARD_TYPE="$reward_type"
export LOOKAHEAD_K="$lookahead_k"
export HIDDEN_DIMS="$hidden_dims"
export EPSILON_START="$epsilon_start"
export EPSILON_END="$epsilon_end"
export EPSILON_DECAY_STEPS="$epsilon_decay_steps"
export LEARNING_RATE="$learning_rate"
export BATCH_SIZE="$batch_size"
export UPDATE_FREQUENCY="$update_frequency"
export REPLAY_BUFFER_SIZE="$replay_buffer_size"
export MAX_ITERS="$max_iters"
export CHECKPOINT_TAG="$checkpoint_tag"
export LOG_BASE_DIR="$LOG_BASE_DIR"
export RESULTS_BASE_DIR="$RESULTS_BASE_DIR"
export CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR"
export LOG_TRANSITIONS="${LOG_TRANSITIONS:-0}"

./scripts/train_mooncake_continuous.sh

echo "========================================================"
echo "Finished: $(date)"
echo "========================================================"