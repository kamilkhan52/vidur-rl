#!/bin/bash
#SBATCH --job-name=EXP03_RL_Arch
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/slurm/slurm-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-5

# EXP03: RL Architecture & Lookahead Sweep
# Test different hidden layer architectures and lookahead depths
# Target: 7b-16R Tool @ Scale 1.5 (baseline E2E: 11.63s)

set -e

# Directories
VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"
SCRATCH_DIR="/scratch/alpine/kamil@colostate.edu/vidur-rl"
PARAM_FILE="$VIDUR_ROOT/scripts/exp03_params.csv"

# Create output directories
mkdir -p "$SCRATCH_DIR/slurm"
mkdir -p "$SCRATCH_DIR/exp03"

# Read parameters from CSV using Python (handles quoted fields correctly)
read HIDDEN_DIMS LOOKAHEAD_K < <(python3 <<EOF
import csv
import sys
task_id = int(sys.argv[1])
with open('$PARAM_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['task_id']) == task_id:
            print(row['hidden_dims'], row['lookahead'])
            break
EOF
$SLURM_ARRAY_TASK_ID)

echo "=========================================="
echo "EXP03 - Task $SLURM_ARRAY_TASK_ID"
echo "Hidden Dims: $HIDDEN_DIMS"
echo "Lookahead K: $LOOKAHEAD_K"
echo "=========================================="

# Fixed configuration
MODEL="meta-llama/Llama-2-7b-hf"
REPLICAS=16
TENSOR_PARALLEL=1
TRACE_FILE="$VIDUR_ROOT/data/processed_traces/mooncake_toolagent_trace.csv"
SCALE=1.5
REWARD_TYPE="ttft"
NUM_ITERATIONS=50

# Output directory
EXPERIMENT_ID="exp03_h${HIDDEN_DIMS}_k${LOOKAHEAD_K}"
EXPERIMENT_DIR="$SCRATCH_DIR/exp03/${EXPERIMENT_ID}"
mkdir -p "$EXPERIMENT_DIR"

# Checkpoint path (shared across iterations)
CHECKPOINT_PATH="$EXPERIMENT_DIR/checkpoint.pt"

# Activate conda environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"

# Run graduation loop
for ITER in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "=========================================="
    echo "Iteration $ITER / $NUM_ITERATIONS"
    echo "=========================================="
    
    ITER_DIR="$EXPERIMENT_DIR/iter${ITER}"
    mkdir -p "$ITER_DIR"
    
    python3 -m vidur.main \
        --replica_config_device a100 \
        --replica_config_model_name "$MODEL" \
        --replica_config_tensor_parallel_size "$TENSOR_PARALLEL" \
        --cluster_config_num_replicas "$REPLICAS" \
        --replica_scheduler_config_type vllm_v1 \
        --global_scheduler_config_type rl_cache_aware \
        --r_l_cache_aware_global_scheduler_config_mode training \
        --r_l_cache_aware_global_scheduler_config_reward_type "$REWARD_TYPE" \
        --r_l_cache_aware_global_scheduler_config_hidden_dims "$HIDDEN_DIMS" \
        --r_l_cache_aware_global_scheduler_config_lookahead_k "$LOOKAHEAD_K" \
        --r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_PATH" \
        --trace_request_generator_config_time_scale_factor "$SCALE" \
        --trace_request_generator_config_trace_file "$TRACE_FILE" \
        --metrics_config_output_dir "$ITER_DIR" \
        --no-metrics_config_write_json_trace \
        --request_generator_config_type trace \
        --cache_config_enable_prefix_caching \
        --r_l_cache_aware_global_scheduler_config_epsilon_decay_steps 1000
    
    echo "Iteration $ITER completed"
done

echo ""
echo "=========================================="
echo "EXP03 Task $SLURM_ARRAY_TASK_ID COMPLETE"
echo "Hidden Dims: $HIDDEN_DIMS"
echo "Lookahead K: $LOOKAHEAD_K"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================================="
