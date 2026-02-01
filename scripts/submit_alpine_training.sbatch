#!/bin/bash
#SBATCH --job-name=vidur_rl_train
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/vidur_rl_train_%j.out
#SBATCH --error=slurm_logs/vidur_rl_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu

# Vidur RL Training - Cache-Aware Global Scheduler
# Trains an RL agent for LLM inference scheduling

echo "========================================================"
echo "Vidur RL Training - 16-GPU Toolagent"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64GB"
echo "Start Time: $(date)"
echo ""

# Project paths
VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

# Load required modules
echo "Loading modules..."
module purge
module load anaconda
echo "  Loaded: anaconda"
echo ""

# Activate conda environment
echo "Activating conda environment..."
conda activate vidur
echo "  Activated: vidur"
echo ""

# Change to project directory
cd "$VIDUR_ROOT"
echo "Working directory: $(pwd)"
echo ""

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# Epsilon configuration (can be overridden via environment)
export EPSILON_START="${EPSILON_START:-0.90}"
export EPSILON_END="${EPSILON_END:-0.05}"
export EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-8000}"
export RESET_EPSILON_ON_LOAD="${RESET_EPSILON_ON_LOAD:-1}"
export TEMP_DIR="${TEMP_DIR:-tmp}"
echo "Epsilon Configuration:"
echo "  EPSILON_START: $EPSILON_START"
echo "  EPSILON_END: $EPSILON_END"
echo "  EPSILON_DECAY_STEPS: $EPSILON_DECAY_STEPS"
echo "  RESET_EPSILON_ON_LOAD: $RESET_EPSILON_ON_LOAD"
echo "  TEMP_DIR: $TEMP_DIR"
echo ""

# Verify trace file exists
TRACE_FILE="data/processed_traces/mooncake_toolagent_trace.csv"
if [ ! -f "$TRACE_FILE" ]; then
    echo "ERROR: Trace file not found: $TRACE_FILE"
    exit 1
fi
echo "Trace file: $TRACE_FILE ($(wc -l < "$TRACE_FILE") lines)"
echo ""

echo "========================================================"
echo "Starting Training"
echo "========================================================"
echo ""

# Run training script
./scripts/train_16gpu_toolagent.sh

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "Job Complete"
echo "========================================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo ""

exit $EXIT_CODE
