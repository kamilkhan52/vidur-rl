#!/bin/bash
#SBATCH --job-name=vidur_global_sweep
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/vidur_sweep_%j.out
#SBATCH --error=slurm_logs/vidur_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu

# Vidur Global Sweep - RL Cache-Aware Scheduling
# Runs a comprehensive grid sweep across models, replicas, and schedulers

echo "========================================================"
echo "Vidur Global Sweep - Comprehensive Research Grid"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128GB"
echo "Start Time: $(date)"
echo ""

# Project paths
VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

# Load required modules
echo "Initializing Conda..."
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
echo "  Sourced: /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh"
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

# Create directories
mkdir -p slurm_logs
mkdir -p tmp

# Sweep Configuration
export MAX_WORKERS="${MAX_WORKERS:-32}"
export TEMP_DIR="${TEMP_DIR:-tmp}"
echo "Sweep Configuration:"
echo "  MAX_WORKERS: $MAX_WORKERS"
echo "  TEMP_DIR: $TEMP_DIR"
echo ""

echo "========================================================"
echo "Starting Global Sweep"
echo "========================================================"
echo ""

# Run the sweep script
# We pass MAX_WORKERS to control parallelism on the node
python scripts/run_global_sweep.py

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "Job Complete"
echo "========================================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo ""

exit $EXIT_CODE
