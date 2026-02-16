#!/bin/bash
#SBATCH --job-name=priority2_toolconv
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-%A_%a.err
#SBATCH --array=1-128

#
# Priority 2 Tool/Conv Resubmission - Fixed trace files
#
# Runs 128 baseline configurations:
# - 4 schedulers: STICKY_LOR, ROUND_ROBIN, LOR, RANDOM
# - 4 configs: 7b-8R, 7b-16R, 70b-8R, 70b-16R
# - 2 traces: Tool, Conv (Synth already complete)
# - Tool: 4 scales, Conv: 4 scales
#
# Submit: sbatch scripts/submit_priority2_toolconv.sh
# Monitor: squeue -u $USER --array
# Cancel: scancel JOBID

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECTS_DIR="/projects/kamil@colostate.edu"
SCRATCH_DIR="/scratch/alpine/kamil@colostate.edu"
REPO_DIR="${PROJECTS_DIR}/vidur-rl"

# Conda environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

# Ensure output directories exist
mkdir -p "${SCRATCH_DIR}/vidur-rl/baselines/slurm"

# ============================================================================
# TASK MAPPING
# ============================================================================

# Define all 128 tasks (scheduler, model, replicas, trace, scale)
# Format: "SCHEDULER MODEL REPLICAS TRACE SCALE"

TASKS=(
    # 7b-8R-Tool (4 scales × 4 schedulers = 16)
    "STICKY_LOR 7b 8R Tool 1.5"
    "STICKY_LOR 7b 8R Tool 3.0"
    "STICKY_LOR 7b 8R Tool 6.0"
    "STICKY_LOR 7b 8R Tool 12.0"
    "ROUND_ROBIN 7b 8R Tool 1.5"
    "ROUND_ROBIN 7b 8R Tool 3.0"
    "ROUND_ROBIN 7b 8R Tool 6.0"
    "ROUND_ROBIN 7b 8R Tool 12.0"
    "LOR 7b 8R Tool 1.5"
    "LOR 7b 8R Tool 3.0"
    "LOR 7b 8R Tool 6.0"
    "LOR 7b 8R Tool 12.0"
    "RANDOM 7b 8R Tool 1.5"
    "RANDOM 7b 8R Tool 3.0"
    "RANDOM 7b 8R Tool 6.0"
    "RANDOM 7b 8R Tool 12.0"
    
    # 7b-8R-Conv (4 scales × 4 schedulers = 16)
    "STICKY_LOR 7b 8R Conv 1.5"
    "STICKY_LOR 7b 8R Conv 3.0"
    "STICKY_LOR 7b 8R Conv 6.0"
    "STICKY_LOR 7b 8R Conv 12.0"
    "ROUND_ROBIN 7b 8R Conv 1.5"
    "ROUND_ROBIN 7b 8R Conv 3.0"
    "ROUND_ROBIN 7b 8R Conv 6.0"
    "ROUND_ROBIN 7b 8R Conv 12.0"
    "LOR 7b 8R Conv 1.5"
    "LOR 7b 8R Conv 3.0"
    "LOR 7b 8R Conv 6.0"
    "LOR 7b 8R Conv 12.0"
    "RANDOM 7b 8R Conv 1.5"
    "RANDOM 7b 8R Conv 3.0"
    "RANDOM 7b 8R Conv 6.0"
    "RANDOM 7b 8R Conv 12.0"
    
    # 7b-16R-Tool (4 scales × 4 schedulers = 16)
    "STICKY_LOR 7b 16R Tool 1.5"
    "STICKY_LOR 7b 16R Tool 3.0"
    "STICKY_LOR 7b 16R Tool 6.0"
    "STICKY_LOR 7b 16R Tool 12.0"
    "ROUND_ROBIN 7b 16R Tool 1.5"
    "ROUND_ROBIN 7b 16R Tool 3.0"
    "ROUND_ROBIN 7b 16R Tool 6.0"
    "ROUND_ROBIN 7b 16R Tool 12.0"
    "LOR 7b 16R Tool 1.5"
    "LOR 7b 16R Tool 3.0"
    "LOR 7b 16R Tool 6.0"
    "LOR 7b 16R Tool 12.0"
    "RANDOM 7b 16R Tool 1.5"
    "RANDOM 7b 16R Tool 3.0"
    "RANDOM 7b 16R Tool 6.0"
    "RANDOM 7b 16R Tool 12.0"
    
    # 7b-16R-Conv (4 scales × 4 schedulers = 16)
    "STICKY_LOR 7b 16R Conv 1.5"
    "STICKY_LOR 7b 16R Conv 3.0"
    "STICKY_LOR 7b 16R Conv 6.0"
    "STICKY_LOR 7b 16R Conv 12.0"
    "ROUND_ROBIN 7b 16R Conv 1.5"
    "ROUND_ROBIN 7b 16R Conv 3.0"
    "ROUND_ROBIN 7b 16R Conv 6.0"
    "ROUND_ROBIN 7b 16R Conv 12.0"
    "LOR 7b 16R Conv 1.5"
    "LOR 7b 16R Conv 3.0"
    "LOR 7b 16R Conv 6.0"
    "LOR 7b 16R Conv 12.0"
    "RANDOM 7b 16R Conv 1.5"
    "RANDOM 7b 16R Conv 3.0"
    "RANDOM 7b 16R Conv 6.0"
    "RANDOM 7b 16R Conv 12.0"
    
    # 70b-8R-Tool (4 scales × 4 schedulers = 16)
    "STICKY_LOR 70b 8R Tool 1.5"
    "STICKY_LOR 70b 8R Tool 3.0"
    "STICKY_LOR 70b 8R Tool 6.0"
    "STICKY_LOR 70b 8R Tool 12.0"
    "ROUND_ROBIN 70b 8R Tool 1.5"
    "ROUND_ROBIN 70b 8R Tool 3.0"
    "ROUND_ROBIN 70b 8R Tool 6.0"
    "ROUND_ROBIN 70b 8R Tool 12.0"
    "LOR 70b 8R Tool 1.5"
    "LOR 70b 8R Tool 3.0"
    "LOR 70b 8R Tool 6.0"
    "LOR 70b 8R Tool 12.0"
    "RANDOM 70b 8R Tool 1.5"
    "RANDOM 70b 8R Tool 3.0"
    "RANDOM 70b 8R Tool 6.0"
    "RANDOM 70b 8R Tool 12.0"
    
    # 70b-8R-Conv (4 scales × 4 schedulers = 16)
    "STICKY_LOR 70b 8R Conv 1.5"
    "STICKY_LOR 70b 8R Conv 3.0"
    "STICKY_LOR 70b 8R Conv 6.0"
    "STICKY_LOR 70b 8R Conv 12.0"
    "ROUND_ROBIN 70b 8R Conv 1.5"
    "ROUND_ROBIN 70b 8R Conv 3.0"
    "ROUND_ROBIN 70b 8R Conv 6.0"
    "ROUND_ROBIN 70b 8R Conv 12.0"
    "LOR 70b 8R Conv 1.5"
    "LOR 70b 8R Conv 3.0"
    "LOR 70b 8R Conv 6.0"
    "LOR 70b 8R Conv 12.0"
    "RANDOM 70b 8R Conv 1.5"
    "RANDOM 70b 8R Conv 3.0"
    "RANDOM 70b 8R Conv 6.0"
    "RANDOM 70b 8R Conv 12.0"
    
    # 70b-16R-Tool (4 scales × 4 schedulers = 16)
    "STICKY_LOR 70b 16R Tool 1.5"
    "STICKY_LOR 70b 16R Tool 3.0"
    "STICKY_LOR 70b 16R Tool 6.0"
    "STICKY_LOR 70b 16R Tool 12.0"
    "ROUND_ROBIN 70b 16R Tool 1.5"
    "ROUND_ROBIN 70b 16R Tool 3.0"
    "ROUND_ROBIN 70b 16R Tool 6.0"
    "ROUND_ROBIN 70b 16R Tool 12.0"
    "LOR 70b 16R Tool 1.5"
    "LOR 70b 16R Tool 3.0"
    "LOR 70b 16R Tool 6.0"
    "LOR 70b 16R Tool 12.0"
    "RANDOM 70b 16R Tool 1.5"
    "RANDOM 70b 16R Tool 3.0"
    "RANDOM 70b 16R Tool 6.0"
    "RANDOM 70b 16R Tool 12.0"
    
    # 70b-16R-Conv (4 scales × 4 schedulers = 16)
    "STICKY_LOR 70b 16R Conv 1.5"
    "STICKY_LOR 70b 16R Conv 3.0"
    "STICKY_LOR 70b 16R Conv 6.0"
    "STICKY_LOR 70b 16R Conv 12.0"
    "ROUND_ROBIN 70b 16R Conv 1.5"
    "ROUND_ROBIN 70b 16R Conv 3.0"
    "ROUND_ROBIN 70b 16R Conv 6.0"
    "ROUND_ROBIN 70b 16R Conv 12.0"
    "LOR 70b 16R Conv 1.5"
    "LOR 70b 16R Conv 3.0"
    "LOR 70b 16R Conv 6.0"
    "LOR 70b 16R Conv 12.0"
    "RANDOM 70b 16R Conv 1.5"
    "RANDOM 70b 16R Conv 3.0"
    "RANDOM 70b 16R Conv 6.0"
    "RANDOM 70b 16R Conv 12.0"
)

# ============================================================================
# MAIN
# ============================================================================

# Get task for this array index
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if [ $TASK_INDEX -ge ${#TASKS[@]} ]; then
    echo "ERROR: Task index $TASK_INDEX out of range (max ${#TASKS[@]})"
    exit 1
fi

TASK="${TASKS[$TASK_INDEX]}"
read -r SCHEDULER MODEL REPLICAS TRACE SCALE <<< "$TASK"

echo "============================================================"
echo "SLURM Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Task: $SCHEDULER / $MODEL / $REPLICAS / $TRACE / Scale $SCALE"
echo "============================================================"
echo ""

# Change to repo directory
cd "$REPO_DIR"

# Run the baseline sweep (with fixed error handling)
./scripts/run_baseline_sweep.sh "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo ""
echo "============================================================"
echo "Task completed successfully"
echo "============================================================"
