#!/bin/bash
#SBATCH --job-name=priority2_missing
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-missing-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-missing-%A_%a.err
#SBATCH --array=1-32

#
# Priority 2 Missing Tasks - 7b-16R Tool & Conv
#
# Fills the gap from main Priority 2 sweep (Job 23901516).
# Main sweep included 176 tasks but missed 7b-16R Tool and Conv traces.
#
# This job runs the 32 missing configurations:
# - 7b-16R-Tool: 16 tasks (4 schedulers × 4 scales)
# - 7b-16R-Conv: 16 tasks (4 schedulers × 4 scales)
#
# Submit: sbatch scripts/submit_priority2_missing.sh
# Monitor: squeue -u $USER --array
# Runs in parallel with main sweep (Job 23901516)

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
# TASK MAPPING - 7b-16R Tool & Conv Only
# ============================================================================

TASKS=(
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
)

# Total: 16 + 16 = 32 tasks

# ============================================================================
# MAIN EXECUTION
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

# Run the baseline sweep
./scripts/run_baseline_sweep.sh "$SCHEDULER" "$MODEL" "$REPLICAS" "$TRACE" "$SCALE"

echo ""
echo "============================================================"
echo "Task completed successfully"
echo "============================================================"
