#!/bin/bash
#SBATCH --job-name=priority2_baseline
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-%A_%a.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/slurm/slurm-%A_%a.err
#SBATCH --array=1-224

#
# Priority 2 SLURM Array Job - Full Config × Trace Sweep
#
# Runs 224 baseline configurations across:
# - 4 schedulers: STICKY_LOR, ROUND_ROBIN, LOR, RANDOM
# - 4 configs: 7b-8R, 7b-16R, 70b-8R, 70b-16R
# - 3 traces: Synth, Tool, Conv
# - Variable scales per trace
#
# Submit: sbatch scripts/submit_priority2_sweep.sh
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

# Define all 224 tasks (scheduler, model, replicas, trace, scale)
# Format: "SCHEDULER MODEL REPLICAS TRACE SCALE"

TASKS=(
    # 7b-8R-Synth (6 scales × 4 schedulers = 24)
    "STICKY_LOR 7b 8R Synth 0.75"
    "STICKY_LOR 7b 8R Synth 1.0"
    "STICKY_LOR 7b 8R Synth 1.5"
    "STICKY_LOR 7b 8R Synth 3.0"
    "STICKY_LOR 7b 8R Synth 6.0"
    "STICKY_LOR 7b 8R Synth 12.0"
    "ROUND_ROBIN 7b 8R Synth 0.75"
    "ROUND_ROBIN 7b 8R Synth 1.0"
    "ROUND_ROBIN 7b 8R Synth 1.5"
    "ROUND_ROBIN 7b 8R Synth 3.0"
    "ROUND_ROBIN 7b 8R Synth 6.0"
    "ROUND_ROBIN 7b 8R Synth 12.0"
    "LOR 7b 8R Synth 0.75"
    "LOR 7b 8R Synth 1.0"
    "LOR 7b 8R Synth 1.5"
    "LOR 7b 8R Synth 3.0"
    "LOR 7b 8R Synth 6.0"
    "LOR 7b 8R Synth 12.0"
    "RANDOM 7b 8R Synth 0.75"
    "RANDOM 7b 8R Synth 1.0"
    "RANDOM 7b 8R Synth 1.5"
    "RANDOM 7b 8R Synth 3.0"
    "RANDOM 7b 8R Synth 6.0"
    "RANDOM 7b 8R Synth 12.0"
    
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
    
    # 7b-16R-Synth (6 scales × 4 schedulers = 24, NEW scales 0.75/1.0 only)
    "STICKY_LOR 7b 16R Synth 0.75"
    "STICKY_LOR 7b 16R Synth 1.0"
    "ROUND_ROBIN 7b 16R Synth 0.75"
    "ROUND_ROBIN 7b 16R Synth 1.0"
    "LOR 7b 16R Synth 0.75"
    "LOR 7b 16R Synth 1.0"
    "RANDOM 7b 16R Synth 0.75"
    "RANDOM 7b 16R Synth 1.0"
    # Note: 7b-16R-Synth scales 1.5-12.0 already done in Priority 1
    # Note: 7b-16R-Tool/Conv already done in Priority 1
    
    # 70b-8R-Synth (6 scales × 4 schedulers = 24)
    "STICKY_LOR 70b 8R Synth 0.75"
    "STICKY_LOR 70b 8R Synth 1.0"
    "STICKY_LOR 70b 8R Synth 1.5"
    "STICKY_LOR 70b 8R Synth 3.0"
    "STICKY_LOR 70b 8R Synth 6.0"
    "STICKY_LOR 70b 8R Synth 12.0"
    "ROUND_ROBIN 70b 8R Synth 0.75"
    "ROUND_ROBIN 70b 8R Synth 1.0"
    "ROUND_ROBIN 70b 8R Synth 1.5"
    "ROUND_ROBIN 70b 8R Synth 3.0"
    "ROUND_ROBIN 70b 8R Synth 6.0"
    "ROUND_ROBIN 70b 8R Synth 12.0"
    "LOR 70b 8R Synth 0.75"
    "LOR 70b 8R Synth 1.0"
    "LOR 70b 8R Synth 1.5"
    "LOR 70b 8R Synth 3.0"
    "LOR 70b 8R Synth 6.0"
    "LOR 70b 8R Synth 12.0"
    "RANDOM 70b 8R Synth 0.75"
    "RANDOM 70b 8R Synth 1.0"
    "RANDOM 70b 8R Synth 1.5"
    "RANDOM 70b 8R Synth 3.0"
    "RANDOM 70b 8R Synth 6.0"
    "RANDOM 70b 8R Synth 12.0"
    
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
    
    # 70b-16R-Synth (6 scales × 4 schedulers = 24)
    "STICKY_LOR 70b 16R Synth 0.75"
    "STICKY_LOR 70b 16R Synth 1.0"
    "STICKY_LOR 70b 16R Synth 1.5"
    "STICKY_LOR 70b 16R Synth 3.0"
    "STICKY_LOR 70b 16R Synth 6.0"
    "STICKY_LOR 70b 16R Synth 12.0"
    "ROUND_ROBIN 70b 16R Synth 0.75"
    "ROUND_ROBIN 70b 16R Synth 1.0"
    "ROUND_ROBIN 70b 16R Synth 1.5"
    "ROUND_ROBIN 70b 16R Synth 3.0"
    "ROUND_ROBIN 70b 16R Synth 6.0"
    "ROUND_ROBIN 70b 16R Synth 12.0"
    "LOR 70b 16R Synth 0.75"
    "LOR 70b 16R Synth 1.0"
    "LOR 70b 16R Synth 1.5"
    "LOR 70b 16R Synth 3.0"
    "LOR 70b 16R Synth 6.0"
    "LOR 70b 16R Synth 12.0"
    "RANDOM 70b 16R Synth 0.75"
    "RANDOM 70b 16R Synth 1.0"
    "RANDOM 70b 16R Synth 1.5"
    "RANDOM 70b 16R Synth 3.0"
    "RANDOM 70b 16R Synth 6.0"
    "RANDOM 70b 16R Synth 12.0"
    
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

# Total: 24+16+16+8+24+16+16+24+16+16+24+16+16 = 232 tasks
# But we only need 224 (excluding Priority 1 completed 7b-16R scales 1.5-12.0)
# Actually: keeping it simple, let array handle all 232, script will skip completed

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
