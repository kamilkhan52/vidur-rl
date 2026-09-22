#!/bin/bash
#
# Priority Baseline Sweep - Sequential Execution
#
# Runs all baseline configs sequentially within a single SLURM job.
# Avoids array job issues by using a simple loop.
#
# Usage: ./scripts/run_priority_sequential.sh <priority>
#
# Priority 1: All schedulers × 7b-16R-Synth × All Scales (66 configs)

set -e

PRIORITY=${1:-1}
SCRATCH_ROOT="/scratch/alpine/kamil@colostate.edu"

# Priority configurations
declare -A SCHEDULERS_1=("1" "RANDOM" "2" "ROUND_ROBIN" "3" "LOR" "4" "LOP" "5" "LOP_BINARY" "6" "LOP_BATCH_END" "7" "LOP_UNCACHED" "8" "STICKY_RR" "9" "STICKY_LOR" "10" "RANKED_STICKY_LOP" "11" "TOLERANT_STICKY")
SCALES="48.0 24.0 12.0 6.0 3.0 1.5"
MODEL="7b"
REPLICAS="16R"
TRACE="Synth"

echo "Starting Priority $PRIORITY baseline sweep"
echo "Date: $(date)"

# Activate environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd /projects/kamil@colostate.edu/vidur-rl

# Run each configuration
total=0
completed=0
skipped=0

for sched in "${SCHEDULERS_1[@]}"; do
    for scale in $SCALES; do
        total=$((total + 1))
        
        output_dir="${SCRATCH_ROOT}/vidur-rl/baselines/${sched}/${MODEL}-${REPLICAS}-${TRACE}/scale${scale}"
        
        # Skip if already completed
        if [ -f "${output_dir}/summary.csv" ]; then
            echo "SKIP: $sched / $MODEL / $REPLICAS / $TRACE / Scale $scale"
            skipped=$((skipped + 1))
            continue
        fi
        
        echo "RUN: $sched / $MODEL / $REPLICAS / $TRACE / Scale $scale"
        
        # Run baseline
        ./scripts/run_baseline_sweep.sh "$sched" "$MODEL" "$REPLICAS" "$TRACE" "$scale"
        
        completed=$((completed + 1))
    done
done

echo ""
echo "========================================"
echo "Priority $PRIORITY Complete"
echo "Total: $total"
echo "Completed: $completed"
echo "Skipped: $skipped"
echo "Date: $(date)"
echo "========================================"
