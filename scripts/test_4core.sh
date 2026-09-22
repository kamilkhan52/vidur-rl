#!/bin/bash
#SBATCH --job-name=vidur_test_4core
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/test_4core.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/test_4core.err

#
# Test Job: Verify Vidur runs with 4 cores instead of 18
#
# Purpose: If successful, we can increase parallel runs from ~18 to ~56!
#

set -e

echo "============================================================"
echo "Test: Vidur with 4 cores + 64GB memory"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64G"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Conda environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

# Change to repo directory
cd /projects/kamil@colostate.edu/vidur-rl

# Run a simple baseline test (STICKY_LOR, 7b, 16R, Synth, Scale 12.0)
# This is a known-good config from Priority 1
echo "Running test configuration..."
./scripts/run_baseline_sweep.sh STICKY_LOR 7b 16R Synth 12.0

echo ""
echo "============================================================"
echo "Test completed successfully!"
echo "End time: $(date)"
echo "============================================================"
