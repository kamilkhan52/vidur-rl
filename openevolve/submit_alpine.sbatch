#!/bin/bash
#SBATCH --job-name=openevolve
#SBATCH --account=csu65_alpine1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve/slurm_%j.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/openevolve/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -e

VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"

echo "========================================================"
echo "OpenEvolve RL Formulation Search"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================================"

bash openevolve/run_alpine.sh --iterations 100

echo "========================================================"
echo "Finished: $(date)"
echo "========================================================"
