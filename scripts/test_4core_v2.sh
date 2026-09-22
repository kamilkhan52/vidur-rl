#!/bin/bash
#SBATCH --job-name=vidur_4core_test
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/test_4core_v2.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/test_4core_v2.err

#
# Test Job: Verify Vidur runs with 4 cores instead of 18
#

set -e

echo "============================================================"
echo "Test: Vidur with 4 cores + 64GB memory"
echo "Node: $(hostname)"
echo "CPUs requested: 4"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 64G"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Conda environment
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

# Change to repo directory
REPO_DIR="/projects/kamil@colostate.edu/vidur-rl"
cd "$REPO_DIR"

# Output directory
OUTPUT_DIR="/scratch/alpine/kamil@colostate.edu/vidur-rl/test_4core_output"
mkdir -p "$OUTPUT_DIR"

# Run Vidur directly with minimal config (Scale 12.0 should complete quickly)
echo "Running Vidur simulation..."
python -m vidur.main \
    --replica_config_device a100 \
    --replica_config_model_name meta-llama/Llama-2-7b-hf \
    --cluster_config_num_replicas 16 \
    --replica_config_tensor_parallel_size 1 \
    --replica_scheduler_config_type vllm \
    --global_scheduler_config_type sticky_lor \
    --trace_request_generator_config_trace_file data/processed_traces/mooncake_synthetic_trace.csv \
    --trace_request_generator_config_time_scale_factor 12.0 \
    --metrics_config_output_dir "$OUTPUT_DIR" \
    --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 \
    --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
    --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test PASSED! Vidur runs successfully with 4 cores."
    echo "Output files:"
    ls -lh "$OUTPUT_DIR"
else
    echo "❌ Test FAILED with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE
