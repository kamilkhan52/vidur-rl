#!/bin/bash
#SBATCH --job-name=postfix_eval
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-1
#SBATCH --output=slurm_logs/postfix_eval_%A_%a.out
#SBATCH --error=slurm_logs/postfix_eval_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <task_file.tsv> <run_root> [inference_epsilon]"
    exit 1
fi

TASK_FILE="$1"
RUN_ROOT="$2"
INFERENCE_EPSILON="${3:-0.05}"
VIDUR_ROOT="/projects/kamil@colostate.edu/vidur-rl"

echo "========================================================"
echo "Postfix inference evaluation array task"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "Task file: $TASK_FILE"
echo "Run root: $RUN_ROOT"
echo "========================================================"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur

cd "$VIDUR_ROOT"
mkdir -p slurm_logs

TASK_TSV=$(python3 - <<PY
import csv
import sys

task_id = str(${SLURM_ARRAY_TASK_ID})
with open("${TASK_FILE}", newline="") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row["task_id"] == task_id:
            print("\t".join([
                row["label"],
                row["exp_family"],
                row["checkpoint_path"],
                row.get("completed_iters", ""),
            ]))
            break
    else:
        sys.exit(1)
PY
)

if [[ -z "$TASK_TSV" ]]; then
    echo "Error: No task found for $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=$'\t' read -r LABEL EXP_FAMILY CHECKPOINT_PATH COMPLETED_ITERS <<< "$TASK_TSV"

OUTPUT_DIR="$RUN_ROOT/$LABEL"
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Label: $LABEL"
echo "  Experiment family: $EXP_FAMILY"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Completed iterations at launch: ${COMPLETED_ITERS:-unknown}"
echo "  Output dir: $OUTPUT_DIR"
echo "  Inference epsilon: $INFERENCE_EPSILON"
echo ""

CMD=(
    python3 -m vidur.main
    --replica_config_device a100
    --replica_config_model_name meta-llama/Llama-2-7b-hf
    --replica_config_tensor_parallel_size 1
    --cluster_config_num_replicas 4
    --replica_scheduler_config_type vllm_v1
    --global_scheduler_config_type rl_cache_aware
    --request_generator_config_type trace
    --trace_request_generator_config_trace_file data/processed_traces/mooncake_toolagent_trace.csv
    --trace_request_generator_config_time_scale_factor 20.0
    --trace_request_generator_config_max_tokens 122752
    --cache_config_enable_prefix_caching
    --r_l_cache_aware_global_scheduler_config_mode inference
    --r_l_cache_aware_global_scheduler_config_model_checkpoint_path "$CHECKPOINT_PATH"
    --r_l_cache_aware_global_scheduler_config_inference_epsilon "$INFERENCE_EPSILON"
    --metrics_config_output_dir "$OUTPUT_DIR"
    --no-metrics_config_write_json_trace
    --no-metrics_config_store_operation_metrics
    --no-metrics_config_store_batch_metrics
    --no-metrics_config_enable_chrome_trace
)

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/simulation.log"

RESULT_DIR=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
if [[ -z "$RESULT_DIR" ]]; then
    echo "Error: No timestamped result directory found in $OUTPUT_DIR"
    exit 1
fi

METRICS_FILE="$RESULT_DIR/request_metrics.csv"
if [[ ! -f "$METRICS_FILE" ]]; then
    echo "Error: Missing request metrics file: $METRICS_FILE"
    exit 1
fi

python3 - <<PY
import csv
from pathlib import Path
import pandas as pd

label = "$LABEL"
exp_family = "$EXP_FAMILY"
checkpoint_path = "$CHECKPOINT_PATH"
completed_iters = "$COMPLETED_ITERS"
result_dir = Path("$RESULT_DIR")
metrics_file = Path("$METRICS_FILE")
output_dir = Path("$OUTPUT_DIR")

df = pd.read_csv(metrics_file)
counts = df['replica'].value_counts()
imbalance = (counts.max() - counts.min()) / len(df)

summary = {
    'label': label,
    'exp_family': exp_family,
    'checkpoint_path': checkpoint_path,
    'completed_iters_at_launch': completed_iters,
    'completed_requests': len(df),
    'mean_e2e': df['request_e2e_time'].mean(),
    'median_e2e': df['request_e2e_time'].median(),
    'p95_e2e': df['request_e2e_time'].quantile(0.95),
    'p99_e2e': df['request_e2e_time'].quantile(0.99),
    'mean_ttft': df['prefill_e2e_time'].mean(),
    'mean_scheduling_delay': df['request_scheduling_delay'].mean(),
    'load_imbalance': imbalance,
    'result_dir': str(result_dir),
}

summary_file = output_dir / 'summary.csv'
with open(summary_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

print('Summary written to', summary_file)
print('Mean/Median/P99 E2E:', f"{summary['mean_e2e']:.3f}", f"{summary['median_e2e']:.3f}", f"{summary['p99_e2e']:.3f}")
PY

echo "========================================================"
echo "Finished: $(date)"
echo "========================================================"