# Scheduler Evaluation Script

Comprehensive evaluation framework for comparing multiple scheduling policies across different workloads in the Vidur simulator.

## Features

- **Multiple Schedulers**: Compare Random, Round Robin, LOR, LOP_UNCACHED, Tolerant Sticky, and RL Cache-Aware
- **Multiple Workloads**: Test on different trace files (ShareGPT, Splitwise Conv, etc.)
- **Statistical Significance**: Run multiple iterations per configuration
- **Automated Analysis**: Extract metrics and generate comparison plots
- **Comprehensive Metrics**: E2E latency, TTFT, throughput, load balance, and more

## Quick Start

### 1. Basic Usage (All Schedulers, All Workloads)

```bash
cd /home/kamil/vidur
python scripts/evaluate_schedulers.py --output_dir results/eval_full
```

### 2. Quick Test (Subset of Schedulers)

```bash
python scripts/evaluate_schedulers.py \
  --schedulers round_robin rl_cache_aware \
  --workloads splitwise_conv \
  --quick \
  --output_dir results/quick_test
```

### 3. Production Run (Multiple Iterations)

```bash
python scripts/evaluate_schedulers.py \
  --num_runs 3 \
  --output_dir results/eval_3runs
```

## Command Line Options

### Required Options

- `--output_dir DIR`: Directory to store results and plots

### Optional Options

- `--schedulers SCHED [SCHED ...]`: Specific schedulers to evaluate
  - Available: `random`, `round_robin`, `lor`, `lop_uncached`, `tolerant_sticky`, `rl_cache_aware`
  - Default: All schedulers

- `--workloads WORK [WORK ...]`: Specific workloads to test
  - Available: `splitwise_conv`, `sharegpt`
  - Default: All workloads

- `--num_runs N`: Number of runs per configuration (default: 1)
  - Use 3-5 for statistical significance

- `--quick`: Quick mode - disables expensive metrics for faster evaluation

## Available Schedulers

1. **random**: Random replica selection
2. **round_robin**: Cyclic assignment across replicas
3. **lor**: Least Outstanding Requests
4. **lop_uncached**: Least Outstanding Prefills (cache-aware)
5. **tolerant_sticky**: Sticky routing with tolerance threshold
6. **rl_cache_aware**: Reinforcement Learning-based cache-aware scheduler

## Available Workloads

1. **splitwise_conv**: Splitwise conversational trace (19K requests)
   - Long conversations with shared prefixes
   - Tests cache affinity

2. **sharegpt**: ShareGPT trace (9K requests)
   - Diverse user interactions
   - Tests general performance

## Output Structure

```
results/eval_TIMESTAMP/
├── evaluation.log              # Detailed execution log
├── results.csv                 # Metrics in CSV format
├── results.json                # Metrics in JSON format (nested structures)
├── plots/                      # Generated plots
│   ├── mean_e2e_latency.png   # Mean E2E latency comparison
│   ├── p99_e2e_latency.png    # P99 E2E latency comparison
│   ├── mean_ttft.png          # TTFT comparison
│   ├── throughput.png         # Throughput comparison
│   ├── load_imbalance.png     # Load balance comparison
│   ├── latency_distributions.png  # Percentile distributions
│   └── summary_table.png      # Summary table with highlights
└── simulations/               # Raw simulation outputs
    ├── random_splitwise_conv_run0/
    ├── round_robin_splitwise_conv_run0/
    └── ...
```

## Metrics Collected

### Latency Metrics
- **Mean E2E Time**: Average end-to-end request latency
- **Median E2E Time**: 50th percentile latency
- **P90/P95/P99 E2E Time**: High percentile latencies
- **Mean TTFT**: Average time-to-first-token (prefill latency)

### Throughput Metrics
- **Throughput**: Tokens processed per second
- **Total Tokens**: Total tokens across all requests

### Load Balance Metrics
- **Load Imbalance**: (Max - Min) / Total requests
  - 0.0 = perfect balance
  - Higher values = more imbalanced

### Scheduling Metrics
- **Mean Scheduling Delay**: Average queueing delay
- **Mean Execution Time**: Average time spent executing

## Examples

### Example 1: Compare RL vs Best Baseline

```bash
python scripts/evaluate_schedulers.py \
  --schedulers rl_cache_aware lop_uncached \
  --workloads splitwise_conv \
  --num_runs 3 \
  --output_dir results/rl_vs_lop
```

### Example 2: Comprehensive Evaluation

```bash
python scripts/evaluate_schedulers.py \
  --num_runs 5 \
  --output_dir results/comprehensive_eval_$(date +%Y%m%d)
```

### Example 3: Debug Single Scheduler

```bash
python scripts/evaluate_schedulers.py \
  --schedulers rl_cache_aware \
  --workloads splitwise_conv \
  --quick \
  --output_dir results/debug_rl
```

## Analyzing Results

### 1. Check Execution Log

```bash
tail -f results/eval_*/evaluation.log
```

### 2. View Summary Table

```bash
open results/eval_*/plots/summary_table.png
```

### 3. Compare Latencies

```bash
open results/eval_*/plots/mean_e2e_latency.png
open results/eval_*/plots/p99_e2e_latency.png
```

### 4. Examine Raw Data

```python
import pandas as pd
df = pd.read_csv('results/eval_*/results.csv')
print(df.groupby('scheduler')['mean_e2e_time'].describe())
```

## Customization

### Adding New Workloads

Edit `scripts/evaluate_schedulers.py` and add to `WORKLOADS` dict:

```python
WORKLOADS = {
    'custom_trace': {
        'request_generator_config_type': 'trace',
        'trace_config_trace_file': 'data/processed_traces/custom.csv',
        'description': 'Custom workload description',
    },
}
```

### Adding RL Checkpoint

To evaluate a trained RL model, edit the script:

```python
if scheduler == 'rl_cache_aware':
    cmd.extend([
        "--r_l_cache_aware_global_scheduler_config_mode", "inference",
        "--r_l_cache_aware_global_scheduler_config_model_checkpoint_path",
        "rl_models/checkpoint_latest.pt"
    ])
```

### Modifying Hardware Configuration

Change in `run_simulation()` method:

```python
cmd = [
    sys.executable, "-m", "vidur.main",
    "--replica_config_device", "h100",  # Change GPU
    "--cluster_config_num_replicas", "8",  # Change replica count
    # ...
]
```

## Troubleshooting

### Issue: Simulations Timing Out

**Solution**: Increase timeout in `run_simulation()`:

```python
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=7200,  # 2 hours
)
```

### Issue: Out of Memory

**Solution**: Use `--quick` flag to disable expensive metrics:

```bash
python scripts/evaluate_schedulers.py --quick
```

### Issue: Missing Trace Files

**Solution**: Ensure trace files exist:

```bash
ls -lh data/processed_traces/
```

If missing, download or generate traces according to Vidur documentation.

### Issue: RL Scheduler Not Working

**Solution**:
1. Verify VLLM V1 scheduler is available
2. Check if prefix caching is causing issues (disabled by default in script)
3. Train a model first or use random initialization

## Performance Tips

1. **Parallel Execution**: The script runs simulations sequentially. For parallel execution, split by scheduler:

```bash
# Terminal 1
python scripts/evaluate_schedulers.py --schedulers random round_robin --output_dir results/batch1

# Terminal 2
python scripts/evaluate_schedulers.py --schedulers lor lop_uncached --output_dir results/batch2
```

2. **Quick Mode**: Always use `--quick` for testing/debugging

3. **Selective Evaluation**: Start with 1-2 schedulers and 1 workload, then expand

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@inproceedings{vidur2024,
  title={Vidur: A Large-Scale Simulation Framework For LLM Inference},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

Same as Vidur project.


### Main Graduation Progress Plot
To generate the main unified graduation progress plot comparing RL runs directly against Best Heuristic and Random routing ceilings for the same environments:
```bash
python3 scripts/plot_graduation_progress.py --input graduation_summary.csv --output graduation_progress_unified.png
```
