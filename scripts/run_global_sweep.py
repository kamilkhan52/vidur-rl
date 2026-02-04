import os
import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_sweep.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationRun:
    trace: str
    num_replicas: int
    scheduler: str
    time_scale: float
    max_requests: Optional[int] = None
    
    @property
    def name(self) -> str:
        trace_short = os.path.basename(self.trace).replace(".csv", "")
        return f"{trace_short}_r{self.num_replicas}_{self.scheduler}_s{self.time_scale}"

    @property
    def output_dir(self) -> str:
        return os.path.join("results", "sweep_Feb03", self.name)

def run_vidur(sim: SimulationRun):
    """Executes a single Vidur simulation run."""
    output_dir = sim.output_dir
    metrics_file = os.path.join(output_dir, "metrics.json")
    
    # Check if already completed
    if os.path.exists(metrics_file):
        return f"SKIP: {sim.name} already exists."

    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        "python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
        "--cluster_config_num_replicas", str(sim.num_replicas),
        "--replica_scheduler_config_type", "vllm_v1",
        "--global_scheduler_config_type", sim.scheduler,
        "--request_generator_config_type", "trace",
        "--trace_request_generator_config_trace_file", sim.trace,
        "--trace_request_generator_config_time_scale_factor", str(sim.time_scale),
        "--cache_config_enable_prefix_caching",
        "--metrics_config_output_dir", output_dir,
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_enable_chrome_trace"
    ]
    
    if sim.max_requests:
        cmd.extend(["--trace_request_generator_config_num_requests", str(sim.max_requests)])

    start_time = time.time()
    try:
        # Run with a 10-minute timeout for the test
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return f"SUCCESS: {sim.name} in {duration:.2f}s"
        else:
            return f"ERROR: {sim.name} failed with code {result.returncode}\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {sim.name} after 600s"
    except Exception as e:
        return f"EXCEPTION: {sim.name} - {str(e)}"

def main():
    # Grid Definition
    TRACES = [
        "data/processed_traces/mooncake_toolagent_trace.csv",
        "data/processed_traces/mooncake_conversation_trace_v2.csv",
        "data/processed_traces/mooncake_synthetic_trace.csv"
    ]
    REPLICAS = [4, 16]
    SCHEDULERS = ["lor", "sticky_lor"]
    SCALES = [1.0, 2.0, 4.0, 8.0]
    
    # Create experiment list
    experiment_grid: List[SimulationRun] = []
    for trace in TRACES:
        for r in REPLICAS:
            for sched in SCHEDULERS:
                for scale in SCALES:
                    # For the first test, we limit requests to 500
                    experiment_grid.append(SimulationRun(trace, r, sched, scale, max_requests=500))

    logger.info(f"Starting experiment sweep with {len(experiment_grid)} total runs.")
    
    # Use 10 cores to leave some headroom on the 12-core machine
    max_workers = 10
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_vidur, sim): sim for sim in experiment_grid}
        
        for future in as_completed(futures):
            res = future.result()
            logger.info(res)

if __name__ == "__main__":
    main()
