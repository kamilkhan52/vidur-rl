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
    model: str
    tp_size: int
    num_replicas: int
    scheduler: str
    time_scale: float
    max_requests: Optional[int] = None
    
    @property
    def name(self) -> str:
        trace_short = os.path.basename(self.trace).replace(".csv", "")
        model_short = "7b" if "7b" in self.model else "70b"
        return f"{model_short}_{trace_short}_r{self.num_replicas}_{self.scheduler}_s{self.time_scale}"

    @property
    def output_dir(self) -> str:
        return os.path.join("results", "comprehensive_sweep", self.name)

def run_vidur(sim: SimulationRun):
    """Executes a single Vidur simulation run."""
    output_dir = sim.output_dir
    
    # Check if already completed (search in subdirectories due to timestamping)
    if os.path.exists(output_dir):
        completed = False
        for root, dirs, files in os.walk(output_dir):
            if "request_metrics.csv" in files or "metrics.json" in files:
                completed = True
                break
        
        if completed:
            # Clean up any partial/empty timestamped subdirectories
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    has_metrics = False
                    for _, _, subfiles in os.walk(item_path):
                        if "request_metrics.csv" in subfiles or "metrics.json" in subfiles:
                            has_metrics = True
                            break
                    if not has_metrics:
                        import shutil
                        shutil.rmtree(item_path)
            
            return f"SKIP: {sim.name} already exists."

    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        "python", "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", sim.model,
        "--replica_config_tensor_parallel_size", str(sim.tp_size),
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
        # Removed timeout to allow long-running simulations (70B/high-load) to finish naturally
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return f"SUCCESS: {sim.name} in {duration:.2f}s"
        else:
            return f"ERROR: {sim.name} failed with code {result.returncode}\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}"
    except Exception as e:
        return f"EXCEPTION: {sim.name} - {str(e)}"

def main():
    # Updated Research Grid: Focusing on the Tension Zone and pushing higher scales
    TRACES = [
        "data/processed_traces/mooncake_toolagent_trace.csv",
        "data/processed_traces/mooncake_synthetic_trace.csv"
    ]
    MODELS = [
        ("meta-llama/Llama-2-7b-hf", 1),
        ("meta-llama/Llama-2-70b-hf", 4)
    ]
    # Dropped 2-replica cases as they are consistently overloaded
    REPLICAS = [4, 8, 16]
    SCHEDULERS = ["lor", "sticky_lor", "random"]
    # Added 16.0 and 32.0 to find the breaking point
    SCALES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    
    experiment_grid: List[SimulationRun] = []
    
    for trace in TRACES:
        for model_name, tp_size in MODELS:
            for r in REPLICAS:
                for sched in SCHEDULERS:
                    for scale in SCALES:
                        # Full trace = No max_requests limit
                        experiment_grid.append(SimulationRun(
                            trace=trace,
                            model=model_name,
                            tp_size=tp_size,
                            num_replicas=r,
                            scheduler=sched,
                            time_scale=scale
                        ))

    logger.info(f"Starting Comprehensive Research Sweep with {len(experiment_grid)} total runs.")
    
    # Support MAX_WORKERS env var for Alpine/HPC
    max_workers = int(os.environ.get("MAX_WORKERS", 6))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_vidur, sim): sim for sim in experiment_grid}
        
        for future in as_completed(futures):
            res = future.result()
            logger.info(res)

if __name__ == "__main__":
    main()
