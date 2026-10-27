"""Astrolabe / Block (evolve/priorwork/astrolabe_core.py) on the test3b deployment: A100 (Vidur a100 profile),
Llama-3.1-8B (Meta-Llama-3-8B profile), TP1, vllm_v1 with cap 128 and chunk 512, Vidur's default network device and
KV size (MemoryPlanner: 483,328 tokens = 30,208 blocks of 16, as KV_CAPACITY["l3_8b"]), shipped attention table path.
Timing model: Vidur's execution-time predictor from the run's prediction bundle 8e6e78e985b87b2a (key checked at
start-up)."""
from evolve.priorwork.astrolabe_core import ReplicaSpec, build

SPEC = ReplicaSpec(
    device="a100", network_device="a100_pairwise_nvlink", model_name="meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size=1, num_blocks=30208, block_size=16, batch_size_cap=128, chunk_size=512,
    attention_input_file="../../data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
    bundle_key="8e6e78e985b87b2a")

choose, observe, _inst = build(SPEC)
