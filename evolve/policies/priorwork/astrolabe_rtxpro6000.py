"""Astrolabe / Block (evolve/priorwork/astrolabe_core.py) on the r6k deployment: RTX PRO 6000 (measured profile,
rtxpro6000), Llama-3.1-8B (Meta-Llama-3-8B profile), TP1, vllm_v1 with cap 128 and chunk 512, 30,208 KV blocks of 16
tokens (evolve/evaluate.py DEVICE_ARGS["rtxpro6000"]), shipped attention table path. Timing model: Vidur's
execution-time predictor from the run's prediction bundle 570195d3050e1c7d (key checked at start-up)."""
from evolve.priorwork.astrolabe_core import ReplicaSpec, build

SPEC = ReplicaSpec(
    device="rtxpro6000", network_device="rtxpro6000_pcie", model_name="meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size=1, num_blocks=30208, block_size=16, batch_size_cap=128, chunk_size=512,
    attention_input_file="../../data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
    bundle_key="570195d3050e1c7d")

choose, observe, _inst = build(SPEC)
