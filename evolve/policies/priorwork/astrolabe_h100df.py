"""Astrolabe / Block (evolve/priorwork/astrolabe_core.py) on the h1 deployment: H100 (Vidur h100 profile with the
decode-attention fix, h100df), Llama-3.1-8B (Meta-Llama-3-8B profile), TP1, vllm_v1 with cap 128 and chunk 512,
30,208 KV blocks of 16 tokens (evolve/evaluate.py DEVICE_ARGS["h100df"]). The timing model is Vidur's execution-time
predictor loaded from the run's prediction bundle 4534b960326a7810 (key checked at start-up)."""
from evolve.priorwork.astrolabe_core import ReplicaSpec, build

SPEC = ReplicaSpec(
    device="h100", network_device="h100_pairwise_nvlink", model_name="meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size=1, num_blocks=30208, block_size=16, batch_size_cap=128, chunk_size=512,
    attention_input_file="../../data/profiling/compute_decodefix/{DEVICE}/{MODEL}/attention.csv",
    bundle_key="4534b960326a7810")

choose, observe, _inst = build(SPEC)
