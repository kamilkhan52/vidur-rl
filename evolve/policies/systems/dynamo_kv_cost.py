"""NVIDIA Dynamo KV-aware router cost (Dynamo docs: KV cache routing), simplified to our views.

cost_i = OVERLAP_WEIGHT * (prefill blocks still to compute on i) + (active decode blocks on i),
computed here in tokens: OVERLAP_WEIGHT * (pending uncached prefill on i + this request's
uncached prompt tokens on i) + (running requests on i x mean context, approximated by the
replica's KV usage x capacity). Docs: higher overlap weight favours TTFT, lower favours ITL.
OVERLAP_WEIGHT = 1.0 is the documented default. Lowest cost wins; ties by RNG.
"""

OVERLAP_WEIGHT = 1.0


def choose(request, replicas, ctx):
    costs = []
    for i, r in enumerate(replicas):
        prefill = r.pending_prefill_tokens + request.num_prefill_tokens - request.cached_prefix_tokens[i]
        decode = r.kv_usage * r.kv_capacity_tokens
        costs.append(OVERLAP_WEIGHT * prefill + decode)
    best = min(costs)
    c = [i for i, v in enumerate(costs) if v == best]
    return c[0] if len(c) == 1 else ctx.rng.choice(c)
