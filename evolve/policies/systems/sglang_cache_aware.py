"""SGLang router cache-aware policy (sgl-router docs / v0.4 blog), simplified to our views.

If the load is imbalanced (max - min outstanding > 32 AND max > 1.1 x min), route to the
replica with the fewest outstanding requests (load balancing mode). Otherwise, if the best
prefix match covers more than cache_threshold = 0.5 of the prompt, route to that replica;
else route to the replica with the most free KV cache (the docs' "smallest tree").
Thresholds are the router's documented defaults (balance_abs_threshold 32,
balance_rel_threshold 1.1, cache_threshold 0.5).
"""

CACHE_THRESHOLD = 0.5
BAL_ABS = 32
BAL_REL = 1.1


def choose(request, replicas, ctx):
    loads = [r.outstanding_requests for r in replicas]
    hi, lo = max(loads), min(loads)
    if hi - lo > BAL_ABS and hi > BAL_REL * lo:
        c = [i for i, l in enumerate(loads) if l == lo]
        return c[0] if len(c) == 1 else ctx.rng.choice(c)
    n = max(request.num_prefill_tokens, 1)
    best = max(request.cached_prefix_tokens)
    if best / n > CACHE_THRESHOLD:
        c = [i for i, v in enumerate(request.cached_prefix_tokens) if v == best]
        return c[0] if len(c) == 1 else ctx.rng.choice(c)
    least = min(r.kv_usage for r in replicas)
    c = [i for i, r in enumerate(replicas) if r.kv_usage == least]
    return c[0] if len(c) == 1 else ctx.rng.choice(c)
