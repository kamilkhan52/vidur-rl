"""Power of two choices on KV load.

Sample two distinct replicas uniformly at random and route to the one with less KV load: KV blocks in use
(kv_usage x capacity) plus uncached prompt tokens routed there and not yet prefilled (they will claim KV soon);
ties go to the first sample. Prefix-cache blind. A floor baseline (readiness plan §5 item 1).
"""


def _kv_load(r):
    return r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens


def choose(request, replicas, ctx):
    a, b = ctx.rng.sample(range(len(replicas)), 2)
    return b if _kv_load(replicas[b]) < _kv_load(replicas[a]) else a
