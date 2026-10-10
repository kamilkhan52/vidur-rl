"""Join the shortest queue on KV load (production-stack / llm-d style KV-utilisation scorer, cache blind).

Route to the replica with the least KV load: KV blocks in use (kv_usage x capacity) plus uncached prompt tokens
routed there and not yet prefilled. Ties: fewer outstanding requests, then the scheduler's RNG.
JSQ on outstanding requests is the built-in `lor`. A floor baseline (readiness plan §5 item 1).
"""


def choose(request, replicas, ctx):
    keys = [(r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens, r.outstanding_requests)
            for r in replicas]
    best = min(keys)
    cands = [i for i, k in enumerate(keys) if k == best]
    return cands[0] if len(cands) == 1 else ctx.rng.choice(cands)
