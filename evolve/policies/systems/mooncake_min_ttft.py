"""Mooncake-style conductor (Qin et al., FAST'25, Alg. 1), simplified to our views.

Pick the replica with the lowest estimated time to first token for this request:
queued prefill work on the replica plus this request's UNCACHED prompt tokens there,
at a fixed prefill rate (constant cancels in the argmin). Mooncake also adds KV-transfer
time between prefill and decode pools; there is no disaggregation here, so it is omitted.
Ties: fewer outstanding requests, then the scheduler's RNG.
"""


def choose(request, replicas, ctx):
    best, cands = None, []
    for i, r in enumerate(replicas):
        est = r.pending_prefill_tokens + request.num_prefill_tokens - request.cached_prefix_tokens[i]
        key = (est, r.outstanding_requests)
        if best is None or key < best:
            best, cands = key, [i]
        elif key == best:
            cands.append(i)
    return cands[0] if len(cands) == 1 else ctx.rng.choice(cands)
