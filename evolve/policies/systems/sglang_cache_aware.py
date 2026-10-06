"""SGLang router cache-aware policy (sgl-model-gateway src/policies/cache_aware.rs at sglang main ad7a9b5,
saved to research/vidur-rl-prior-art/fulltext/), simplified to our views.

If the load is imbalanced (max - min outstanding > BAL_ABS and max > BAL_REL x min), route to the
replica with the fewest outstanding requests. Otherwise match the request against the router's OWN
prefix index (the approximate radix tree, updated with every routed request: tree.insert on route):
if the best match covers more than CACHE_THRESHOLD of the prompt, route to that replica; else route
to the least-loaded replica (cache_aware.rs l.452-466; the header comment's "smallest tree" is stale).
Thresholds are the router's launch defaults (main.rs / router_args.py l.59-61: cache 0.3, abs 64,
rel 1.5); the library-internal CacheAwareConfig::default() is 0.5 / 32 / 1.1.
Simplifications: block-level (token) match instead of characters; no LRU eviction (the default tree
limit is far above our traces); outstanding requests stand in for the worker load counter.
"""

CACHE_THRESHOLD = 0.3
BAL_ABS = 64
BAL_REL = 1.5


def _least(idx, loads, ctx):
    lo = min(loads[i] for i in idx)
    c = [i for i in idx if loads[i] == lo]
    return c[0] if len(c) == 1 else ctx.rng.choice(c)


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def choose(request, replicas, ctx):
    idx = ctx.state.setdefault("sglang_index", [set() for _ in replicas])
    loads = [r.outstanding_requests for r in replicas]
    allidx = list(range(len(replicas)))
    hi, lo = max(loads), min(loads)
    if hi - lo > BAL_ABS and hi > BAL_REL * lo:
        pick = _least(allidx, loads, ctx)
    else:
        m = [_match(idx[i], request.prefix_hashes) for i in allidx]
        best = max(m)
        rate = best * request.block_tokens / max(request.num_prefill_tokens, 1)
        if best > 0 and rate > CACHE_THRESHOLD:
            c = [i for i in allidx if m[i] == best]
            pick = c[0] if len(c) == 1 else ctx.rng.choice(c)
        else:
            pick = _least(allidx, loads, ctx)
    idx[pick].update(request.prefix_hashes)
    return pick
