"""Mooncake-style conductor (Qin et al., FAST'25, Alg. 1), simplified to our views.

Pick the replica with the lowest estimated time to first token for this request:
queued prefill work on the replica plus this request's UNCACHED prompt tokens there.
Simplifications: a linear prefill cost (Mooncake fits a polynomial regression on prompt length and
prefix), and no hot-prefix replication (Alg. 1 l.5-8, 20-21: Mooncake may copy the best prefix to a
less busy prefill instance and charge the transfer time); there is no P/D disaggregation here.
Ties: fewer outstanding requests, then the scheduler's RNG.
Prefix match (FAST'25 §4.1): block keys are compared against each instance's cache keys, and the
Conductor "updates the cache ... for that instance" when it assigns a request. So the match is the
larger of the replica's actual prefix cache and the prefixes of requests routed there but not yet
prefilled (in flight; from the harness), plus requests routed earlier in the same arrival burst.
Nothing is credited after it has been prefilled and evicted.
"""


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def choose(request, replicas, ctx):
    burst = ctx.state.get("mooncake_burst")
    if burst is None or burst[0] != ctx.now:
        burst = (ctx.now, [set() for _ in replicas])
        ctx.state["mooncake_burst"] = burst
    idx = burst[1]
    best, cands = None, []
    for i, r in enumerate(replicas):
        hit = max(request.cached_prefix_tokens[i],
                  request.inflight_prefix_tokens[i] if request.inflight_prefix_tokens else 0,
                  min(_match(idx[i], request.prefix_hashes) * request.block_tokens, request.num_prefill_tokens))
        est = r.pending_prefill_tokens + request.num_prefill_tokens - hit
        key = (est, r.outstanding_requests)
        if best is None or key < best:
            best, cands = key, [i]
        elif key == best:
            cands.append(i)
    pick = cands[0] if len(cands) == 1 else ctx.rng.choice(cands)
    idx[pick].update(request.prefix_hashes)
    return pick
