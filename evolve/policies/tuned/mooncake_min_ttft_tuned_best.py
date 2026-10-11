# Frozen by evolve.analysis.tune_baselines --install from tune_mooncake_v41/best.json: fitness +0.0376 after 125 evaluations,
# code_version cbeb5161f046, params {'QUEUE_W': 0.212829, 'LEN_QUAD': 0.23827, 'KV_W': 0.04456}.
"""Mooncake conductor (see systems/mooncake_min_ttft.py) with tunable cost constants, for the equal-budget tuning
control (readiness plan §5 item 4). QUEUE_W = 1, KV_W = 0, LEN_QUAD = 0 reproduces systems/mooncake_min_ttft.py.

Estimated TTFT ~ QUEUE_W * queued prefill + own uncached tokens * (1 + LEN_QUAD * prompt / 1000)
                 + KV_W * running KV tokens.
  QUEUE_W   weight of queued prefill work against this request's own prefill.
  LEN_QUAD  superlinear prompt-length cost (Mooncake fits a polynomial in prompt length and prefix).
  KV_W      weight of decode-side load (running KV tokens), a load-balance term the paper's Alg. 1 omits.
"""

QUEUE_W = 0.212829
LEN_QUAD = 0.23827
KV_W = 0.04456


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
        own = (request.num_prefill_tokens - hit) * (1.0 + LEN_QUAD * request.num_prefill_tokens / 1000.0)
        est = QUEUE_W * r.pending_prefill_tokens + own + KV_W * r.kv_usage * r.kv_capacity_tokens
        key = (est, r.outstanding_requests)
        if best is None or key < best:
            best, cands = key, [i]
        elif key == best:
            cands.append(i)
    pick = cands[0] if len(cands) == 1 else ctx.rng.choice(cands)
    idx[pick].update(request.prefix_hashes)
    return pick
