"""NVIDIA Dynamo KV-aware router cost (docs @519e735550: docs/router/routing-concepts.md l.31, router-design.md
l.13, l.55-96; source @6ebaf07bcb: lib/kv-router/src/sequences/prompt_registry.rs l.257-317), simplified to our views.

cost_i = PREFILL_SCALE * prefill_i + decode_i, lowest cost wins (temperature 0; ties by RNG):
  prefill_i = prefill tokens still to compute on i (pending uncached prefill) + this request's tokens not
              already cached on i (overlap from KV events = the replica's actual prefix cache);
  decode_i  = potential decode blocks = active blocks on i + incoming_active_blocks, where
              incoming_active_blocks = request tokens NOT already shared with active sequences on i
              (additional_active_blocks = query_len - overlap_depth with the active sequences).
Our views: active blocks ~ running KV (kv_usage x capacity) + queued uncached prompts (pending prefill);
overlap with active sequences = the prefix shared with requests routed to i and not yet finished
(active_prefix_tokens, exact from the harness) or routed to i earlier in the same arrival burst (the router
registers a request's blocks as active when it routes it). Cache left by finished requests is NOT credited
here (prompt_registry.rs l.33-37: "not cache misses"); it enters only the prefill term.
Simplifications: tokens instead of blocks; running KV includes output blocks (Dynamo counts prompt-side
blocks by default); queued prompts are counted by their uncached tokens without de-duplication and without
their cached-prefix blocks; a partly prefilled request is counted in both; overlap-credit decay not modelled. PREFILL_SCALE = 1.0 is unit scale (the
docs give no default for prefill_load_scale).
"""

PREFILL_SCALE = 1.0


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def choose(request, replicas, ctx):
    burst = ctx.state.get("dynamo_burst")
    if burst is None or burst[0] != ctx.now:
        burst = (ctx.now, [set() for _ in replicas])
        ctx.state["dynamo_burst"] = burst
    costs = []
    active = request.active_prefix_tokens or (0,) * len(replicas)
    for i, r in enumerate(replicas):
        prefill = r.pending_prefill_tokens + request.num_prefill_tokens - request.cached_prefix_tokens[i]
        same_burst = min(_match(burst[1][i], request.prefix_hashes) * request.block_tokens, request.num_prefill_tokens)
        overlap_active = max(active[i], same_burst)
        incoming_active = request.num_prefill_tokens - overlap_active
        decode = r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens + incoming_active
        costs.append(PREFILL_SCALE * prefill + decode)
    best = min(costs)
    c = [i for i, v in enumerate(costs) if v == best]
    pick = c[0] if len(c) == 1 else ctx.rng.choice(c)
    burst[1][pick].update(request.prefix_hashes)
    return pick
