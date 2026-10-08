"""Congestion-priced cache routing with a cluster-level congestion ramp (70B).

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess resident KV tokens
           + decode-stall price            # uncached_j * co-decoders_j (miss only)
           + TTFT-tail guard               # convex, ramped in with congestion
           + kv_pressure_j                 # preemption risk

Change vs parent: the parent's knee ramp was mean_load / n, i.e. congestion in
units of the ARRIVING prompt.  Short prompts saw full ramp (strong stall price
and TTFT guard they hardly need) while long prompts -- which set the P99 TTFT
tail and whose many prefill chunks create the slow mixed steps behind P99
TBT/TPOT -- saw ramp ~0 even past the knee.  The ramp now measures congestion
in units of the TYPICAL prompt (EWMA of recent prompt lengths), so it is a
property of the cluster, not the request: at the knee, every request
(especially long misses) pays the stall price and the TTFT guard; below the
knee the ramp is zero as before.  The miss externality in lambda keeps the
own-prompt normalisation (backlog relative to this request's own work).
"""

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- cluster-level congestion ramp (0 below the knee, 1 once queues build) --
RAMP_START = 0.5           # congestion (typical-prompt units) where the ramp starts
RAMP_WIDTH = 1.5           # congestion span over which it reaches 1
PROMPT_EWMA_ALPHA = 0.05   # smoothing of the typical prompt length
PROMPT_EWMA_INIT = 7000.0  # prior: median Mooncake prompt

# --- decode-stall term (miss-only) -----------------------------------------
STALL_FLOOR_W = 0.042      # always-on weight
STALL_KNEE_W = 0.10        # extra weight at full congestion ramp

# --- convex TTFT-tail guard -------------------------------------------------
TTFT_GUARD_W = 1.0         # prompt-equivalents per (excess / prompt)^2
TTFT_SLACK_FRAC = 0.10     # free excess, relative to the best estimate
TTFT_SLACK_PROMPT = 0.25   # ... plus this many own-prompt units

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def _typical_prompt(n, state):
    """EWMA of prompt lengths: the cluster's unit of prefill work."""
    avg = state.get("prompt_ewma", PROMPT_EWMA_INIT)
    if n > 0:
        avg += PROMPT_EWMA_ALPHA * (n - avg)
        state["prompt_ewma"] = avg
    return avg if avg > 1.0 else 1.0


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1
    typical = _typical_prompt(n, ctx.state)

    # ---- 1. cluster-wide quantities ---------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    # Miss price: backlog measured against this request's own work.
    occupancy = min(1.0, mean_running / RUNNING_REF)
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    own_congestion = mean_load / n_pos
    lam += MISS_EXTERNALITY * max(0.0, own_congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF

    # Knee ramp: a property of the cluster, in typical-prompt units.
    cluster_congestion = mean_load / typical
    ramp = min(1.0, max(0.0, (cluster_congestion - RAMP_START) / RAMP_WIDTH))
    stall_weight = STALL_FLOOR_W + STALL_KNEE_W * ramp

    # Resident KV tokens plus prompts queued for prefill.
    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. uncached work and Mooncake-style TTFT estimate ------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    uncached = []
    est = []
    for i, r in enumerate(replicas):
        cached = max(request.cached_prefix_tokens[i], inflight[i])
        u = n - cached
        if u < 0:
            u = 0
        uncached.append(u)
        est.append(r.pending_prefill_tokens + u)
    best_est = min(est)
    ttft_free = best_est + TTFT_SLACK_FRAC * best_est + TTFT_SLACK_PROMPT * n_pos
    guard_w = TTFT_GUARD_W * ramp

    # ---- 3. score every replica ------------------------------------------
    costs = []
    for i, r in enumerate(replicas):
        u = uncached[i]

        decode_excess = max(0.0, r.running - mean_running)
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)
        cost = load[i] + lam_i * u

        # Decode-batch price: sequences this request will share steps with.
        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            cost += BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        # Decode-context price on resident (unique) KV tokens.
        ctx_excess = context[i] + u - mean_ctx
        if ctx_excess > 0.0:
            cost += CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

        # Miss-only decode-stall price: prefill chunks of our uncached tokens
        # share mixed steps with every sequence decoding here.
        if u > 0:
            co_decoders = max(r.outstanding_requests, r.running)
            cost += stall_weight * u * co_decoders

        # Convex TTFT-tail guard: nearly free for small detours from the
        # min-TTFT replica, prohibitive for detours of several prompts.
        if guard_w > 0.0:
            over = est[i] - ttft_free
            if over > 0.0:
                cost += guard_w * over * over / n_pos

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    return _select(request, replicas, ctx, costs, num, n)


def _select(request, replicas, ctx, costs, num, n):
    best = min(costs)

    sticky = request.session_last_replica
    if 0 <= sticky < num:
        margin = max(HYSTERESIS_MIN, HYSTERESIS_FRAC * n)
        if costs[sticky] <= best + margin:
            return sticky

    cands = [i for i, c in enumerate(costs) if c <= best + 1e-9]
    if len(cands) == 1:
        return cands[0]

    warmest = max(request.cached_prefix_tokens[i] for i in cands)
    cands = [i for i in cands if request.cached_prefix_tokens[i] == warmest]
    if len(cands) == 1:
        return cands[0]

    quietest = min(replicas[i].running for i in cands)
    cands = [i for i in cands if replicas[i].running == quietest]
    if len(cands) == 1:
        return cands[0]

    fewest = min(replicas[i].outstanding_requests for i in cands)
    cands = [i for i in cands if replicas[i].outstanding_requests == fewest]
    return ctx.rng.choice(cands)
