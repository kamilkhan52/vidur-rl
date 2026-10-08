"""Congestion-priced cache routing with a lambda-matched TTFT-tail guard (70B).

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess resident KV tokens
           + decode-stall price            # uncached_j * co-decoders_j (fixed floor)
           + TTFT-tail guard               # ramped in with CLUSTER congestion,
                                           #   weight proportional to lambda
           + kv_pressure_j                 # preemption risk

Change vs parent: under congestion the miss price lambda grows up to 7x,
but the TTFT guard kept a fixed weight of 0.5 per excess token.  A warm
replica could therefore carry ~5x the saved prefill as extra queue and
still win, so long prompts were parked behind deep queues and set the P99
TTFT (Tool-s2.0 1.08x bar).  The guard weight is now TTFT_GUARD_W * ramp *
lambda: the same congestion signal that makes a miss expensive also makes
straying from the fastest first token expensive, which bounds the TTFT a
cache hit may buy to a small multiple of the work it saves.  Below the
knee (ramp = 0) the policy is identical to the parent.
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

# --- decode-stall term (miss-only, fixed weight) ---------------------------
STALL_W = 0.042            # token-equivalents per (uncached token x co-decoder)

# --- congestion ramp for the guard (0 below the knee, 1 once queues build) --
GUARD_REF_TOKENS = 8000.0  # backlog scale: never larger than ~a median prompt
RAMP_START = 0.5           # congestion (reference-prompt units) where ramp starts
RAMP_WIDTH = 1.5           # congestion span over which it reaches 1

# --- soft TTFT-tail guard ---------------------------------------------------
TTFT_GUARD_W = 0.5         # token-equivalents per excess token, per unit lambda
TTFT_SLACK_FRAC = 0.10     # free excess, relative to the best estimate
TTFT_SLACK_PROMPT = 0.25   # ... plus this many own-prompt units

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    # ---- 1. cluster-wide quantities ---------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    occupancy = min(1.0, mean_running / RUNNING_REF)
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    congestion = mean_load / n_pos            # per-prompt: prices the miss
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF

    # Guard ramp: cluster congestion on a size-independent scale.
    guard_scale = min(float(n_pos), GUARD_REF_TOKENS)
    guard_congestion = mean_load / guard_scale
    ramp = min(1.0, max(0.0, (guard_congestion - RAMP_START) / RAMP_WIDTH))

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

    # Guard grows with the miss price, so a cache hit can only buy a bounded
    # multiple of its own savings in extra time-to-first-token.
    guard_w = TTFT_GUARD_W * ramp * lam

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

        # Miss-only decode-stall price (fixed weight; zero for a full hit).
        if u > 0:
            co_decoders = max(r.outstanding_requests, r.running)
            cost += STALL_W * u * co_decoders

        # Soft TTFT-tail guard: only under congestion, only beyond the slack.
        if guard_w > 0.0:
            over = est[i] - ttft_free
            if over > 0.0:
                cost += guard_w * over

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
