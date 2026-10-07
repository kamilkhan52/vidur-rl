"""ABLATION (2026-10-07) of candidates_v3/579f2ac9531f.py: imbalance guards, stall price, hysteresis and in-flight view removed plus KV penalty. Not a search output."""
"""Congestion-priced, tail-aware cache routing with a marginal-social-cost
attended-context price and a TWO-PART decode-stall price on cache misses.

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + absolute / relative imbalance guards
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess resident KV tokens
           + attended-context price        # private + external decode cost (ramped)
           + decode-stall price            # uncached_j * co-decoders_j
                                           #   (fixed floor + congestion-ramped knee part)
           + kv_pressure_j                 # preemption risk

Change: at the knee TTFT is far better than the bar (0.48-0.82) while TBT and
TPOT trail it by 8-12%.  The parent's stall price saturated at 0.07 per
(uncached token x co-decoder), small next to the miss price lambda (up to 7),
so misses still landed on heavy decode batches; every prefill chunk of a miss
is a slow mixed step for each sequence decoding there.  The stall price is now
a fixed floor (same as the parent's at zero congestion, so below-the-knee
behaviour is unchanged) plus a knee component that ramps in with congestion,
doubling the full-ramp weight.  Queueing slack in TTFT is thereby spent on
keeping misses off large decode batches.  Full cache hits pay nothing, so
cache affinity (and E2E) is preserved.
"""

from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 1e30  # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 0.0       # scale-free tail guard, in prompt-units

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- attended-context term (private + external) -----------------------------
ATTN_WEIGHT_MAX = 0.4      # token-equivalents per attended context token (saturated)
ATTN_RAMP_START = 0.5      # congestion (own-prompt units) where the price starts
ATTN_RAMP_WIDTH = 1.5      # congestion span over which it ramps to full weight

# --- decode-stall term (miss-only) -----------------------------------------
STALL_FLOOR_W = 0.0      # always-on weight (= parent's weight at zero congestion)
STALL_KNEE_W = 0.0        # extra weight at full congestion ramp (TTFT has slack)

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 0.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.0
HYSTERESIS_MIN = 0.0


def _attended_context(replicas, state):
    """Per-replica FIFO of prompt lengths of requests still outstanding.

    The FIFO is trimmed (oldest first) to the replica's outstanding_requests
    count; returns the queues and their sums (attended context tokens)."""
    queues = state.get("attn_queues")
    if queues is None or len(queues) != len(replicas):
        queues = [deque() for _ in replicas]
        state["attn_queues"] = queues
        state["attn_sums"] = [0] * len(replicas)
    sums = state["attn_sums"]
    for i, r in enumerate(replicas):
        q = queues[i]
        while len(q) > r.outstanding_requests:
            sums[i] -= q.popleft()
    return queues, sums


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    queues, attended = _attended_context(replicas, ctx.state)

    # ---- 1. price of a cache miss ----------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    congestion = mean_load / n_pos
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_load if mean_load > n_pos else n_pos
    batch_norm = mean_out + BATCH_REF

    # Congestion ramp: 0 below the knee, 1 once queues build.
    ramp = (congestion - ATTN_RAMP_START) / ATTN_RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0
    attn_weight = ATTN_WEIGHT_MAX * ramp
    # Miss-only stall price: fixed floor at every load, plus a knee component
    # that spends the TTFT slack of a congested cluster on decode smoothness.
    stall_weight = STALL_FLOOR_W + STALL_KNEE_W * ramp

    # Resident KV tokens plus prompts queued for prefill.
    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = (0,) * num  # ABLATION: no in-flight prefix view
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)

        cost = load[i] + lam_i * uncached

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        # Decode-batch price: sequences this request will share steps with.
        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            cost += BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        # Decode-context price on resident (unique) KV tokens.
        ctx_excess = context[i] + uncached - mean_ctx
        if ctx_excess > 0.0:
            cost += CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

        # Attended-context price = marginal social decode cost:
        #   private:  attend over prompts of sequences outstanding here;
        #   external: each of them now also attends over our n tokens.
        if attn_weight > 0.0:
            co_resident = len(queues[i])
            cost += attn_weight * (attended[i] + n * co_resident)

        # Decode-stall price: our uncached tokens are prefilled in chunks that
        # share mixed steps with every sequence decoding here (running now, or
        # queued ahead of us under FCFS).  Each such step is a long gap for
        # each of them, so the slow gaps created ~ uncached * co_decoders.
        # Zero for a full cache hit: never pulls hits off warm replicas.
        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += stall_weight * uncached * co_decoders

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n)
    queues[pick].append(n)
    attended[pick] += n
    return pick


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
