"""Below-knee min-TTFT affinity + convex tail guards, load-faded knee stack.

Reference (fitness 0.1144) wins the saturated point on every metric but
loses the below-knee point on P99 TTFT/TBT.  The parent's attempt to fix
this added an always-on below-knee convex pending guard with coefficient
1.0 -- far too large: at realistic backlogs the term is the same order as
the whole base cost, so below the knee the policy collapses to min-pending
and throws away the cache affinity / decode balance that the reference
relies on (TBT got worse, fitness 0.1113).

This version keeps the idea but fixes the magnitude:
  * below-knee convex pending guard with coefficient 0.10 (frac-scaled, so
    a fully warm request still reduces exactly to production min-TTFT) --
    it only tilts the worst prefill queue, never dominates the choice;
  * the always-on convex P99-TBT stall guard is strengthened
    (STALL_TAIL_W 0.025 -> 0.045) to recover the TBT tail the parent lost.
Everything else (RAMP_START = 0.5, 80/40 Dynamo balance scaled by frac,
fade, knee stack, hysteresis, tie-breaks) matches the 0.1144 reference.
"""

from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5
RUNNING_REF = 4.0
MISS_EXTERNALITY = 0.5
LAMBDA_MAX = 7.0

# --- below-knee decode-block / queue-balance term (optimum 80/40) ----------
DYN_RUN_W = 80.0
DYN_OUT_W = 40.0

# --- always-on convex tail shaping -----------------------------------------
STALL_ALWAYS_W = 0.020     # linear stall floor (uncached * co_decoders)
STALL_TAIL_W = 0.045       # convex P99-TBT guard (strengthened)
PENDING_TAIL_W = 0.025     # convex P99-TTFT guard
PENDING_REF = 6000.0

# --- below-knee convex prefill-queue guard (fra-scaled, gentle) ------------
PENDING_BELOW_CONVEX = 0.10

# --- knee-only extra tail shaping (load-scaled) ----------------------------
PENDING_KNEE_TAIL = 0.035
STALL_KNEE_TAIL = 0.025

# --- load-adaptive fade of spreading terms ---------------------------------
LOAD_UNIT = 10000.0
LOAD_FADE_START = 2.0

# --- queueing delay terms --------------------------------------------------
DECODE_WEIGHT = 100.0

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 0.5
BATCH_BASE = 500.0
BATCH_REF = 4.0
INTERFERENCE = 0.05

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 0.05
KV_REF = 16000.0

# --- attended-context term -------------------------------------------------
ATTN_WEIGHT_MAX = 0.02

# --- congestion ramp -------------------------------------------------------
RAMP_START = 0.5
RAMP_WIDTH = 1.5

# --- gated decode-stall term -----------------------------------------------
STALL_FLOOR_W = 0.01
STALL_KNEE_W = 0.03

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.20
HYSTERESIS_MIN = 500.0


def _attended_context(replicas, state):
    """Per-replica FIFO of prompt lengths of requests still outstanding."""
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
    n_pos = n if n > 0 else 1.0

    queues, attended = _attended_context(replicas, ctx.state)

    # ---- 1. load statistics ----------------------------------------------
    full = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_full = sum(full) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0

    congestion = mean_full / n_pos
    ramp = (congestion - RAMP_START) / RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0
    below = 1.0 - ramp

    # Load-adaptive fade: as cluster load grows, spreading terms shrink so
    # the router stops sacrificing prefix cache exactly when misses are most
    # expensive.  Below the fade start, tail_scale = 1.
    load_level = mean_full / LOAD_UNIT
    tail_scale = 1.0 / (1.0 + max(0.0, load_level - LOAD_FADE_START))

    lam = 1.0 + ramp * DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF
    pending_norm = mean_pending + PENDING_REF
    run_den = mean_running + 1.0

    attn_weight = ATTN_WEIGHT_MAX * ramp * tail_scale
    stall_weight = (STALL_FLOOR_W + STALL_KNEE_W) * ramp * tail_scale

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0
        frac = uncached / n_pos

        co_decoders = r.outstanding_requests
        if r.running > co_decoders:
            co_decoders = r.running
        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + ramp * INTERFERENCE * decode_excess)

        # base: production min-estimated-TTFT + knee decode interference
        cost = r.pending_prefill_tokens + ramp * DECODE_WEIGHT * r.running * frac
        cost += lam_i * uncached

        # below-knee Dynamo-style balance (optimum 80/40), scaled by frac:
        # a fully warm hit (frac = 0) reduces exactly to production min-TTFT,
        # while cold requests keep the full balance that wins saturated loads.
        if below > 0.0:
            cost += below * frac * (DYN_RUN_W * r.running
                                    + DYN_OUT_W * r.outstanding_requests)

        # convex P99-TTFT guard: only the above-mean pending queue (which
        # sets the longest prefill wait) is penalised.
        pend_excess = r.pending_prefill_tokens - mean_pending
        if pend_excess > 0.0:
            cost += (tail_scale * PENDING_TAIL_W
                     * pend_excess * pend_excess / pending_norm)

        # Gentle below-knee convex prefill-queue nudge (frac-scaled).  The
        # parent used coefficient 1.0, which dominated the base cost and
        # destroyed cache affinity / decode balance; here it only tilts the
        # worst prefill queue, and a fully warm request (frac = 0) still
        # reduces exactly to production min-TTFT.
        if below > 0.0 and uncached > 0 and pend_excess > 0.0:
            cost += (below * frac * PENDING_BELOW_CONVEX
                     * pend_excess * pend_excess / pending_norm)

        # always-on linear stall floor + convex P99-TBT guard: a long
        # uncached prefill landing on a busier-than-average decoder.
        if uncached > 0:
            cost += STALL_ALWAYS_W * uncached * co_decoders
            if decode_excess > 0.0:
                cost += (tail_scale * STALL_TAIL_W * uncached
                         * decode_excess * decode_excess / run_den)

        # knee-gated stack (load-scaled spreading)
        if ramp > 0.0:
            batch_excess = r.outstanding_requests - mean_out
            if batch_excess > 0.0:
                cost += (tail_scale * ramp * BATCH_PRICE * (uncached + BATCH_BASE)
                         * batch_excess * batch_excess / batch_norm)

            ctx_excess = context[i] + uncached - mean_ctx
            if ctx_excess > 0.0:
                cost += (tail_scale * ramp * CONTEXT_PRICE
                         * ctx_excess * ctx_excess / ctx_norm)

            if attn_weight > 0.0:
                co_resident = len(queues[i])
                cost += attn_weight * (attended[i] * frac
                                       + uncached * co_resident)

            if uncached > 0:
                cost += stall_weight * uncached * co_decoders

            if pend_excess > 0.0:
                cost += (tail_scale * ramp * PENDING_KNEE_TAIL
                         * pend_excess * pend_excess / pending_norm)
            if uncached > 0 and decode_excess > 0.0:
                cost += (tail_scale * ramp * STALL_KNEE_TAIL * uncached
                         * decode_excess * decode_excess / run_den)

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n, ramp)
    queues[pick].append(n)
    attended[pick] += n
    return pick


def _select(request, replicas, ctx, costs, num, n, ramp):
    best = min(costs)

    # Session stickiness only matters above the knee; below it, cached_prefix_tokens
    # already carries affinity and stickiness adds avoidable queueing.
    sticky = request.session_last_replica
    if ramp > 0.0 and 0 <= sticky < num:
        margin = ramp * max(HYSTERESIS_MIN, HYSTERESIS_FRAC * n)
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
