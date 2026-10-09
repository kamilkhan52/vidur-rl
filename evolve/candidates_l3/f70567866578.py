"""Below-knee decode-block bias relaxed toward production min-estimated-TTFT.

Currency: prefill-token-equivalents of delay.

    cost_j = pending_prefill_j
           + ramp * DECODE_WEIGHT * running_j * frac
           + lam_j * uncached_j
           + below * ( DYN_RUN_W * running_j * frac
                     + DYN_OUT_W * outstanding_j
                     + PENDING_TAIL_W * pend_excess_j^2 / pending_norm )
           + STALL_ALWAYS_W * uncached_j * co_decoders_j
           + STALL_TAIL_W   * uncached_j * decode_excess_j^2 / (mean_running+1)
           + ramp * ( decode-batch + decode-context + attended + gated stall
                      + excess_j^2 / IMBALANCE_SCALE )
           + kv_pressure_j

    frac  = uncached_j / num_prefill_tokens
    below = 1 - ramp

Change vs. the parent (below-knee only, since the changed weights multiply
`below` and therefore vanish at the saturation knee):

  * DYN_RUN_W 300 -> 150 and DYN_OUT_W 150 -> 60.  Stage-1 feedback shows the
    parent wins E2E/TPOT/TBT below the knee (0.94-0.97 / ~1.0 / ~1.0) but is
    3-8 % *worse* than the bar on P99 TTFT, with Tool-s1.0 the worst scenario.
    Below the knee the bar's TTFT is set by production min-estimated-TTFT,
    which routes purely on `pending + uncached`.  The parent's below-knee
    Dynamo-style bias instead diverts requests to low-decode-load replicas
    whose *prefill queue* is longer — the very thing that sets P99 TTFT — and
    hence adds cold prefills.  Relaxing those weights (keeping them small but
    non-zero so the below-knee E2E/TBT advantage is not surrendered) moves
    below-knee TTFT back toward 1.0 without changing the always-on stall floor
    that protects TBT, nor any ramp-gated knee term.
"""

from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.5     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- below-knee decode-block / queue-balance term (weakened) ----------------
DYN_RUN_W = 150.0          # prompt-equivalents per co-decode, scaled by frac (was 300)
DYN_OUT_W = 60.0           # prompt-equivalents per outstanding request (was 150)

# --- always-on decode-stall floor ------------------------------------------
STALL_ALWAYS_W = 0.022     # uncached * co_decoders, never gated
STALL_TAIL_W = 0.008       # convex surcharge on excess co-decoders (P99 TBT)

# --- below-knee prefill-queue tail guard (P99 TTFT) ------------------------
PENDING_TAIL_W = 0.008     # convex guard on excess pending prefill
PENDING_REF = 8000.0       # softens the guard when the cluster is idle

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 100.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 1e30     # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 0.0        # scale-free tail guard, in prompt-units

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 0.5          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_BASE = 500.0         # decode-sequence addition independent of prefill work
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 0.05       # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- attended-context term (private + external) -----------------------------
ATTN_WEIGHT_MAX = 0.02     # token-equivalents per attended context token (saturated)

# --- congestion ramp (also the gate on every decode-aware term) -------------
RAMP_START = 0.5           # congestion (own-prompt units) where the price starts
RAMP_WIDTH = 1.5           # congestion span over which it ramps to full weight

# --- gated decode-stall term ------------------------------------------------
STALL_FLOOR_W = 0.01       # gated weight at full congestion
STALL_KNEE_W = 0.03        # extra gated weight at full congestion ramp

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.20
HYSTERESIS_MIN = 500.0


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
    n_pos = n if n > 0 else 1.0

    queues, attended = _attended_context(replicas, ctx.state)

    # ---- 1. congestion signal (decode-aware, replica-level) --------------
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

    # Below-knee weight: fades out as the congestion ramp takes over, so the
    # knee's decode-aware stack is unchanged.
    below = 1.0 - ramp

    # Miss price.  At ramp = 0 this is 1: production min-estimated-TTFT.
    lam = 1.0 + ramp * DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_full if mean_full > n_pos else n_pos
    batch_norm = mean_out + BATCH_REF
    pending_norm = mean_pending + PENDING_REF

    attn_weight = ATTN_WEIGHT_MAX * ramp
    stall_weight = (STALL_FLOOR_W + STALL_KNEE_W) * ramp

    # Resident KV tokens plus prompts queued for prefill.
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

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + ramp * INTERFERENCE * decode_excess)

        # Prefill-proportional decode interference: only the request's own
        # uncached prefill work can be slowed by co-resident decodes.
        cost = r.pending_prefill_tokens + ramp * DECODE_WEIGHT * r.running * frac
        cost += lam_i * uncached

        # Below-knee production-matching term.  Weakened vs. the parent so that
        # below the knee the placement is closer to min-estimated-TTFT (the bar
        # there), instead of diverting requests toward low-decode-load replicas
        # whose prefill queue is longer -- which is what sets P99 TTFT.
        if below > 0.0:
            cost += below * (DYN_RUN_W * r.running * frac
                             + DYN_OUT_W * r.outstanding_requests)

            # Convex guard on the *excess* pending prefill of this replica:
            # P99 TTFT is set by the single longest prefill queue, which the
            # linear pending term only minimises in sum.
            pend_excess = r.pending_prefill_tokens - mean_pending
            if pend_excess > 0.0:
                cost += (below * PENDING_TAIL_W
                         * pend_excess * pend_excess / pending_norm)

        # Always-on decode-stall price.  Our uncached tokens share mixed steps
        # with every sequence decoding here; the *surplus* of co-decoders over
        # the mean is exactly what produces the one long stall that sets P99
        # TBT, so it is priced convexly.  Zero for warm hits.
        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += STALL_ALWAYS_W * uncached * co_decoders
            if decode_excess > 0.0:
                cost += (STALL_TAIL_W * uncached * decode_excess * decode_excess
                         / (mean_running + 1.0))

        # Every remaining decode-aware term is gated by the congestion ramp, so
        # at ramp = 0 the cost is exactly min-estimated-TTFT plus the above
        # below-knee and always-on terms.
        if ramp > 0.0:
            excess = full[i] - mean_full
            if excess > 0.0:
                cost += ramp * excess * excess / IMBALANCE_SCALE
                rel = excess / load_ref
                cost += ramp * REL_IMBALANCE * n_pos * rel * rel

            # Decode-batch price: the request's own prefill window (uncached)
            # during which it shares mixed steps with decoders, plus a small
            # constant for the decode sequence it adds.
            batch_excess = r.outstanding_requests - mean_out
            if batch_excess > 0.0:
                cost += (ramp * BATCH_PRICE * (uncached + BATCH_BASE)
                         * batch_excess * batch_excess / batch_norm)

            # Decode-context price on resident (unique) KV tokens.
            ctx_excess = context[i] + uncached - mean_ctx
            if ctx_excess > 0.0:
                cost += ramp * CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

            # Attended-context price = marginal social decode cost, again
            # proportional to the request's own prefill work.
            if attn_weight > 0.0:
                co_resident = len(queues[i])
                cost += attn_weight * (attended[i] * frac + uncached * co_resident)

            # Gated decode-stall price.
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
    pick = _select(request, replicas, ctx, costs, num, n, ramp)
    queues[pick].append(n)
    attended[pick] += n
    return pick


def _select(request, replicas, ctx, costs, num, n, ramp):
    best = min(costs)

    # Session stickiness is a knee phenomenon: below it, cached_prefix_tokens
    # already carries the affinity and stickiness only adds avoidable queueing.
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
