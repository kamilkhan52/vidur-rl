"""Load-gated min-TTFT: the min-estimated-TTFT core is always on, and the
entire decode-aware stack is gated by a cluster-level congestion ramp.

    congestion = max( mean_full / n_pos , mean_full / PROMPT_REF )
    ramp       = clamp((congestion - RAMP_START) / RAMP_WIDTH, 0, 1)

    cost_j = pending_prefill_j + lam_j * uncached_j            # core (min-TTFT)
           + ramp * ( DECODE_WEIGHT * running_j * frac_j
                    + DYN_OUT_W    * outstanding_j
                    + decode-batch + decode-context + attended
                    + stall(floor + knee) + stall-tail )
           + kv_pressure_j

    frac = uncached_j / num_prefill_tokens
    lam_j = lam * (1 + ramp * INTERFERENCE * decode_excess_j)

Hypothesis: the parent split decode-side pricing between `ramp *` and `below *`
(= 1 - ramp).  The below-knee half is a pure diversion away from the shortest
prefill queue -- the very quantity the production min-estimated-TTFT bar
minimises -- which is why the parent trails the bar on P99 TTFT at the light
(s1.0) points.  Moving every decode-side term under `ramp` makes the policy
exactly min-estimated-TTFT below the knee and bit-for-bit the parent's
decode-aware stack at the saturation knee.  The cluster-level ramp floor
(normalised by the median prompt) keeps long requests from falling out of the
knee stack, so the s0.75 win is preserved.
"""

from collections import deque

# --- congestion gauge / ramp ------------------------------------------------
PROMPT_REF = 7000.0        # fixed cluster reference prompt (median ~7k)
RAMP_START = 0.5           # congestion where the decode stack starts
RAMP_WIDTH = 1.5           # span over which it ramps to full weight

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.5     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- decode-batch / queue-balance terms (all ramp-gated) --------------------
DECODE_WEIGHT = 100.0      # prefill budget stolen by each concurrent decode
DYN_OUT_W = 40.0           # prompt-equivalents per outstanding request
BATCH_PRICE = 0.5          # per excess^2 / (mean + BATCH_REF)
BATCH_BASE = 500.0         # decode-sequence addition independent of prefill work
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ----------------------------------------------------
CONTEXT_PRICE = 0.05       # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- attended-context term (private + external) -----------------------------
ATTN_WEIGHT_MAX = 0.02     # token-equivalents per attended context token

# --- decode-stall term ------------------------------------------------------
STALL_FLOOR_W = 0.01       # base stall weight at full congestion
STALL_KNEE_W = 0.03        # extra stall weight at full congestion
STALL_TAIL_W = 0.008       # convex surcharge on excess co-decoders (P99 TBT)

# --- preemption risk --------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ----------------------------------------------------------
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

    # ---- 1. cluster-level congestion gauge and ramp ----------------------
    full = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running
            for r in replicas]
    mean_full = sum(full) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0

    # Cluster-level floor (normalised by the median prompt) so a long request
    # at the knee cannot keep the ramp near zero and stay in diversion mode.
    congestion_req = mean_full / n_pos
    congestion_cluster = mean_full / PROMPT_REF
    congestion = congestion_req if congestion_req > congestion_cluster \
        else congestion_cluster

    ramp = (congestion - RAMP_START) / RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0

    # Miss price.  At ramp = 0 this is exactly 1: production min-estimated-TTFT.
    lam = 1.0 + ramp * DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF
    stall_weight = STALL_FLOOR_W + STALL_KNEE_W

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

        # ---- min-estimated-TTFT core (always on) -------------------------
        cost = r.pending_prefill_tokens + lam_i * uncached

        # ---- decode-aware stack, gated by the congestion ramp ------------
        if ramp > 0.0:
            cost += ramp * (DECODE_WEIGHT * r.running * frac
                            + DYN_OUT_W * r.outstanding_requests)

            # Decode-batch price: sequences this request shares steps with,
            # plus a constant for the decode sequence it adds.
            batch_excess = r.outstanding_requests - mean_out
            if batch_excess > 0.0:
                cost += (ramp * BATCH_PRICE * (uncached + BATCH_BASE)
                         * batch_excess * batch_excess / batch_norm)

            # Decode-context price on resident (unique) KV tokens.
            ctx_excess = context[i] + uncached - mean_ctx
            if ctx_excess > 0.0:
                cost += ramp * CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

            # Attended-context price = marginal social decode cost.
            co_resident = len(queues[i])
            cost += ramp * ATTN_WEIGHT_MAX * (attended[i] * frac
                                              + uncached * co_resident)

            # Decode-stall price: surplus co-decoders set P99 TBT.
            if uncached > 0:
                co_decoders = r.outstanding_requests
                if r.running > co_decoders:
                    co_decoders = r.running
                cost += ramp * stall_weight * uncached * co_decoders
                if decode_excess > 0.0:
                    cost += (ramp * STALL_TAIL_W * uncached
                             * decode_excess * decode_excess
                             / (mean_running + 1.0))

        # Preemption risk is an always-on safety guard.
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
