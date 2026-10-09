"""Below-knee decode-block balance + knee-tail shaping (v6).

Identical to the parent v5 in the below-knee regime (all new terms are gated
by `ramp`, so at ramp = 0 the cost is exactly the parent's).  The only change
is that the *knee* stack now contains two convex guards on the excess-over-mean
of the two quantities that set the P99 tails at saturation:

    + ramp * PENDING_KNEE_TAIL * pend_excess^2 / pending_norm   (P99 TTFT)
    + ramp * STALL_KNEE_TAIL   * uncached * decode_excess^2 / (mean_running+1)
                                                                (P99 TBT)

Both are zero for the least-loaded replica, so they sharpen the worst replica
without distorting spreading.  All parent constants are unchanged.
"""

from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5
RUNNING_REF = 4.0
MISS_EXTERNALITY = 0.5
LAMBDA_MAX = 7.0

# --- below-knee decode-block / queue-balance term --------------------------
DYN_RUN_W = 300.0
DYN_OUT_W = 150.0

# --- always-on decode-stall floor ------------------------------------------
STALL_ALWAYS_W = 0.015

# --- NEW: knee-only convex tail guards (gated by ramp) ---------------------
PENDING_KNEE_TAIL = 0.05   # excess pending prefill -> P99 TTFT
STALL_KNEE_TAIL = 0.03     # excess co-decoders * uncached -> P99 TBT
PENDING_REF = 8000.0       # softens the pending guard when the cluster is idle

# --- queueing delay terms --------------------------------------------------
DECODE_WEIGHT = 100.0
IMBALANCE_SCALE = 1e30
REL_IMBALANCE = 0.0

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 0.5
BATCH_BASE = 500.0
BATCH_REF = 4.0
INTERFERENCE = 0.05

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 0.05
KV_REF = 16000.0

# --- attended-context term --------------------------------------------------
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

    lam = 1.0 + ramp * DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_full if mean_full > n_pos else n_pos
    batch_norm = mean_out + BATCH_REF
    pending_norm = mean_pending + PENDING_REF

    attn_weight = ATTN_WEIGHT_MAX * ramp
    stall_weight = (STALL_FLOOR_W + STALL_KNEE_W) * ramp

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

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

        cost = r.pending_prefill_tokens + ramp * DECODE_WEIGHT * r.running * frac
        cost += lam_i * uncached

        if below > 0.0:
            cost += below * (DYN_RUN_W * r.running * frac
                             + DYN_OUT_W * r.outstanding_requests)

        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += STALL_ALWAYS_W * uncached * co_decoders

        if ramp > 0.0:
            excess = full[i] - mean_full
            if excess > 0.0:
                cost += ramp * excess * excess / IMBALANCE_SCALE
                rel = excess / load_ref
                cost += ramp * REL_IMBALANCE * n_pos * rel * rel

            batch_excess = r.outstanding_requests - mean_out
            if batch_excess > 0.0:
                cost += (ramp * BATCH_PRICE * (uncached + BATCH_BASE)
                         * batch_excess * batch_excess / batch_norm)

            ctx_excess = context[i] + uncached - mean_ctx
            if ctx_excess > 0.0:
                cost += ramp * CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

            if attn_weight > 0.0:
                co_resident = len(queues[i])
                cost += attn_weight * (attended[i] * frac + uncached * co_resident)

            if uncached > 0:
                co_decoders = r.outstanding_requests
                if r.running > co_decoders:
                    co_decoders = r.running
                cost += stall_weight * uncached * co_decoders

            # ---- NEW: knee-tail shaping ---------------------------------
            # P99 TTFT is the single longest prefill queue, so price excess
            # pending prefill convexly (zero for the lightest replica).
            pend_excess = r.pending_prefill_tokens - mean_pending
            if pend_excess > 0.0:
                cost += (ramp * PENDING_KNEE_TAIL
                         * pend_excess * pend_excess / pending_norm)

            # P99 TBT is the single worst decode stall: a long uncached prefill
            # landing on a busier-than-average decoder.  Price co-decoder
            # excess convexly, scaled by this request's own uncached work.
            if uncached > 0 and decode_excess > 0.0:
                cost += (ramp * STALL_KNEE_TAIL * uncached
                         * decode_excess * decode_excess / (mean_running + 1.0))

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    pick = _select(request, replicas, ctx, costs, num, n, ramp)
    queues[pick].append(n)
    attended[pick] += n
    return pick


def _select(request, replicas, ctx, costs, num, n, ramp):
    best = min(costs)

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
