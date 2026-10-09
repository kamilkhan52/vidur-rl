"""Below-knee min-estimated-TTFT core + knee-gated decode stack, with a
cache-independent running-decode-context price.

Currency: prefill-token-equivalents of delay.

    cost_j = pending_prefill_j
           + ramp * DECODE_WEIGHT * running_j * frac
           + lam_j * uncached_j
           + below * ( DYN_RUN_W * running_j * frac
                     + DYN_OUT_W * outstanding_j
                     + PENDING_TAIL_W * pend_excess_j^2 / pending_norm )
           + STALL_ALWAYS_W * uncached_j * co_decoders_j
           + STALL_TAIL_W   * uncached_j * decode_excess_j^2 / (mean_running+1)
           + ramp * ( decode-batch + decode-context + attended + gated stall )
           + ramp * DECODE_CTX_PROMPTS * n * (dctx_excess_j / dctx_norm)^2   # NEW
           + kv_pressure_j

    dctx_j = running_j * (mean prompt length of requests outstanding on j)

Change vs. the parent: the knee's largest loss is P99 TPOT on Synth while
its TTFT is far ahead of the bar.  The parent's projected-KV price left the
numbers unchanged, so physical KV occupancy is not the discriminating
signal.  With heavy prefix sharing a replica can host many decoders that
each attend over a long prompt while using little physical KV; decode step
time (hence TPOT) grows with that attended context.  Every decode-aware
knee term in the parent scales with the request's *uncached* fraction and
so vanishes for warm hits -- exactly the requests that pile onto hot-prefix
replicas.  The new term prices the excess running decode context of a
replica for every request, warm or cold, in own-prompt units so it can
compete with the miss price at the knee.  Below the knee (ramp = 0)
behaviour is the parent's.
"""

from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.5     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- below-knee decode-block / queue-balance term ---------------------------
DYN_RUN_W = 150.0          # prompt-equivalents per co-decode, scaled by frac
DYN_OUT_W = 60.0           # prompt-equivalents per outstanding request

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

# --- running decode-context tail price (NEW, cache-independent) -------------
DECODE_CTX_PROMPTS = 4.0   # own-prompt units at twice the fleet-mean decode context
DECODE_CTX_REF = 16000.0   # softens the guard when few decoders are resident

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


def _decode_context(replicas, queues, attended):
    """Estimated attention context of the sequences currently decoding on
    each replica: running count times the mean prompt length of the requests
    outstanding there.  Unlike physical KV usage this counts shared prefixes
    once per sequence, which is what decode step time pays for."""
    out = []
    for i, r in enumerate(replicas):
        k = len(queues[i])
        mean_len = attended[i] / k if k > 0 else 0.0
        running = r.running if r.running < k or k == 0 else k
        out.append(running * mean_len)
    return out


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

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # Running decode context (attention work per decode step).
    dctx = _decode_context(replicas, queues, attended)
    mean_dctx = sum(dctx) / num
    dctx_norm = mean_dctx + DECODE_CTX_REF

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

        cost = r.pending_prefill_tokens + ramp * DECODE_WEIGHT * r.running * frac
        cost += lam_i * uncached

        # Below-knee production-matching term (vanishes at the knee).
        if below > 0.0:
            cost += below * (DYN_RUN_W * r.running * frac
                             + DYN_OUT_W * r.outstanding_requests)
            pend_excess = r.pending_prefill_tokens - mean_pending
            if pend_excess > 0.0:
                cost += (below * PENDING_TAIL_W
                         * pend_excess * pend_excess / pending_norm)

        # Always-on decode-stall price (zero for warm hits).
        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += STALL_ALWAYS_W * uncached * co_decoders
            if decode_excess > 0.0:
                cost += (STALL_TAIL_W * uncached * decode_excess * decode_excess
                         / (mean_running + 1.0))

        # Knee-gated decode-aware stack.
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

            # NEW: cache-independent price on excess running decode context.
            # A warm hit decodes just as slowly on an overloaded replica as a
            # cold one, so this is NOT scaled by frac.
            dctx_excess = dctx[i] - mean_dctx
            if dctx_excess > 0.0:
                rel_d = dctx_excess / dctx_norm
                cost += ramp * DECODE_CTX_PROMPTS * n_pos * rel_d * rel_d

        # Preemption risk: flat always-on guard.
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

    # Session stickiness only at the knee; below it the cache view carries
    # the affinity already.
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
