"""Below-knee min-TTFT affinity + convex tail guards + a crowding discount on
the prefix-cache credit, measured in BOTH live-sequence COUNT and RESIDENT
CONTEXT TOKENS, with slope, cap and the context term all bought with spare
capacity.

Lineage evidence.  The credit-bounded crowding discount is the only mechanism
in this lineage that ever helped (-0.0065 -> -0.0038); gating it by spare
capacity rather than fleet congestion helped again (-0.0038 -> -0.0015), and
buying cap and slope headroom with spare capacity helped once more
(-0.0015 -> +0.0028).  Every cache-INDEPENDENT additive spreading toll failed
(share EMA -0.0109, decode toll -0.0122, unfaded pending price -0.0190,
unfaded resident-context price -0.0050), and long-run work-share EMA variants
failed (-0.0016, -0.0030) because they were gated by `below`, which is 0
during the very bursts that set the tail.

Residual loss of the best ancestor.  P99 E2E at the s1.0 points (Conv
1.092/1.042, Tool 1.059/1.016) while P99 TTFT, TPOT and TBT at those same
points are within 1-3% of the bar.  No queueing loss and no single long stall,
yet a late last token: the lost time is SUSTAINED decode slow-down from
co-residency.

Diagnosis (why a count-based discount cannot see it).  A fully warm request is
invisible to every load signal here: `pending_prefill_tokens` counts only
UNCACHED tokens, and `frac = 0` zeroes the DYN 80/40 balance, the stall floor
and the convex guards.  The crowding discount is the only term that prices a
warm pile -- but it prices it in REQUEST COUNTS.  Decode cost is not per
request, it is per resident TOKEN: prefix sharing saves prefill, it does not
save decode attention, so eight co-resident 7k-token sequences make the replica
attend ~56k tokens per step instead of 7k.  Four stacked 20k-token multi-turn
conversations and four stacked 500-token ones therefore look identical to
`out_excess`, even though only the first ruins a long output's E2E.  That is
exactly the Conv-s1.0 signature, where the residual loss is largest.

Fix (one term).  Add a second, context-denominated excess inside the SAME
credit-bounded discount, with a weight bought entirely with spare capacity:

    kv_i       = kv_usage_i * kv_capacity_tokens          (resident context)
    spare      = (1/R) * sum_i max(0, 1 - outstanding_i / QUIET_REF)
    discount   = crowd_w * out_excess^2 / (mean_out + CROWD_REF)      (count)
               + KV_CROWD_SPARE_W * spare
                 * ((kv_i - mean_kv) / (mean_kv + KV_CROWD_REF))^2    (NEW)
    discount   = min(discount, CROWD_MAX + CROWD_OVERSHOOT * spare)
    cost_i    += crowd_gate * discount * lam_i * cached

  * EXACTLY ZERO AT SATURATION.  With spare = 0 (every replica busy -- all the
    s0.75 points) the new term's weight is 0 and crowd_gate = below ~ 0, so the
    policy is bit-identical to its ancestor and the large saturated wins
    (Conv-s0.75 ttft 0.853 / tbt 0.885, Synth-s0.75 tbt 0.863, Tool-s0.75 ttft
    0.944) are preserved.
  * NEAR-ZERO WHEN GENUINELY IDLE.  The KV_CROWD_REF damping means one 7k
    sequence against an empty fleet yields a discount of ~0.04: a two-deep warm
    pile on an otherwise quiet cluster is harmless and stays unpriced.
  * ACTIVE EXACTLY IN THE BURST-ON-A-DRAINED-FLEET REGIME (ramp ~ 1 but several
    replicas idle), where a replica holding 3-5x the fleet-average context pays
    a large withdrawal and the request is displaced onto an idle replica.  There
    the re-prefill runs in PARALLEL, costing almost no wall-clock TTFT, while
    the avoided co-residency is repaid at every decode step -- and the displaced
    request replicates the hot prefix, giving the next sibling a warm, quieter
    home.
  * MEAN-REFERENCED, CREDIT-BOUNDED, COLD-NEUTRAL.  Uniform stacking is free
    (excess = 0); the discount can at most make the router cache-indifferent
    (plus the spare-funded overshoot); and it is identically zero for cold
    requests, so the tuned cold-request balance (DYN 80/40 and the convex
    guards) is untouched -- the failure mode of every additive toll tried here.
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

# --- crowding discount on the prefix-cache credit: COUNT dimension ---------
CROWD_W = 0.25             # saturated slope: credit withdrawn per excess^2
CROWD_SPARE_W = 1.00       # extra slope, paid for by spare capacity
CROWD_REF = 3.0            # warm-up damping, in outstanding requests
CROWD_MAX = 1.00           # baseline cap: never more than the whole credit
CROWD_OVERSHOOT = 1.00     # extra cap headroom, paid for by spare capacity

# --- NEW: crowding discount, RESIDENT-CONTEXT dimension --------------------
KV_CROWD_SPARE_W = 1.00    # slope, bought entirely with spare capacity
KV_CROWD_REF = 30000.0     # warm-up damping, in resident context tokens

# --- spare-capacity gate for the crowding discount -------------------------
QUIET_REF = 2.0            # outstanding requests at which a replica is "busy"

# --- always-on convex tail shaping -----------------------------------------
STALL_ALWAYS_W = 0.020     # linear stall floor (uncached * co_decoders)
STALL_TAIL_W = 0.045       # convex P99-TBT guard
PENDING_TAIL_W = 0.025     # convex P99-TTFT guard
PENDING_REF = 6000.0

# --- below-knee convex prefill-queue guard (frac-scaled, gentle) -----------
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
        if sums[i] < 0:
            sums[i] = 0
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

    # Spare capacity: the fraction of the fleet that is empty or nearly empty.
    # Unlike `ramp` (a fleet AVERAGE, which one burst of long prompts
    # saturates), this sees that most replicas are still idle during a burst on
    # an otherwise drained fleet -- the regime in which replicating a hot
    # prefix is almost free, because the re-prefill runs in parallel.
    spare = 0.0
    for r in replicas:
        q = 1.0 - r.outstanding_requests / QUIET_REF
        if q > 0.0:
            spare += q
    spare /= num

    crowd_gate = below if below > spare else spare
    # Slope, cap and the resident-context dimension of the crowding discount
    # are all bought with spare capacity.  With spare = 0 the discount keeps
    # its shallow saturated slope, is bounded by the credit itself, and the
    # context dimension is switched off entirely; only when idle replicas
    # exist does it become steep enough -- and allowed to overshoot -- to
    # displace a fully warm request from a pile-up.
    crowd_w = CROWD_W + CROWD_SPARE_W * spare
    crowd_cap = CROWD_MAX + CROWD_OVERSHOOT * spare
    kv_crowd_w = KV_CROWD_SPARE_W * spare

    # Resident context per replica, in KV tokens.  This is the quantity each
    # co-resident sequence re-attends at EVERY decode step, and therefore the
    # correct unit for a decode-co-residency price: prefix sharing saves
    # prefill, not decode attention.
    kv_tokens = [r.kv_usage * r.kv_capacity_tokens for r in replicas]
    mean_kv = sum(kv_tokens) / num
    kv_crowd_norm = mean_kv + KV_CROWD_REF

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
    crowd_norm = mean_out + CROWD_REF

    attn_weight = ATTN_WEIGHT_MAX * ramp * tail_scale
    stall_weight = (STALL_FLOOR_W + STALL_KNEE_W) * ramp * tail_scale

    context = [kv_tokens[i] + replicas[i].pending_prefill_tokens
               for i in range(num)]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        if cached > n:
            cached = n
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

        # Crowding discount on the cache credit.  The credit lam_i * cached is
        # withdrawn convexly when this replica is more crowded than the fleet
        # average, in EITHER of two dimensions:
        #   (a) live-sequence COUNT      -- more concurrent decoders, and
        #   (b) RESIDENT CONTEXT TOKENS  -- more attention work per step.
        # (b) is the term a count-only discount is blind to: four stacked 20k
        # multi-turn conversations cost four times the per-step attention of
        # four stacked 500-token ones, yet have the same `out_excess`.  Its
        # weight is bought entirely with spare capacity, so it vanishes under
        # saturation (where stacking is the right call) and is damped near
        # idle (where a shallow warm pile is harmless).  Both dimensions share
        # one cap, so a prefix hit can never become a liability beyond the
        # spare-funded overshoot; both are zero for cold requests and for a
        # balanced fleet.
        if crowd_gate > 0.0 and cached > 0:
            discount = 0.0
            out_excess = r.outstanding_requests - mean_out
            if out_excess > 0.0:
                discount += crowd_w * out_excess * out_excess / crowd_norm
            if kv_crowd_w > 0.0:
                kv_excess = kv_tokens[i] - mean_kv
                if kv_excess > 0.0:
                    x = kv_excess / kv_crowd_norm
                    discount += kv_crowd_w * x * x
            if discount > crowd_cap:
                discount = crowd_cap
            if discount > 0.0:
                cost += crowd_gate * discount * lam_i * cached

        # below-knee Dynamo-style balance (optimum 80/40), scaled by frac:
        # a fully warm hit (frac = 0) reduces to the discount above, while
        # cold requests keep the full balance that wins saturated loads.
        if below > 0.0:
            cost += below * frac * (DYN_RUN_W * r.running
                                    + DYN_OUT_W * r.outstanding_requests)

        # convex P99-TTFT guard: only the above-mean pending queue (which
        # sets the longest prefill wait) is penalised.
        pend_excess = r.pending_prefill_tokens - mean_pending
        if pend_excess > 0.0:
            cost += (tail_scale * PENDING_TAIL_W
                     * pend_excess * pend_excess / pending_norm)

        # gentle below-knee convex prefill-queue nudge (frac-scaled)
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

    # Session stickiness only matters above the knee; below it,
    # cached_prefix_tokens already carries affinity and stickiness adds
    # avoidable queueing (and avoidable decode stacking).
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
