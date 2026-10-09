"""Load-gated min-TTFT with light-load relief for the always-on decode terms.

Hypothesis: the parent's binding constraint is its worst scenario, Tool s1.0
P99 TTFT (ratio 1.09-1.12), the regime where the bar's production routers are
strongest.  There the fleet has spare capacity (ramp = 0), so the decision
reduces to `min-TTFT core + DECODE_BASE_W * running + a miss-gated stall
term` -- a needless deviation from pure min-estimated-TTFT, because decode
interference is negligible when there is spare capacity.  These terms pull
long, loosely-cached (frac ~ 1) requests off their warmest replica, adding
recompute to TTFT.  We fade the always-on decode price (and the miss-gated
stall floor) toward zero as the fleet becomes uncongested, but only for
requests that actually carry a large cache miss (frac high): well-cached
requests keep the full decode price, so the Synth TPOT behaviour and the
whole saturation-knee regime (ramp = 1) are byte-for-byte unchanged.
"""

from collections import deque

# --- congestion gauge / ramp ------------------------------------------------
DECODE_REF = 175.0         # reference weight used only to gauge congestion
ATTN_RAMP_START = 0.5      # congestion (own-prompt units) where extras start
ATTN_RAMP_WIDTH = 1.5      # congestion span over which extras ramp to full

# --- pricing of a cache miss ------------------------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
IMBALANCE_SCALE = 1e30     # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 0.5        # scale-free tail guard, in prompt-units
PREFILL_TAIL = 1.5         # convex price on this replica's own prefill backlog
PREFILL_REF_FRAC = 1.0     # tail normalization = (1 + FRAC) * own prompt size
WAIT_TAIL = 0.4            # convex price on excess queued requests
WAIT_REF = 1.0             # softener so a single queued request is not overpriced

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- attended-context term (private + external) -----------------------------
ATTN_WEIGHT_MAX = 0.4      # token-equivalents per attended context token (saturated)

# --- decode-stall term (miss-only, RELATIVE to the fleet mean) --------------
STALL_FLOOR_W = 0.06       # always-on weight on excess concurrent decoders
STALL_KNEE_W = 0.10        # extra weight at full congestion ramp

# --- cache-aware per-request fade of the decode-side prices -----------------
MISS_FLOOR = 0.30          # decode prices retained for a fully cached request

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0
HYST_LO = 0.0              # no hysteresis at zero congestion

# --- absolute decode-load tie-breaker (always on) --------------------------
DECODE_BASE_W = 20.0

# --- NEW: light-load relief for the always-on decode terms ------------------
LIGHT_CONG = 0.25          # congestion below which the fleet is "uncongested"
LOWLOAD_DECODE_RELIEF = 0.7  # fraction of DECODE_BASE_W removed at light load
LOWLOAD_STALL_RELIEF = 0.5   # fraction of STALL_FLOOR_W removed at light load


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


def _effective_cached(request, num):
    """Best estimate of the leading prefix already usable on each replica."""
    cache = request.cached_prefix_tokens
    inflight = request.inflight_prefix_tokens or (0,) * num
    active = request.active_prefix_tokens or (0,) * num
    eff = []
    for i in range(num):
        c = cache[i]
        if inflight[i] > c:
            c = inflight[i]
        if active[i] > c:
            c = active[i]
        eff.append(c)
    return eff


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    queues, attended = _attended_context(replicas, ctx.state)

    # ---- 1. congestion gauge and load ramp --------------------------------
    work = [r.pending_prefill_tokens + DECODE_REF * r.running for r in replicas]
    mean_work = sum(work) / num
    congestion = mean_work / n_pos

    ramp = (congestion - ATTN_RAMP_START) / ATTN_RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0

    # Light-load indicator: 1 when the fleet is comfortably below the knee,
    # 0 once congestion reaches LIGHT_CONG.  Used to relieve the always-on
    # decode terms exactly where spare capacity makes them unnecessary.
    light = (LIGHT_CONG - congestion) / LIGHT_CONG
    if light < 0.0:
        light = 0.0
    elif light > 1.0:
        light = 1.0

    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num
    mean_pend = sum(r.pending_prefill_tokens for r in replicas) / num
    mean_wait = sum(r.waiting for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF
    pend_norm = (1.0 + PREFILL_REF_FRAC) * n_pos

    attn_weight = ATTN_WEIGHT_MAX * ramp
    # Miss-gated stall floor, relieved at light load (only ever affects
    # requests that actually carry a miss, so well-cached traffic is safe).
    stall_floor = STALL_FLOOR_W * (1.0 - LOWLOAD_STALL_RELIEF * light)
    stall_weight = stall_floor + STALL_KNEE_W * ramp

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    mean_load = mean_pend + DECODE_WEIGHT * mean_running
    load_ref = mean_load if mean_load > n_pos else n_pos

    # ---- 2. effective cache hits and score every replica ------------------
    eff_cached = _effective_cached(request, num)
    costs = []
    for i, r in enumerate(replicas):
        cached = eff_cached[i]
        unc = n - cached
        if unc < 0:
            unc = 0
        frac = unc / n_pos
        f = MISS_FLOOR + (1.0 - MISS_FLOOR) * frac

        # min-TTFT core (always on): queued prefill + this request's miss.
        mt = r.pending_prefill_tokens + unc

        # ---- congestion extras, faded in by the load ramp ----------------
        extra = DECODE_WEIGHT * f * r.running
        extra += (lam - 1.0) * unc

        pend_excess = r.pending_prefill_tokens - mean_pend
        if pend_excess > 0.0:
            rel_p = pend_excess / pend_norm
            extra += PREFILL_TAIL * n_pos * rel_p * rel_p

        wait_excess = r.waiting - mean_wait
        if wait_excess > 0.0:
            rel_w = wait_excess / (mean_wait + WAIT_REF)
            extra += WAIT_TAIL * n_pos * rel_w * rel_w

        load_i = r.pending_prefill_tokens + DECODE_WEIGHT * r.running
        excess = load_i - mean_load
        if excess > 0.0:
            extra += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            extra += REL_IMBALANCE * n_pos * rel * rel

        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            extra += BATCH_PRICE * f * n_pos * batch_excess * batch_excess / batch_norm

        ctx_excess = context[i] + unc - mean_ctx
        if ctx_excess > 0.0:
            extra += CONTEXT_PRICE * f * ctx_excess * ctx_excess / ctx_norm

        if attn_weight > 0.0:
            co_resident = len(queues[i])
            extra += attn_weight * f * (attended[i] + n * co_resident)

        cost = mt + ramp * extra

        # ---- absolute decode-load tie-breaker ----------------------------
        # Relieved at light load, but only in proportion to this request's
        # miss: a well-cached request still pays the full price (its own
        # prefill is trivial, so avoiding busy decoders is what protects it).
        decode_base = DECODE_BASE_W * (1.0 - LOWLOAD_DECODE_RELIEF * light * frac)
        cost += decode_base * r.running

        # ---- decode-stall price: RELATIVE to the fleet mean --------------
        if unc > 0:
            dec_excess = r.running - mean_running
            if dec_excess > 0.0:
                cost += stall_weight * unc * dec_excess

        # Preemption risk is an always-on safety guard.
        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n, eff_cached, ramp)
    queues[pick].append(n)
    attended[pick] += n
    return pick


def _select(request, replicas, ctx, costs, num, n, eff_cached, ramp):
    best = min(costs)

    sticky = request.session_last_replica
    if 0 <= sticky < num:
        scale = HYST_LO + (1.0 - HYST_LO) * ramp
        margin = max(HYSTERESIS_MIN, HYSTERESIS_FRAC * n) * scale
        if costs[sticky] <= best + margin:
            return sticky

    cands = [i for i, c in enumerate(costs) if c <= best + 1e-9]
    if len(cands) == 1:
        return cands[0]

    warmest = max(eff_cached[i] for i in cands)
    cands = [i for i in cands if eff_cached[i] == warmest]
    if len(cands) == 1:
        return cands[0]

    quietest = min(replicas[i].running for i in cands)
    cands = [i for i in cands if replicas[i].running == quietest]
    if len(cands) == 1:
        return cands[0]

    fewest = min(replicas[i].outstanding_requests for i in cands)
    cands = [i for i in cands if replicas[i].outstanding_requests == fewest]
    return ctx.rng.choice(cands)
