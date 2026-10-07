"""Congestion-priced cache routing with a sustained-pressure gate,
pressure-scaled decode-interference pricing, attention-load pricing, and a
PREFILL-DUTY price on each replica's recent cache-miss inflow.

One currency: prefill-token-equivalents of delay.

    S      = sustained gate in [0,1]: time-weighted EMA (tau = 30 s) of mean
             queued prefill per replica, in units of an EMA prompt length
    g      = min(1, instantaneous mean pending per replica / this prompt)
    lambda = 1 + S * (g * DECODE_MISS_PRICE * occupancy + externality)  (capped)
    share_j = 1 + slope(S) * relative projected batch excess    (clamped)
    flow_j = exponentially decayed (tau = 15 s) uncached tokens routed to j
    cost_j = load_j + lambda * share_j * uncached_j
           + DECODE_TAX * uncached_j * relative running excess   (always on)
           + imbalance / batch / context guards
           + linear context price (own attention slowdown)
           + linear DUTY price on relative flow_j (own prefill interference)
           + projected kv_pressure_j

Why a duty price.  With chunked prefill, a decode step that shares an
iteration with a prefill chunk is several times slower than a pure decode
step.  A request's TPOT is therefore set mostly by how much miss prefill
passes through its replica while it decodes.  Cache-affine pricing at the
knee sends hits to warm replicas and concentrates misses on a few cold
ones.  Their pending queue drains fast, so no instantaneous signal shows it,
but decodes there set the P99 TPOT.  The recent miss inflow (decayed over
about one decode lifetime) predicts that interference.  Priced linearly and
two-sided, it steers comparable choices toward low-duty replicas without
overriding large cache savings.
"""

import math

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0
RUNNING_REF = 4.0
QUEUE_REF = 1.0
MISS_EXTERNALITY = 0.8
LAMBDA_MAX = 7.0

# --- sustained-pressure gate ------------------------------------------------
PRESSURE_TAU = 30.0
PROMPT_ALPHA = 0.05
S_LO = 0.5
S_HI = 2.0

# --- decode interference of a miss (relative, two-sided, pressure-scaled) ---
SHARE = 0.5
SHARE_GATED = 1.0
SHARE_MIN = 0.5
SHARE_MIN_DROP = 0.15
SHARE_MAX = 2.5
SHARE_MAX_GATED = 1.0
DECODE_TAX = 0.25

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0
IMBALANCE_SCALE = 30000.0
REL_IMBALANCE = 30.0

# --- projected batch guard (in miss units) ----------------------------------
BATCH_GUARD = 0.5

# --- projected context (attention load), in miss units ----------------------
CTX_GUARD_BASE = 0.25
CTX_GUARD_GATED = 1.5
CTX_PRICE_BASE = 0.08
CTX_PRICE_GATED = 0.35

# --- prefill duty (recent miss inflow), in miss units -----------------------
FLOW_TAU = 15.0            # ~ one decode lifetime (seconds)
DUTY_PRICE_BASE = 0.05     # always on (small: protects below-knee TTFT)
DUTY_PRICE_GATED = 0.30    # added in proportion to sustained pressure S
DUTY_REL_MIN = -1.0
DUTY_REL_MAX = 2.0

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0
KV_DEFAULT_CAPACITY = 122752.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def sustained_gate(state, now, mean_pending, n_pos):
    """Time-weighted EMA of the queued backlog -> gate in [0, 1]."""
    if "p_ema" not in state:
        state["p_ema"] = 0.0
        state["n_ema"] = float(n_pos)
        state["t_last"] = now
    dt = now - state["t_last"]
    if dt > 0.0:
        w = math.exp(-dt / PRESSURE_TAU)
        state["p_ema"] = w * state["p_ema"] + (1.0 - w) * mean_pending
        state["t_last"] = now
    state["n_ema"] += PROMPT_ALPHA * (n_pos - state["n_ema"])

    queued_prompts = state["p_ema"] / max(1.0, state["n_ema"])
    s = (queued_prompts - S_LO) / (S_HI - S_LO)
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    return s


def decayed_flow(state, now, num):
    """Per-replica exponentially decayed count of routed uncached tokens."""
    flow = state.get("flow")
    if flow is None or len(flow) != num:
        flow = [0.0] * num
        state["flow"] = flow
        state["f_last"] = now
    dt = now - state["f_last"]
    if dt > 0.0:
        w = math.exp(-dt / FLOW_TAU)
        for i in range(num):
            flow[i] *= w
        state["f_last"] = now
    return flow


def select(request, replicas, ctx, costs, batch):
    num = len(replicas)
    n = request.num_prefill_tokens
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

    quietest = min(batch[i] for i in cands)
    cands = [i for i in cands if batch[i] == quietest]
    if len(cands) == 1:
        return cands[0]

    fewest = min(replicas[i].outstanding_requests for i in cands)
    cands = [i for i in cands if replicas[i].outstanding_requests == fewest]
    return ctx.rng.choice(cands)


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    # ---- 1. system price of a cache miss ---------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / num

    S = sustained_gate(ctx.state, ctx.now, mean_pending, n_pos)
    flow = decayed_flow(ctx.state, ctx.now, num)

    g = (mean_pending / n_pos) / QUEUE_REF
    if g > 1.0:
        g = 1.0

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    inflation = DECODE_MISS_PRICE * occupancy * g
    inflation += MISS_EXTERNALITY * max(0.0, mean_load / n_pos - 1.0)
    lam = 1.0 + S * inflation
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_load if mean_load > n_pos else n_pos

    batch = [r.running + r.waiting for r in replicas]
    mean_batch = sum(batch) / num
    batch_ref = mean_batch + 1.0
    running_ref = mean_running + 1.0

    share_slope = SHARE + SHARE_GATED * S
    share_lo = SHARE_MIN - SHARE_MIN_DROP * S
    share_hi = SHARE_MAX + SHARE_MAX_GATED * S

    caps = [r.kv_capacity_tokens if r.kv_capacity_tokens > 0 else KV_DEFAULT_CAPACITY
            for r in replicas]
    ctx_tok = [replicas[i].kv_usage * caps[i] + replicas[i].pending_prefill_tokens
               for i in range(num)]
    mean_ctx = sum(ctx_tok) / num
    ctx_ref = mean_ctx + n_pos
    ctx_guard_w = (CTX_GUARD_BASE + CTX_GUARD_GATED * S) * lam * n_pos
    ctx_price_w = (CTX_PRICE_BASE + CTX_PRICE_GATED * S) * lam * n_pos

    # Prefill duty: recent miss inflow relative to the cluster mean.
    mean_flow = sum(flow) / num
    flow_ref = mean_flow + n_pos
    duty_w = (DUTY_PRICE_BASE + DUTY_PRICE_GATED * S) * lam * n_pos

    # ---- 2. effective cached prefix (exact in-flight view) ----------------
    inflight = request.inflight_prefix_tokens or (0,) * num

    # ---- 3. score every replica ------------------------------------------
    costs = []
    uncached_list = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0
        uncached_list.append(uncached)

        rel_batch = (batch[i] - mean_batch) / batch_ref
        share = 1.0 + share_slope * rel_batch
        if share < share_lo:
            share = share_lo
        elif share > share_hi:
            share = share_hi

        cost = load[i] + lam * share * uncached

        rel_running = (r.running - mean_running) / running_ref
        if rel_running > 0.0:
            cost += DECODE_TAX * uncached * rel_running

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        if rel_batch > 0.0:
            cost += BATCH_GUARD * lam * n_pos * rel_batch * rel_batch

        rel_ctx = (ctx_tok[i] + n - mean_ctx - n / num) / ctx_ref
        cost += ctx_price_w * rel_ctx
        if rel_ctx > 0.0:
            cost += ctx_guard_w * rel_ctx * rel_ctx

        # Prefill duty our decode will share steps with (two-sided, bounded).
        rel_duty = (flow[i] - mean_flow) / flow_ref
        if rel_duty < DUTY_REL_MIN:
            rel_duty = DUTY_REL_MIN
        elif rel_duty > DUTY_REL_MAX:
            rel_duty = DUTY_REL_MAX
        cost += duty_w * rel_duty

        projected = r.kv_usage + uncached / caps[i]
        if projected > KV_SOFT_LIMIT:
            x = (projected - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 4. choose and record this request's prefill inflow ---------------
    idx = select(request, replicas, ctx, costs, batch)
    flow[idx] += uncached_list[idx]
    return idx
