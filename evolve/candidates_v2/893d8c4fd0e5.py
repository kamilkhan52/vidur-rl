"""Congestion-priced, tail-aware cache routing with a prefill-rate (heat) price.

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + absolute / relative imbalance guards
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess resident KV tokens
           + prefill-heat price (NEW)      # convex in excess recent prefill rate
           + kv_pressure_j                 # preemption risk

Change with respect to the parent: under chunked prefill a decoding sequence's
TPOT is set by how often its replica's steps carry a prefill chunk, i.e. by the
replica's prefill-token *rate*, largely independent of how many sequences share
the replica.  The parent discounted misses on replicas with few outstanding
sequences, turning them into miss sinks whose few decoders then dominated the
P99 TPOT at the knee.  (1) The outstanding-count externality is now a surcharge
only (no discount).  (2) Each replica carries an exponentially decayed sum of
uncached tokens routed to it (tau ~ one decode phase); its excess over the mean
(including our own uncached tokens) is priced convexly, gated by real prefill
congestion so the light-load (min-TTFT-like) behaviour is preserved.
"""

import math

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- TPOT externality of a miss (surcharge only) ----------------------------
TPOT_EXT = 0.6             # miss-price increase per relative excess of outstanding

# --- prefill-heat (TPOT) price (NEW) ----------------------------------------
HEAT_TAU = 15.0            # seconds; ~ duration of a decode phase
HEAT_PRICE = 2.0           # weight of excess^2 / (mean_heat + prompt) at full gate
HEAT_GATE_MIN = 0.2        # residual weight when there is no prefill backlog

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 30000.0  # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 30.0       # scale-free tail guard, in prompt-units

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guards when the cluster is nearly idle

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def decayed_heat(ctx, num):
    """Per-replica exponentially decayed sum of routed uncached prompt tokens."""
    st = ctx.state
    heat = st.get("heat")
    if heat is None or len(heat) != num:
        heat = [0.0] * num
        st["heat"] = heat
        st["heat_t"] = ctx.now
    dt = ctx.now - st.get("heat_t", ctx.now)
    if dt > 0:
        f = math.exp(-dt / HEAT_TAU)
        for i in range(num):
            heat[i] *= f
        st["heat_t"] = ctx.now
    return heat


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1
    heat = decayed_heat(ctx, num)

    # ---- 1. price of a cache miss ----------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num
    mean_heat = sum(heat) / num

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

    # Heat price is gated by real prefill backlog (in units of this prompt).
    p_cong = mean_pending / n_pos
    heat_gate = p_cong if p_cong < 1.0 else 1.0
    if heat_gate < HEAT_GATE_MIN:
        heat_gate = HEAT_GATE_MIN
    heat_w = HEAT_PRICE * heat_gate
    heat_norm = mean_heat + n_pos

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
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

        # Surcharge-only TPOT externality: crowded replicas pay more per miss,
        # quiet replicas are NOT discounted (avoids miss-sink replicas).
        batch_excess = r.outstanding_requests - mean_out
        ext = 1.0
        if batch_excess > 0.0:
            ext += TPOT_EXT * batch_excess / batch_norm
        cost = load[i] + lam * ext * uncached

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        if batch_excess > 0.0:
            cost += BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        ctx_excess = context[i] + uncached - mean_ctx
        if ctx_excess > 0.0:
            cost += CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

        # Prefill-heat price: recent prefill rate this replica's decoders feel,
        # including the uncached tokens we would add.
        heat_excess = heat[i] + uncached - mean_heat
        if heat_excess > 0.0:
            cost += heat_w * heat_excess * heat_excess / heat_norm

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
    pick = select(request, replicas, ctx, costs, n, num)
    heat[pick] += uncached_list[pick]
    return pick


def select(request, replicas, ctx, costs, n, num):
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
