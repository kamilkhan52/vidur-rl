"""Absolute per-replica decode-batch pricing + cache-conditional stickiness.

    cost_j = load_j                                    # backlog this request waits behind
           + lambda_j * uncached_j                     # miss work, priced at system cost
           + DECODE_INT * uncached_j * running_j       # prefill chunks steal decode steps
           + DECODE_SLOT * batch_j                     # small linear decode-slot price
           + BATCH_QUAD * max(0, batch_j - KNEE)^2     # ABSOLUTE per-replica batch tail price
           + LIGHT_SPREAD * light * max(0, batch_j-1)^2# light-load spread term
           + absolute + relative imbalance guards
           + KV-preemption penalty (all hits + uncached footprint)

Change vs. the parent (single conceptual fix):

  * Hysteresis is applied ONLY when the sticky replica has a *strict* cache
    advantage over every other replica.  When several replicas are equally
    warm for this prompt (common with shared system prompts in tool traces),
    switching costs no re-prefill, so anchoring the session there only
    concentrates decode batches and inflates the P99-TPOT tail.  In that case
    we fall through to cost comparison / the quietest-warm tie-break and
    spread.  When the sticky replica is uniquely warm the behaviour is exactly
    the parent's, so below-knee cache locality (the TTFT/E2E win) is kept.
  * A light-load spread term on the future decode batch, gated to vanish once
    the cluster mean batch exceeds LIGHT_BATCH_REF, tightens TPOT in the
    light regime without touching the saturated regime where cache grouping
    is what produces the E2E/TTFT wins.
"""

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 1.5     # extra prompt-equivalents when decodes are resident
RUNNING_REF = 6.0           # decode count at which the occupancy floor saturates
MISS_EXTERNALITY = 1.0      # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 8.0            # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 160.0       # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 40000.0   # absolute tail guard (binds at light load)
REL_IMBALANCE = 20.0        # scale-free tail guard, in prompt-units

# --- decode-interference / tail terms --------------------------------------
INTERFERENCE = 0.05         # fractional miss-price surcharge per excess decode
DECODE_INT = 0.12           # cost per (uncached token x resident decode)
DECODE_SLOT = 20.0          # small linear price on the absolute future batch

# --- ABSOLUTE per-replica decode-batch tail price ---------------------------
BATCH_KNEE = 5.0            # a replica's own batch below this pays nothing
BATCH_QUAD = 15.0           # token-equivalent cost per (excess decode)^2

# --- light-load spread term -------------------------------------------------
LIGHT_BATCH_REF = 3.0       # term vanishes once mean future batch reaches this
LIGHT_SPREAD = 4000.0       # price for stacking decodes while the cluster is light

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.70        # KV occupancy where preemption risk starts
KV_PENALTY = 20000.0        # token-equivalent cost at a full KV cache
KV_ALL_PENALTY = 7000.0     # uniform component (hits still allocate output KV)
KV_TOKEN_SCALE = 6.0        # extra KV weight per full prompt of KV footprint
KV_OUTPUT_CONST = 512.0     # estimated output KV tokens (unknown to router)

# --- session handling ------------------------------------------------------
HYSTERESIS_FRAC = 0.05      # anti-flapping margin for a UNIQUELY warm replica
HYSTERESIS_MIN = 150.0

# --- future-decode awareness -----------------------------------------------
WAITING_WEIGHT = 0.8        # waiting requests count partially toward future batch


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    # ---- 1. price of a cache miss ----------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num

    batch = [r.running + WAITING_WEIGHT * r.waiting for r in replicas]
    mean_batch = sum(batch) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy

    congestion = mean_load / n_pos
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_load if mean_load > n_pos else n_pos

    # Light-load gate: 1 when the cluster is quiet, 0 once mean batch >= ref.
    light = 1.0 - mean_batch / LIGHT_BATCH_REF
    if light < 0.0:
        light = 0.0

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    active = request.active_prefix_tokens or (0,) * num
    eff_cached = [0] * num
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        if active[i] > cached:
            cached = active[i]
        eff_cached[i] = cached

        uncached = n - cached
        if uncached < 0:
            uncached = 0

        decode_excess = batch[i] - mean_batch
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)

        cost = load[i] + lam_i * uncached
        cost += DECODE_INT * uncached * r.running
        cost += DECODE_SLOT * batch[i]

        # ABSOLUTE per-replica convex tail price (the P99-TPOT lever).
        exc = batch[i] - BATCH_KNEE
        if exc > 0.0:
            cost += BATCH_QUAD * exc * exc

        # Light-load spread: discourage piling a second decode on a replica
        # while the whole cluster is still quiet.  Vanishes at the knee.
        ge = batch[i] - 1.0
        if ge > 0.0:
            cost += LIGHT_SPREAD * light * ge * ge

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        # Preemption / recompute risk.
        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            x2 = x * x
            footprint = 0.0
            if r.kv_capacity_tokens > 0:
                footprint = (uncached + KV_OUTPUT_CONST) / r.kv_capacity_tokens
            cost += KV_ALL_PENALTY * x2
            cost += KV_PENALTY * x2 * (1.0 + KV_TOKEN_SCALE * footprint)

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
    best = min(costs)

    # Hysteresis ONLY when the sticky replica is uniquely warm: then sticking
    # avoids a re-prefill.  If warmth is tied, switching is free (no
    # re-prefill), so anchoring would only concentrate decode batches and
    # inflate the TPOT tail -- fall through and spread instead.
    sticky = request.session_last_replica
    if 0 <= sticky < num:
        other_max = 0
        for j in range(num):
            if j != sticky and eff_cached[j] > other_max:
                other_max = eff_cached[j]
        if eff_cached[sticky] > other_max:
            margin = max(HYSTERESIS_MIN, HYSTERESIS_FRAC * n)
            if costs[sticky] <= best + margin:
                return sticky

    cands = [i for i, c in enumerate(costs) if c <= best + 1e-9]
    if len(cands) == 1:
        return cands[0]

    # Ties (typically warm/idle replicas) go to the warmest cache, then to the
    # smallest future decode batch, then to the shortest queue.
    warmest = max(eff_cached[i] for i in cands)
    cands = [i for i in cands if eff_cached[i] == warmest]
    if len(cands) == 1:
        return cands[0]

    quietest = min(batch[i] for i in cands)
    cands = [i for i in cands if batch[i] == quietest]
    if len(cands) == 1:
        return cands[0]

    fewest = min(replicas[i].outstanding_requests for i in cands)
    cands = [i for i in cands if replicas[i].outstanding_requests == fewest]
    return ctx.rng.choice(cands)
