"""Congestion-priced, tail-aware cache routing with an ATTENTION-AWARE,
DECODE-AWARE Mooncake-style TTFT admissibility band and an attention-aware
decode-stall price.

One currency: prefill-token-equivalents of delay (linear-token units).

    cost_j = load_j                        # weighted backlog this request waits behind
           + lambda_j * work_j             # weighted work it adds, priced at system cost
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess resident KV tokens
           + attended-context price        # private + external decode cost (ramped)
           + decode-stall price            # work_j * co-decoders_j
           + kv_pressure_j                 # preemption risk

Admissibility: only replicas whose estimated TTFT
    est_j = (queued_work_j + work_j) * (1 + resident_kv_j / RESIDENT_REF)
lies within a (relative) slack of the best estimate are eligible.

A prefill token attending to c context tokens costs ~(1 + c / ATTN_CTX)
linear-token equivalents, so work_j = uncached_j * factor(cached_j, uncached_j).

Change: with chunked prefill every engine step that carries a prefill chunk
also carries the decode batch, whose attention streams the whole resident KV
cache (~128 KB/token for Llama-3-8B GQA).  A replica holding many resident
decode tokens therefore prefills more slowly per token than an idle one with
the same queue.  The Mooncake-style estimate ignores this; the band now scales
the prefill-work estimate by a per-step slowdown 1 + resident / RESIDENT_REF
(roofline: a ~2k-token chunk ~110 ms vs ~22 ms to read 350k KV tokens).  This
only reshapes the band (below the knee); it dissolves at the knee as before.
"""

import math
from collections import deque

# --- prefill work model ----------------------------------------------------
ATTN_CTX = 26000.0         # context length at which attention doubles per-token cost
RESIDENT_REF = 1.0e6       # resident KV tokens that double the prefill step time

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the guard when little context is resident

# --- attended-context term (private + external) -----------------------------
ATTN_WEIGHT_MAX = 0.4      # token-equivalents per attended context token (saturated)
ATTN_RAMP_START = 0.5      # congestion (prompt units) where the price starts
ATTN_RAMP_WIDTH = 1.5      # congestion span over which it ramps to full weight

# --- decode-stall term (miss-only, attention-weighted) -----------------------
STALL_FLOOR_W = 0.042      # always-on weight
STALL_KNEE_W = 0.10        # extra weight at full congestion ramp (TTFT has slack)

# --- TTFT admissibility band -----------------------------------------------
TTFT_SLACK = 0.5           # prompt-units of slack at band_ramp = 0.5
TTFT_TIE = 256.0           # token-equivalents treated as a TTFT tie
REL_SLACK = 0.05           # fraction of the best TTFT estimate treated as a tie
PROMPT_REF = 7000.0        # typical prompt: cluster-scale unit of congestion
BAND_TAU = 120.0           # seconds: smoothing of the band's congestion signal
BAND_RAMP_START = 3.0      # smoothed queued prefill (prompt units) where band widens
BAND_RAMP_WIDTH = 8.0      # span over which it widens to dissolution
EXCLUDED = 1e18            # cost added to replicas outside the band

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def _work_factor(cached, uncached):
    """Linear-token equivalents per uncached prefill token: each token
    attends on average to the cached prefix plus half the new suffix."""
    return 1.0 + (cached + 0.5 * uncached) / ATTN_CTX


def _queued_prefill_work(replicas, state):
    """Attention-weighted prefill backlog per replica.

    Each replica keeps a FIFO of (uncached_tokens, work_factor) for requests
    routed there.  Prefill is FCFS, so the replica's pending_prefill_tokens
    are the newest entries; older entries are trimmed and a partially
    prefilled head entry is counted pro rata.  Untracked excess counts at
    factor 1."""
    num = len(replicas)
    queues = state.get("pf_queues")
    if queues is None or len(queues) != num:
        queues = [deque() for _ in replicas]
        state["pf_queues"] = queues
        state["pf_sums"] = [0] * num
        state["pf_wsums"] = [0.0] * num
    sums = state["pf_sums"]
    wsums = state["pf_wsums"]
    work = []
    for i, r in enumerate(replicas):
        q = queues[i]
        p = r.pending_prefill_tokens
        while q and sums[i] - q[0][0] >= p:
            t, f = q.popleft()
            sums[i] -= t
            wsums[i] -= t * f
        if not q:
            sums[i] = 0
            wsums[i] = 0.0
        if sums[i] >= p:
            w = wsums[i] - (sums[i] - p) * q[0][1] if q else 0.0
        else:
            w = wsums[i] + (p - sums[i])
        work.append(w if w > 0.0 else 0.0)
    return queues, sums, wsums, work


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


def _linear_ramp(x, start, width):
    """0 below start, 1 above start + width (linear in between)."""
    y = (x - start) / width
    if y < 0.0:
        return 0.0
    if y > 1.0:
        return 1.0
    return y


def _sustained_prefill_congestion(replicas, state, now):
    """EMA (time constant BAND_TAU) of mean queued prefill, in prompt units."""
    num = len(replicas)
    x = sum(r.pending_prefill_tokens for r in replicas) / num / PROMPT_REF
    ema = state.get("band_ema")
    last = state.get("band_t")
    if ema is None or last is None:
        ema = 0.0
        last = now
    dt = now - last
    if dt > 0.0:
        a = 1.0 - math.exp(-dt / BAND_TAU)
        ema += a * (x - ema)
        last = now
    state["band_ema"] = ema
    state["band_t"] = last
    return ema


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    queues, attended = _attended_context(replicas, ctx.state)
    pf_queues, pf_sums, pf_wsums, queued_work = _queued_prefill_work(
        replicas, ctx.state)

    # ---- 1. price of a cache miss ----------------------------------------
    load = [queued_work[i] + DECODE_WEIGHT * r.running
            for i, r in enumerate(replicas)]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    congestion = mean_load / n_pos
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF

    # Pricing ramp: instantaneous congestion in own-prompt units.
    ramp = _linear_ramp(congestion, ATTN_RAMP_START, ATTN_RAMP_WIDTH)
    attn_weight = ATTN_WEIGHT_MAX * ramp
    stall_weight = STALL_FLOOR_W + STALL_KNEE_W * ramp

    # Band ramp: sustained queued prefill (decodes excluded, bursts smoothed).
    sustained = _sustained_prefill_congestion(replicas, ctx.state, ctx.now)
    band_ramp = _linear_ramp(sustained, BAND_RAMP_START, BAND_RAMP_WIDTH)

    # Resident KV tokens (decode attention read every step).
    resident = [r.kv_usage * r.kv_capacity_tokens for r in replicas]
    # Resident KV tokens plus prompts queued for prefill.
    context = [resident[i] + r.pending_prefill_tokens
               for i, r in enumerate(replicas)]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    costs = []
    est_ttft = []
    uncached_list = []
    factor_list = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0
        if cached > n:
            cached = n
        f = _work_factor(cached, uncached)
        own_work = uncached * f
        uncached_list.append(uncached)
        factor_list.append(f)

        # Attention- and decode-aware TTFT estimate: prefill work ahead of and
        # including ours, slowed by the decode batch's KV reads in every step.
        step_slowdown = 1.0 + resident[i] / RESIDENT_REF
        est_ttft.append((queued_work[i] + own_work) * step_slowdown)

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)

        cost = load[i] + lam_i * own_work

        # Decode-batch price: sequences this request will share steps with.
        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            cost += BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        # Decode-context price on resident (unique) KV tokens.
        ctx_excess = context[i] + uncached - mean_ctx
        if ctx_excess > 0.0:
            cost += CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

        # Attended-context price = marginal social decode cost.
        if attn_weight > 0.0:
            co_resident = len(queues[i])
            cost += attn_weight * (attended[i] + n * co_resident)

        # Decode-stall price: each prefill chunk lengthens the step of every
        # co-resident decode in proportion to its attention-weighted work.
        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += stall_weight * own_work * co_decoders

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. TTFT admissibility band: a relative tie (small fraction of the
    #         soonest start) when congestion is transient, widening with
    #         sustained prefill backlog, dissolved at the knee.
    if band_ramp < 1.0:
        best_est = min(est_ttft)
        slack = (TTFT_TIE + REL_SLACK * best_est
                 + TTFT_SLACK * n_pos * band_ramp / (1.0 - band_ramp))
        for i in range(num):
            if est_ttft[i] > best_est + slack:
                costs[i] += EXCLUDED

    # ---- 4. choose and record --------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n)
    queues[pick].append(n)
    attended[pick] += n
    u = uncached_list[pick]
    if u > 0:
        f = factor_list[pick]
        pf_queues[pick].append((u, f))
        pf_sums[pick] += u
        pf_wsums[pick] += u * f
    return pick


def _select(request, replicas, ctx, costs, num, n):
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
