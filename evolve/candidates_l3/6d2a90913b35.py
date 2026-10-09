"""Parent policy with ALL congestion extras gated by SUSTAINED load.

Hypothesis: the s1.0 P99-TTFT deficit (worst scenario: Tool s1.0) comes from
the congestion extras that the instantaneous own-prompt-unit ramp switches on
during every transient burst (decode-weight, prefill/wait tails, imbalance,
batch, context, attended-context, stall-knee).  Below the knee these pull
requests away from the soonest-prefill replica, which is exactly what the
Mooncake min-TTFT router optimises.  The parent only gated the cache-miss
surcharge; we now scale every ramp-faded extra by the same time-based EMA of
mean queued prefill (with a small floor), so bursts at s1.0 are routed by
min-TTFT while sustained knee backlog (s0.75) restores full parent pricing.
"""

import math
from collections import deque

# --- congestion gauge / ramp ------------------------------------------------
DECODE_REF = 175.0
ATTN_RAMP_START = 0.5
ATTN_RAMP_WIDTH = 1.5

# --- sustained-load gate ------------------------------------------------------
PROMPT_REF = 7000.0        # typical prompt: fixed unit of cluster backlog
GATE_TAU = 60.0            # seconds: EMA time constant of queued prefill
GATE_START = 1.0           # sustained queued prefill (prompts/replica) where gate opens
GATE_WIDTH = 2.0           # span over which the gate opens fully
EXTRA_FLOOR = 0.3          # fraction of congestion extras kept under bursts

# --- pricing of a cache miss ------------------------------------------------
DECODE_WEIGHT = 175.0
DECODE_MISS_PRICE = 2.0
RUNNING_REF = 4.0
MISS_EXTERNALITY = 0.8
LAMBDA_MAX = 7.0

# --- queueing delay terms (token-equivalents) ------------------------------
IMBALANCE_SCALE = 1e30
REL_IMBALANCE = 0.5
PREFILL_TAIL = 1.5
PREFILL_REF_FRAC = 1.0
WAIT_TAIL = 0.4
WAIT_REF = 1.0

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0
BATCH_REF = 4.0

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5
KV_REF = 16000.0

# --- attended-context term --------------------------------------------------
ATTN_WEIGHT_MAX = 0.4

# --- decode-stall term (miss-only, RELATIVE to the fleet mean) --------------
STALL_FLOOR_W = 0.06
STALL_KNEE_W = 0.10

# --- cache-aware per-request fade of the decode-side prices -----------------
MISS_FLOOR = 0.30

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0
HYST_LO = 0.0

# --- absolute decode-load tie-breaker (always on) --------------------------
DECODE_BASE_W = 20.0


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


def _sustained_gate(replicas, state, now):
    """Gate in [0,1] from a time-based EMA of mean queued prefill (prompts)."""
    num = len(replicas)
    x = sum(r.pending_prefill_tokens for r in replicas) / num / PROMPT_REF
    ema = state.get("gate_ema")
    last = state.get("gate_t")
    if ema is None or last is None:
        ema = 0.0
        last = now
    dt = now - last
    if dt > 0.0:
        a = 1.0 - math.exp(-dt / GATE_TAU)
        ema += a * (x - ema)
        last = now
    state["gate_ema"] = ema
    state["gate_t"] = last
    g = (ema - GATE_START) / GATE_WIDTH
    if g < 0.0:
        return 0.0
    if g > 1.0:
        return 1.0
    return g


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
    gate = _sustained_gate(replicas, ctx.state, ctx.now)

    # ---- 1. congestion gauge and load ramp --------------------------------
    work = [r.pending_prefill_tokens + DECODE_REF * r.running for r in replicas]
    mean_work = sum(work) / num
    congestion = mean_work / n_pos

    ramp = (congestion - ATTN_RAMP_START) / ATTN_RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0

    # Congestion extras only at full strength under SUSTAINED backlog.
    extra_w = ramp * (EXTRA_FLOOR + (1.0 - EXTRA_FLOOR) * gate)

    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num
    mean_pend = sum(r.pending_prefill_tokens for r in replicas) / num
    mean_wait = sum(r.waiting for r in replicas) / num

    # Cache-miss surcharge: only under sustained backlog.
    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    surcharge = DECODE_MISS_PRICE * occupancy
    surcharge += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    lam = 1.0 + gate * surcharge
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF
    pend_norm = (1.0 + PREFILL_REF_FRAC) * n_pos

    attn_weight = ATTN_WEIGHT_MAX * ramp
    stall_weight = STALL_FLOOR_W + STALL_KNEE_W * extra_w

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

        # ---- congestion extras ------------------------------------------
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

        cost = mt + extra_w * extra

        # ---- absolute decode-load tie-breaker (always on) ----------------
        cost += DECODE_BASE_W * r.running

        # ---- decode-stall price: RELATIVE to the fleet mean ---------------
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
