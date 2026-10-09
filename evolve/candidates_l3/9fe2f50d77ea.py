"""TTFT-feasible, congestion-gated, tail-aware cache routing with
depth-weighted decode-stall pricing.

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                         # backlog this request waits behind
           + lambda_j * uncached_j          # work it adds, priced at system cost
           + g * decode-batch price         # convex in excess outstanding requests
           + g * decode-context price       # convex in excess resident KV tokens
           + ramp * attended-context price  # private + external decode cost
           + depth * decode-stall price     # uncached_j * co-decoders_j
           + kv_pressure_j                  # preemption risk

Change vs parent: the decode-stall price is scaled by the request's prompt
length relative to the running-average prompt.  A prefill chunk's step time
grows with its depth in the prompt (it attends to every earlier token), so
only chunks of LONG prompts produce decode gaps that reach the TBT P99.
Short prompts' chunks are cheap; charging them a stall price only drags them
off their best-TTFT replica.  Depth weighting therefore lets short prompts
follow the TTFT estimate and steers long misses to replicas with few
co-decoders, while the TTFT feasibility filter bounds the delay any request
may trade away.

TTFT feasibility filter: each replica gets a Mooncake-style TTFT estimate
est_j = pending_prefill_j + uncached_j (in-flight prefixes counted as cached).
Only replicas with est_j <= min_est + slack may be chosen.  The slack widens
(and the filter fades out) with SUSTAINED cluster congestion.
"""

import math
from collections import deque

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0
RUNNING_REF = 4.0
MISS_EXTERNALITY = 0.8
LAMBDA_MAX = 7.0

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0
BATCH_REF = 4.0
INTERFERENCE = 0.05

# --- decode-context term ---------------------------------------------------
CONTEXT_PRICE = 1.5
KV_REF = 16000.0

# --- congestion ramp (per request, own-prompt units) -------------------------
ATTN_WEIGHT_MAX = 0.4
ATTN_RAMP_START = 0.5
ATTN_RAMP_WIDTH = 1.5
G_FLOOR = 0.3

# --- decode-stall term (miss-only) -----------------------------------------
STALL_FLOOR_W = 0.042
STALL_KNEE_W = 0.10
DEPTH_MIN = 0.25           # stall-price multiplier floor (short prompts)
DEPTH_MAX = 3.0            # stall-price multiplier cap (very long prompts)

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0

# --- TTFT feasibility filter -------------------------------------------------
SLACK_MIN = 1024.0         # tokens of TTFT estimate we may trade away when quiet
SLACK_REL = 0.15           # fraction of the best TTFT estimate we may trade away
CLUSTER_RAMP_START = 2.0   # sustained cluster pending (avg-prompt units) where slack widens
CLUSTER_RAMP_WIDTH = 6.0   # span over which the filter fades out entirely
PROMPT_EMA_ALPHA = 0.02    # smoothing of the cluster prompt-length unit

# --- sustained-congestion filter (seconds) ----------------------------------
CONG_TAU_UP = 20.0         # time constant when cluster congestion rises
CONG_TAU_DOWN = 60.0       # time constant when it falls


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


def _prompt_scale(n_pos, state):
    """Running average prompt length: the cluster-level congestion unit."""
    ema = state.get("prompt_ema")
    if ema is None:
        ema = float(n_pos)
    else:
        ema += PROMPT_EMA_ALPHA * (n_pos - ema)
    state["prompt_ema"] = ema
    return ema


def _sustained(value, now, state):
    """Time-based EMA of cluster congestion (slow to rise, slower to fall)."""
    prev = state.get("cong_ema")
    last = state.get("cong_time")
    if prev is None or last is None:
        prev, last = 0.0, now
    dt = now - last
    if dt < 0.0:
        dt = 0.0
    tau = CONG_TAU_UP if value > prev else CONG_TAU_DOWN
    alpha = 1.0 - math.exp(-dt / tau)
    ema = prev + alpha * (value - prev)
    state["cong_ema"] = ema
    state["cong_time"] = now
    return ema


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    queues, attended = _attended_context(replicas, ctx.state)
    scale = _prompt_scale(n_pos, ctx.state)

    # ---- 1. cluster state and congestion gate ------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    congestion = mean_load / n_pos
    ramp = (congestion - ATTN_RAMP_START) / ATTN_RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0
    gate = G_FLOOR + (1.0 - G_FLOOR) * ramp

    # Cluster-level congestion for the TTFT filter: only congestion that is
    # both present now and sustained over tens of seconds opens the filter.
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / num
    instant = mean_pending / scale
    sustained = _sustained(instant, ctx.now, ctx.state)
    cluster_cong = instant if instant < sustained else sustained
    cluster_ramp = (cluster_cong - CLUSTER_RAMP_START) / CLUSTER_RAMP_WIDTH
    if cluster_ramp < 0.0:
        cluster_ramp = 0.0

    # ---- 2. price of a cache miss ----------------------------------------
    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + gate * DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    batch_norm = mean_out + BATCH_REF
    attn_weight = ATTN_WEIGHT_MAX * ramp
    interference = gate * INTERFERENCE
    context_price = gate * CONTEXT_PRICE

    # Depth-weighted stall price: long prompts' deep chunks are the slow
    # mixed steps that form the TBT tail; short prompts' chunks are cheap.
    depth = n_pos / scale
    if depth < DEPTH_MIN:
        depth = DEPTH_MIN
    elif depth > DEPTH_MAX:
        depth = DEPTH_MAX
    stall_weight = (gate * STALL_FLOOR_W + STALL_KNEE_W * ramp) * depth

    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 3. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    costs = []
    ttft_est = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0

        # Mooncake-style TTFT estimate (used only by the feasibility filter).
        ttft_est.append(r.pending_prefill_tokens + uncached)

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + interference * decode_excess)

        cost = load[i] + lam_i * uncached

        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            cost += gate * BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        ctx_excess = context[i] + uncached - mean_ctx
        if ctx_excess > 0.0:
            cost += context_price * ctx_excess * ctx_excess / ctx_norm

        if attn_weight > 0.0:
            co_resident = len(queues[i])
            cost += attn_weight * (attended[i] + n * co_resident)

        if uncached > 0:
            co_decoders = r.outstanding_requests
            if r.running > co_decoders:
                co_decoders = r.running
            cost += stall_weight * uncached * co_decoders

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 4. TTFT feasibility filter (fades out under sustained congestion) --
    if cluster_ramp < 1.0:
        min_est = min(ttft_est)
        slack = (SLACK_MIN + SLACK_REL * min_est) / (1.0 - cluster_ramp)
        limit = min_est + slack
        inf = float("inf")
        for i in range(num):
            if ttft_est[i] > limit:
                costs[i] = inf

    # ---- 5. choose --------------------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n)
    queues[pick].append(n)
    attended[pick] += n
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
