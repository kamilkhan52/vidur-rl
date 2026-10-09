"""TTFT-feasible, congestion-gated, tail-aware cache routing with
depth-aware (attention-cost) prefill work estimates.

One currency: prefill-time-equivalents (tokens weighted by attention depth).

    cost_j = W_j                            # depth-weighted backlog we wait behind
           + lambda_j * w_j                 # depth-weighted work we add
           + g * decode-batch price         # convex in excess outstanding requests
           + g * decode-context price       # convex in excess resident KV tokens
           + ramp * attended-context price  # private + external decode cost
           + depth * decode-stall price     # uncached_j * co-decoders_j
           + kv_pressure_j                  # preemption risk

Change vs parent: prefill time is not linear in tokens -- every chunk also
attends to all earlier tokens of its prompt, so a token at depth d costs
(1 + d / ATTN_DEPTH_REF) linear-token-equivalents.  Mooncake-style token
counting under-counts replicas holding long-context prefills, and short
requests queued behind them form the TTFT tail.  We keep a per-replica FIFO
of routed prefill work (remaining uncached tokens, prompt length), reconcile
it with pending_prefill_tokens (replicas prefill FCFS, so completed work is
consumed from the front), and use the resulting depth-weighted backlog both
in the cost and in the TTFT feasibility filter.  Congestion gates still use
raw tokens so the regime calibration is unchanged.
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

# --- attention cost of prefill (tokens of depth at which attention work
#     per token equals the linear-layer work per token) ----------------------
ATTN_DEPTH_REF = 24000.0

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
DEPTH_MIN = 0.25
DEPTH_MAX = 3.0

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0

# --- TTFT feasibility filter -------------------------------------------------
SLACK_MIN = 1024.0
SLACK_REL = 0.15
CLUSTER_RAMP_START = 2.0
CLUSTER_RAMP_WIDTH = 6.0
PROMPT_EMA_ALPHA = 0.02

# --- sustained-congestion filter (seconds) ----------------------------------
CONG_TAU_UP = 20.0
CONG_TAU_DOWN = 60.0


def _work_weight(remaining, n):
    """Time-equivalent of the last `remaining` tokens of an n-token prompt."""
    mean_depth = n - 0.5 * remaining
    return remaining * (1.0 + mean_depth / ATTN_DEPTH_REF)


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


def _weighted_backlog(replicas, state, scale):
    """Depth-weighted pending prefill per replica.

    A FIFO of [remaining_uncached, prompt_len] entries per replica is
    reconciled with pending_prefill_tokens (FCFS: completed work leaves the
    front).  The queue's average cost multiplier is applied to the true
    pending count, so tracking errors only affect the multiplier.
    """
    queues = state.get("work_queues")
    if queues is None or len(queues) != len(replicas):
        queues = [deque() for _ in replicas]
        state["work_queues"] = queues
        state["work_sums"] = [0] * len(replicas)
    sums = state["work_sums"]
    default_factor = 1.0 + 0.5 * scale / ATTN_DEPTH_REF
    backlog = []
    for i, r in enumerate(replicas):
        q = queues[i]
        pending = r.pending_prefill_tokens
        excess = sums[i] - pending
        while excess > 0 and q:
            entry = q[0]
            take = entry[0] if entry[0] < excess else excess
            entry[0] -= take
            sums[i] -= take
            excess -= take
            if entry[0] <= 0:
                q.popleft()
        if not q:
            sums[i] = 0
        if sums[i] > 0:
            weighted = 0.0
            for entry in q:
                weighted += _work_weight(entry[0], entry[1])
            factor = weighted / sums[i]
        else:
            factor = default_factor
        backlog.append(pending * factor)
    return queues, sums, backlog


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
    work_q, work_sums, backlog = _weighted_backlog(replicas, ctx.state, scale)

    # ---- 1. cluster state and congestion gate (raw tokens) ----------------
    raw_load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(raw_load) / num
    mean_running = sum(r.running for r in replicas) / num
    mean_out = sum(r.outstanding_requests for r in replicas) / num

    congestion = mean_load / n_pos
    ramp = (congestion - ATTN_RAMP_START) / ATTN_RAMP_WIDTH
    if ramp < 0.0:
        ramp = 0.0
    elif ramp > 1.0:
        ramp = 1.0
    gate = G_FLOOR + (1.0 - G_FLOOR) * ramp

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
        if cached > n:
            cached = n
        uncached = n - cached
        own_work = _work_weight(uncached, n) if uncached > 0 else 0.0

        # Depth-aware TTFT estimate (feasibility filter).
        ttft_est.append(backlog[i] + own_work)

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + interference * decode_excess)

        cost = backlog[i] + DECODE_WEIGHT * r.running + lam_i * own_work

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

    # ---- 5. choose and record ----------------------------------------------
    pick = _select(request, replicas, ctx, costs, num, n)
    queues[pick].append(n)
    attended[pick] += n

    # Track the work the replica will report as pending (cache hits only).
    sim_uncached = n - request.cached_prefix_tokens[pick]
    if sim_uncached > 0:
        work_q[pick].append([sim_uncached, n])
        work_sums[pick] += sim_uncached
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
