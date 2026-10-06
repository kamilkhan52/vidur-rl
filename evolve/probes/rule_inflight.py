"""PROBE (2026-10-06): evolved rule claude_r1_76be082eb0f5 with its session credit replaced by the
harness's exact in-flight prefix view. Not a search output."""
"""Congestion-priced, tail-aware cache routing with a decode-interference floor
on the price of a cache miss.

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + absolute imbalance guard      # convex min-max guard (light load)
           + relative imbalance guard      # scale-free min-max guard (saturation)
           + concurrency guard             # convex guard on decode batch size
           + kv_pressure_j                 # preemption risk

The one structural change with respect to the pure queueing formulation is the
*floor* of lambda.  A cache miss has two distinct externalities:

  (a) Queueing.  The uncached tokens delay everything already queued behind
      them during the current busy period, roughly `backlog / prompt`
      requests.  This term vanishes when queues are empty.

  (b) Decode interference.  Under chunked prefill, every prefill chunk the miss
      injects steals a step from the requests *already decoding* on that
      replica.  A 7k-token miss is a few hundred milliseconds of stolen decode
      steps whether or not anything is queued, so this term does NOT vanish at
      light load -- it scales with decode occupancy, not with backlog.

Pricing only (a) makes the router a pure work balancer whenever backlogs are
small: excellent TTFT, but it shreds the prefix sharing of a prefix-rich trace
and the extra prefill volume lands on top of resident decodes, which is exactly
the P99 TPOT (and, where decode dominates end-to-end time, the P99 e2e).
Hence

    lambda = 1 + DECODE_MISS_PRICE * min(1, mean_running / RUNNING_REF)
               + MISS_EXTERNALITY * max(0, mean_load / prompt - 1)   (capped)

At a truly idle cluster (nothing decoding) the floor collapses to 1 and the
policy is still a pure least-expected-work balancer, so the TTFT advantage of
spreading is preserved.  As soon as decodes are resident the floor rises to
~3 prompts and the router becomes sticky *before* queues form, which reduces
total cluster prefill volume -- the only lever that lowers co-scheduled prefill
interference on every replica at once.  At saturation (a) takes over and the
cache becomes capacity, as before.

Remaining ideas, unchanged from the parent:

  * Per-replica surcharge on the miss price: an uncached chunk sent to a
    decode-heavy replica steals steps from more decodes, so lambda is scaled up
    where the decode batch is above average.
  * Convex guard on excess concurrent decodes (TPOT grows with batch size, and
    batch size is not proportional to queued tokens).
  * Absolute + scale-free imbalance guards, so the min-max guard has the same
    strength at light load and at saturation.
  * In-flight prefix credit: if a session's previous turn is still waiting on
    its replica, the shared prefix is already booked in that replica's
    pending_prefill_tokens, so we must not charge for it twice.
"""

# --- pricing of a cache miss ------------------------------------------------
# (a) decode-interference floor: load-independent, scales with decode occupancy
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
# (b) queueing externality
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 30000.0  # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 30.0       # scale-free tail guard, in prompt-units

# --- decode delay terms ----------------------------------------------------
CONC_SCALE = 200.0         # token-equivalents per (excess concurrent decode)^2
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85       # KV occupancy where preemption risk starts
KV_PENALTY = 10000.0       # token-equivalent cost at a full KV cache

# --- session handling ------------------------------------------------------
INFLIGHT_WINDOW_S = 30.0   # a turn this recent may still be queued, not cached
SHARED_PREFIX_FRAC = 0.75  # fraction of a previous turn reused by the next one
HYSTERESIS_FRAC = 0.10     # anti-flapping margin for the sticky replica
HYSTERESIS_MIN = 300.0
MAX_SESSIONS = 20000


def _session_table(ctx):
    table = ctx.state.get("sessions")
    if table is None:
        table = {}
        ctx.state["sessions"] = table
    return table


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    # ---- 1. price of a cache miss ----------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num

    # (a) Decode-interference floor.  Present even with empty queues: the
    #     prefill chunks of a miss steal steps from resident decodes.  Absent
    #     when the cluster is idle, where spreading is free and helps TTFT.
    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy

    # (b) Queueing externality: backlog measured in own-prompt units.
    congestion = mean_load / n_pos
    lam += MISS_EXTERNALITY * max(0.0, congestion - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    # Reference scale for the scale-free guard: never let a nearly-idle cluster
    # divide by a tiny mean (there the absolute guard is the meaningful one).
    load_ref = mean_load if mean_load > n_pos else n_pos

    # ---- 2. in-flight prefix credit for the session's own replica ---------
    sessions = _session_table(ctx)
    sticky = request.session_last_replica
    credit = 0                              # extra tokens effectively cached
    if 0 <= sticky < num:
        prev = sessions.get(request.session_id)
        if prev is not None and request.arrived_at - prev[0] <= INFLIGHT_WINDOW_S:
            # The previous turn's prompt is a prefix of this one; if it is still
            # queued on `sticky`, its prefill is already booked in that replica's
            # pending_prefill_tokens, so we must not charge for it again.
            if replicas[sticky].waiting > 0:
                shared = prev[1] if prev[1] < n else n
                credit = int(SHARED_PREFIX_FRAC * shared)

    # ---- 3. score every replica ------------------------------------------
    costs = []
    inflight = request.inflight_prefix_tokens or (0,) * num
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        # PROBE: exact in-flight prefix view (prefixes routed here, not yet prefilled) instead of
        # the 30 s session heuristic; same information the Mooncake baseline receives.
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0

        # Miss price, surcharged where a prefill chunk would disturb more
        # decodes than average (multiplicative, so it matters most where the
        # miss is already expensive).
        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)

        cost = load[i] + lam_i * uncached

        excess = load[i] - mean_load
        if excess > 0:
            # Absolute guard: bounds skew in tokens (binds at light load).
            cost += excess * excess / IMBALANCE_SCALE
            # Scale-free guard: bounds skew as a fraction of the cluster mean,
            # which is what sets chunked-prefill interference at saturation.
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        # Convex guard on decode batch size: TPOT grows with concurrency, and
        # concurrency is not proportional to queued tokens.
        if decode_excess > 0.0:
            cost += CONC_SCALE * decode_excess * decode_excess

        if r.kv_usage > KV_SOFT_LIMIT:      # preemption / recompute risk
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 4. remember this turn -------------------------------------------
    sessions[request.session_id] = (request.arrived_at, n)
    if len(sessions) > MAX_SESSIONS:
        cutoff = request.arrived_at - INFLIGHT_WINDOW_S
        ctx.state["sessions"] = {k: v for k, v in sessions.items() if v[0] > cutoff}

    # ---- 5. choose --------------------------------------------------------
    best = min(costs)

    # Warmth is already priced into the cost; hysteresis only damps flapping
    # between near-equal replicas.
    if 0 <= sticky < num:
        margin = max(HYSTERESIS_MIN, HYSTERESIS_FRAC * n)
        if costs[sticky] <= best + margin:
            return sticky

    cands = [i for i, c in enumerate(costs) if c <= best + 1e-9]
    if len(cands) == 1:
        return cands[0]

    # Ties (typically idle replicas) go to the warmest cache, so new sessions
    # land where the shared system prefix already lives, then to the smallest
    # decode batch, then to the shortest queue.
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
