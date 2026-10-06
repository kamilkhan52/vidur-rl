"""Congestion-priced cache routing whose miss price is set by the number of
decodes the miss will interfere with (projected decode batch).

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                          # backlog this request waits behind
           + lambda * share_j * uncached_j   # work it adds, priced at system cost
                                             #   and by how many decodes it disturbs
           + absolute / relative imbalance guards on load
           + scale-free projected-batch guard
           + kv_pressure_j

Why share_j.  Under chunked prefill with FCFS admission, the uncached tokens of
this request are prefilled while (i) the replica's running requests and (ii)
the requests waiting ahead of it (they are admitted first) are decoding.  The
TPOT damage of a miss is therefore ~ uncached_j * B_j with B_j = running_j +
waiting_j, the projected decode batch.  We express this relative to the
cluster: share_j = 1 + SHARE * (B_j - mean_B) / (mean_B + 1), clamped.  Misses
are steered to replicas with few decoders; hits (little prefill) remain free
to land on busy warm replicas.  At light load this is also what minimises TTFT
(misses go to idle replicas), so the change is aligned with the production
min-TTFT router below the knee.

The batch guard is priced in miss units (lambda * prompt), so it keeps the
same strength relative to cache affinity at light load and at saturation;
TPOT grows with the batch a request decodes in.
"""

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- decode interference of a miss (relative, two-sided) --------------------
SHARE = 0.5                # miss-price slope per relative batch excess
SHARE_MIN = 0.5            # quietest replica: miss priced at half
SHARE_MAX = 2.5            # busiest replica: miss priced at most 2.5x

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 30000.0  # absolute tail guard
REL_IMBALANCE = 30.0       # scale-free tail guard, in prompt-units

# --- projected batch guard (in miss units) ----------------------------------
BATCH_GUARD = 0.5          # lambda*prompt per (relative batch excess)^2

# --- preemption risk -------------------------------------------------------
KV_SOFT_LIMIT = 0.85
KV_PENALTY = 10000.0

# --- anti-flapping ---------------------------------------------------------
HYSTERESIS_FRAC = 0.10
HYSTERESIS_MIN = 300.0


def choose(request, replicas, ctx):
    num = len(replicas)
    n = request.num_prefill_tokens
    n_pos = n if n > 0 else 1

    # ---- 1. system price of a cache miss ---------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
    mean_load = sum(load) / num
    mean_running = sum(r.running for r in replicas) / num

    occupancy = mean_running / RUNNING_REF
    if occupancy > 1.0:
        occupancy = 1.0
    lam = 1.0 + DECODE_MISS_PRICE * occupancy
    lam += MISS_EXTERNALITY * max(0.0, mean_load / n_pos - 1.0)
    if lam > LAMBDA_MAX:
        lam = LAMBDA_MAX

    load_ref = mean_load if mean_load > n_pos else n_pos

    # Projected decode batch: running now + everything admitted ahead of us.
    batch = [r.running + r.waiting for r in replicas]
    mean_batch = sum(batch) / num
    batch_ref = mean_batch + 1.0

    # ---- 2. effective cached prefix (exact in-flight view) ----------------
    inflight = request.inflight_prefix_tokens or (0,) * num

    # ---- 3. score every replica ------------------------------------------
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]       # already booked in pending_prefill_tokens
        uncached = n - cached
        if uncached < 0:
            uncached = 0

        # Decode interference: a miss here disturbs ~batch[i] decodes.
        rel_batch = (batch[i] - mean_batch) / batch_ref
        share = 1.0 + SHARE * rel_batch
        if share < SHARE_MIN:
            share = SHARE_MIN
        elif share > SHARE_MAX:
            share = SHARE_MAX

        cost = load[i] + lam * share * uncached

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        # Scale-free guard on the batch this request will decode in.
        if rel_batch > 0.0:
            cost += BATCH_GUARD * lam * n_pos * rel_batch * rel_batch

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 4. choose --------------------------------------------------------
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
