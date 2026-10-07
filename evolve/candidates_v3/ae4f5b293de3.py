"""Congestion-priced, tail-aware cache routing with a per-replica decode-stall
externality on cache misses.

One currency: prefill-token-equivalents of delay.

    cost_j = load_j                        # backlog this request waits behind
           + lambda_j * uncached_j         # work it adds, priced at system cost
           + stall_j                       # decode gaps its prefill will stretch (NEW)
           + DECODE_BLOCK_PRICE * (ctx_j + uncached_j)   # decode-step length
           + absolute / relative imbalance guards
           + decode-batch price            # convex in excess outstanding requests
           + decode-context price          # convex in excess attended KV tokens
           + kv_pressure_j                 # preemption risk

Change w.r.t. the parent: P99 TBT/TPOT remain 5-15% above the bar while TTFT
has slack.  P99 TBT counts every decode gap; a gap is long when a prefill chunk
shares the step.  A miss of u uncached tokens on replica j produces ~u/chunk
chunked steps, each stretching the gap of every decoder resident there when the
prefill runs (~ outstanding_requests_j, since queued requests are decoding by
then).  The parent's miss price is cluster-wide (identical across replicas) and
its interference surcharge only acts above the mean and weakly, so misses still
land on decode-heavy replicas.  We price each uncached token by the decoders it
will interrupt, normalised by the cluster mean (scale-free):

    stall_j = STALL_PRICE * uncached_j * out_j / (mean_out + STALL_REF)

Hits (uncached ~ 0) pay nothing, so the term also favours cache reuse, which
cuts total prefill work.
"""

# --- pricing of a cache miss ------------------------------------------------
DECODE_MISS_PRICE = 2.0    # extra prompt-equivalents when decodes are resident
RUNNING_REF = 4.0          # per-replica decode count at which the floor saturates
MISS_EXTERNALITY = 0.8     # extra delay-weight per prompt-equivalent of backlog
LAMBDA_MAX = 7.0           # cap: the cache is never an absolute pin

# --- decode-stall externality (NEW) ----------------------------------------
STALL_PRICE = 0.6          # miss-price per (decoders here / mean decoders)
STALL_REF = 2.0            # softens the term when the cluster is nearly idle

# --- queueing delay terms (token-equivalents) ------------------------------
DECODE_WEIGHT = 175.0      # prefill budget stolen by each concurrent decode
IMBALANCE_SCALE = 30000.0  # absolute tail guard; smaller => stricter balancing
REL_IMBALANCE = 30.0       # scale-free tail guard, in prompt-units

# --- decode-batch terms ----------------------------------------------------
BATCH_PRICE = 1.0          # prompt-equivalents per excess^2 / (mean + BATCH_REF)
BATCH_REF = 4.0            # softens the guard when the cluster is nearly idle
INTERFERENCE = 0.05        # fractional miss-price surcharge per excess decode

# --- decode-context terms --------------------------------------------------
DECODE_BLOCK_PRICE = 0.5   # token-equivalents per projected resident token
CONTEXT_PRICE = 1.5        # token-equivalents per excess_ctx^2 / (mean_ctx + KV_REF)
KV_REF = 16000.0           # softens the convex guard when little context is resident

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

    # ---- 1. price of a cache miss ----------------------------------------
    load = [r.pending_prefill_tokens + DECODE_WEIGHT * r.running for r in replicas]
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

    load_ref = mean_load if mean_load > n_pos else n_pos
    batch_norm = mean_out + BATCH_REF
    stall_norm = mean_out + STALL_REF

    # Decode context each replica's steps will attend over: resident KV tokens
    # plus prompts queued there that will become resident once prefilled.
    context = [r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens
               for r in replicas]
    mean_ctx = sum(context) / num
    ctx_norm = mean_ctx + KV_REF

    # ---- 2. score every replica ------------------------------------------
    inflight = request.inflight_prefix_tokens or (0,) * num
    costs = []
    for i, r in enumerate(replicas):
        cached = request.cached_prefix_tokens[i]
        if inflight[i] > cached:
            cached = inflight[i]
        uncached = n - cached
        if uncached < 0:
            uncached = 0

        decode_excess = r.running - mean_running
        if decode_excess < 0.0:
            decode_excess = 0.0
        lam_i = lam * (1.0 + INTERFERENCE * decode_excess)

        cost = load[i] + lam_i * uncached

        # Decode-stall externality: every prefill chunk of this miss stretches
        # one decode gap of each sequence resident here when it is prefilled.
        cost += STALL_PRICE * uncached * r.outstanding_requests / stall_norm

        # Linear decode-block price: projected resident context after this
        # request lands (shared prefix blocks are reused).
        projected_ctx = context[i] + uncached
        cost += DECODE_BLOCK_PRICE * projected_ctx

        excess = load[i] - mean_load
        if excess > 0:
            cost += excess * excess / IMBALANCE_SCALE
            rel = excess / load_ref
            cost += REL_IMBALANCE * n_pos * rel * rel

        # Decode-batch price: sequences this request will share steps with.
        batch_excess = r.outstanding_requests - mean_out
        if batch_excess > 0.0:
            cost += BATCH_PRICE * n_pos * batch_excess * batch_excess / batch_norm

        # Convex decode-context guard against outliers above the mean.
        ctx_excess = projected_ctx - mean_ctx
        if ctx_excess > 0.0:
            cost += CONTEXT_PRICE * ctx_excess * ctx_excess / ctx_norm

        if r.kv_usage > KV_SOFT_LIMIT:
            x = (r.kv_usage - KV_SOFT_LIMIT) / (1.0 - KV_SOFT_LIMIT)
            cost += KV_PENALTY * x * x

        costs.append(cost)

    # ---- 3. choose --------------------------------------------------------
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
