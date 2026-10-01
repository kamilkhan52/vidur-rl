def choose(request, replicas, ctx):
    """Cache-affinity router with min-anchored convex decode guard.

    Sticky LOR gives excellent TTFT/E2E tails via prefix reuse, but its
    weakness is TPOT: a popular session's home accumulates a large decode
    batch, and chunked prefill then steals cycles.  The parent tried to fix
    this with a convex penalty on running above the *average*, but the
    average is skewed by the hot replica itself, so the penalty is too weak
    when one replica dominates.

    This policy keeps the sticky path and the linear relative decode cost
    from the best known variant, but adds a convex penalty on the excess of
    a replica's running batch over the cluster *minimum*.  The minimum is a
    robust lower bound: any replica that is far above the coldest one is
    charged super-linearly, so the single hottest decoder (which sets the
    P99 TPOT tail) is strongly avoided, while a uniformly busy cluster is
    left alone and cache is not needlessly abandoned.
    """
    P = request.num_prefill_tokens
    last = request.session_last_replica
    n = len(replicas)

    DECODE_COST = 1600.0     # token-equiv per outstanding request (queue)
    SHED_FACTOR = 2.0        # shed if home load > factor * cluster min + base
    SHED_BASE   = 2000.0
    RUN_FACTOR  = 1.5        # shed if home decode batch >> cluster average
    RUN_BASE    = 1.5
    PREFILL_FACTOR = 1.5     # shed if home pending prefill >> cluster average
    PREFILL_BASE   = 2000.0
    KV_START    = 0.80
    KV_SCALE    = 30000.0
    RUN_COST    = 400.0      # linear relative decode penalty
    RUN_SQ      = 20.0       # convex penalty on excess over cluster minimum

    # Per-replica load estimate (prefill backlog + decode concurrency).
    load = {}
    min_load = None
    min_run = None
    total_run = 0
    total_pending = 0
    home = None
    for r in replicas:
        i = r.index
        ld = r.pending_prefill_tokens + DECODE_COST * r.outstanding_requests
        load[i] = ld
        total_run += r.running
        total_pending += r.pending_prefill_tokens
        if min_load is None or ld < min_load:
            min_load = ld
        if min_run is None or r.running < min_run:
            min_run = r.running
        if i == last:
            home = r
    avg_run = total_run / n
    avg_pending = total_pending / n

    # --- Session affinity, with load-, decode- and prefill-aware escapes. ---
    if last >= 0 and home is not None:
        overloaded = load[last] > SHED_FACTOR * min_load + SHED_BASE
        decode_hot = home.running > RUN_FACTOR * avg_run + RUN_BASE
        prefill_hot = (home.pending_prefill_tokens
                       > PREFILL_FACTOR * avg_pending + PREFILL_BASE)
        if not (overloaded or decode_hot or prefill_hot):
            return last

    # --- Place / shed: cache, queue, convex decode (min-anchored), KV. ---
    best_idx = -1
    best_cost = None
    best_cached = -1
    for r in replicas:
        i = r.index
        cached = request.cached_prefix_tokens[i]
        if cached > P:
            cached = P
        cost = (P - cached) + load[i]

        # Penalise decode batches that exceed the coldest replica.
        excess_run = r.running - min_run
        if excess_run > 0:
            cost += RUN_COST * excess_run + RUN_SQ * excess_run * excess_run

        if r.kv_usage > KV_START:
            cost += KV_SCALE * (r.kv_usage - KV_START)

        if (best_cost is None or cost < best_cost
                or (cost == best_cost and cached > best_cached)):
            best_cost = cost
            best_cached = cached
            best_idx = i
    return best_idx
