"""Sticky routing that prices decode-batch *excess*, not absolute decode occupancy.

One cost, in seconds, of routing an arriving request to replica r (n = r.running,
mean_n = cluster mean of running, excess = max(0, n - mean_n)):

    slow(r)   = 1 + n / DECODE_REF                     chunked prefill shares the per-step
                                                       token budget with running decodes
    wait(r)   = backlog_sec * slow(r)                  prefill queued ahead of us
              + DECODE_SEC * n * slow(r)               our decode inside that batch (convex)
              + JOIN_SEC * excess(r)                   joining an *above-average* batch
              + KV guard above KV_GUARD                preemption / recompute risk
    own(r)    = price(r) * uncached_sec * slow(r)      prefill we must (re)do here
    harm(r)   = INTERFERE_SEC * uncached_sec * excess(r)   step budget our prefill chunks
                                                           steal from that fat batch

Why the two *excess* terms are the point of this revision.  P99 TPOT is set by the fattest
decode batch in the cluster, yet absolute decode charges are only a few seconds, so under
saturation -- where prefill queues are tens of seconds -- they stop steering anything: a
replica that happens to have drained its prefill queue looks cheap even while it holds the
largest batch, and a 7k uncached prompt dumped there stretches exactly the decodes that set
the TPOT tail.  Measuring the decode batch *relative to the cluster mean* makes the charge
scale-free: it stays visible at any load, is zero for a replica carrying its fair share, and
grows only for the outlier.  Splitting it into the harm we *suffer* (JOIN_SEC, paid per
request) and the harm we *inflict* (INTERFERE_SEC, paid per second of uncached prefill) is
what keeps affinity intact: a returning session with an 80% prefix hit brings ~0.2 s of
prefill and so barely disturbs a hot replica, whereas a fresh 7k prompt brings ~1.2 s and is
pushed to a replica that is decoding little.  Affinity therefore becomes the *cheap* way to
use a busy replica instead of competing with batch balance.

Routing never changes how much decode the cluster performs, only how much *prefill*.  The
price of that created work has a local term (work queued on r is re-paid by everything behind
it on r, FCFS) and a cluster-wide term driven by

    scarcity = clamp(min_r wait(r) / SCARCE_REF, 0, 1)

the committed work at the emptiest replica -- the only place a spill can go.  Balancing is
worth at most the idle capacity available to balance into; saving work is worth something
always.  At scarcity ~ 0 sessions overflow freely off a hot home; at scarcity ~ 1 re-prefill
is priced several-fold and the migration margin widens, because under sustained overload the
binding constraint is cluster prefill throughput and a discarded 6k prefix hurts TTFT, TPOT
and E2E at once.  Scarcity is the min of its instantaneous and EWMA value, so a burst that
momentarily fills every queue is not mistaken for saturation.

A returning session stays home unless an alternative is cheaper by the margin (the re-prefill
it must pay is already inside the costs), or unless home holds RUN_EXCESS more decodes than
the alternative -- a hard override for skew the smooth cost did not resolve.  A new session is
placed by the same cost, with cross-session prefix credit scaled by scarcity, so sessions that
merely share a system prefix stay spread out while decode batches are the scarce thing.
"""

PREFILL_RATE = 6000.0     # replica prefill throughput (tokens/s): tokens -> seconds
DECODE_SEC = 0.10         # seconds of slowdown charged per concurrent decode ...
DECODE_REF = 12.0         # ... growing by this factor again at DECODE_REF concurrent decodes
JOIN_SEC = 0.5            # seconds charged for joining a batch one decode above average
INTERFERE_SEC = 0.4       # decode delay inflicted per second of uncached prefill per excess decode
EXCESS_CAP = 10.0         # cap on measured batch excess (keeps one outlier from dominating)
KV_GUARD = 0.88           # KV usage above which preemption risk is priced in
KV_PENALTY = 8.0          # seconds added at 100% KV usage
MIGRATE_MARGIN = 2.0      # hysteresis: extra seconds of home cost needed to leave home
MIGRATE_SCARCE = 1.5      # margin multiplier added at full scarcity (2.0s -> 5.0s)
RUN_EXCESS = 5            # decode-batch excess at home that forces the valve open
PRICE_SPAN = 3.0          # max markup from a replica's *own* prefill backlog
PRICE_REF = 1.0           # seconds of own backlog per unit of local markup
SCARCE_PRICE = 3.0        # markup on created prefill work at full cluster scarcity
SCARCE_REF = 8.0          # seconds at the emptiest replica that count as full scarcity
SCARCE_EWMA = 0.04        # smoothing of scarcity: separates bursts from sustained overload
PREFIX_REUSE = 0.80       # assumed prompt overlap between consecutive turns of a session
MAX_SESSIONS = 200000     # bound on remembered sessions


def _measure(replicas):
    """Per-replica (wait seconds, prefill slowdown, local price markup, batch excess)."""
    n_rep = len(replicas)
    mean_run = sum(r.running for r in replicas) / n_rep

    waits = []
    slows = []
    local = []
    excess = []
    for r in replicas:
        n = r.running
        slow = 1.0 + n / DECODE_REF
        backlog = r.pending_prefill_tokens / PREFILL_RATE

        over = n - mean_run
        if over < 0.0:
            over = 0.0
        elif over > EXCESS_CAP:
            over = EXCESS_CAP

        w = backlog * slow + DECODE_SEC * n * slow + JOIN_SEC * over
        if r.kv_usage > KV_GUARD:
            w += KV_PENALTY * (r.kv_usage - KV_GUARD) / (1.0 - KV_GUARD)

        markup = backlog / PRICE_REF
        if markup > PRICE_SPAN:
            markup = PRICE_SPAN

        waits.append(w)
        slows.append(slow)
        local.append(markup)
        excess.append(over)
    return waits, slows, local, excess


def _scarcity(waits, state):
    """How little spare capacity the cluster has: 0 = somewhere idle, 1 = all deeply queued.

    Uses the emptiest replica (the only spill target), smoothed so that a burst which
    momentarily fills every queue is not mistaken for sustained saturation.
    """
    instant = min(waits) / SCARCE_REF
    if instant > 1.0:
        instant = 1.0
    avg = state.get("scarcity_ewma")
    if avg is None:
        avg = instant
    avg = (1.0 - SCARCE_EWMA) * avg + SCARCE_EWMA * instant
    state["scarcity_ewma"] = avg
    return instant if instant < avg else avg


def choose(request, replicas, ctx):
    state = ctx.state
    prompts = state.get("session_prompt")
    if prompts is None:
        prompts = {}
        state["session_prompt"] = prompts

    prompt = float(request.num_prefill_tokens)
    home = request.session_last_replica

    waits, slows, local, excess = _measure(replicas)
    scarcity = _scarcity(waits, state)
    # Predicted prefix at home: the previous turn may be queued but not yet prefilled, in
    # which case the cache view understates what home will hold when we get there.
    predicted = PREFIX_REUSE * min(float(prompts.get(request.session_id, 0.0)), prompt)

    costs = []
    for k, r in enumerate(replicas):
        j = r.index
        cached = float(request.cached_prefix_tokens[j])
        if j == home:
            if predicted > cached:
                cached = predicted
        elif home < 0:
            # Cross-session prefix credit is taken only when work is the scarce resource.
            cached *= scarcity
        if cached > prompt:
            cached = prompt

        uncached_sec = (prompt - cached) / PREFILL_RATE
        price = 1.0 + local[k] + SCARCE_PRICE * scarcity
        costs.append(waits[k]
                     + price * uncached_sec * slows[k]          # prefill we must do here
                     + INTERFERE_SEC * uncached_sec * excess[k])  # harm to the fattest batch

    if home < 0:
        best = min(costs)
        picks = [r.index for r, c in zip(replicas, costs) if c <= best + 1e-9]
        target = picks[0] if len(picks) == 1 else ctx.rng.choice(picks)
    else:
        home_cost = None
        home_running = 0
        alt_cost = None
        alt_index = home
        alt_running = 0
        for k, r in enumerate(replicas):
            if r.index == home:
                home_cost = costs[k]
                home_running = r.running
            elif alt_cost is None or costs[k] < alt_cost:
                alt_cost = costs[k]
                alt_index = r.index
                alt_running = r.running

        margin = MIGRATE_MARGIN * (1.0 + MIGRATE_SCARCE * scarcity)
        if alt_cost is None or home_cost is None:
            target = home
        elif home_cost - alt_cost > margin:
            # Clearly cheaper elsewhere, even after paying to re-prefill home's prefix at
            # that replica's own backlog price and at the cluster's scarcity price.
            target = alt_index
        elif home_running - alt_running >= RUN_EXCESS and alt_cost < home_cost:
            # Decode-batch skew the smooth cost did not resolve: P99 TPOT is set by the
            # worst batch, so never keep stacking a session onto the fattest one.
            target = alt_index
        else:
            target = home

    if len(prompts) >= MAX_SESSIONS:
        prompts.clear()
    prompts[request.session_id] = request.num_prefill_tokens
    return target
