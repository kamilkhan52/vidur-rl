"""AIBrix prefix-cache routing behind the gateway's load-imbalance gate (vllm-project/aibrix main 49b52c3:
pkg/plugins/gateway/algorithms/prefix_cache.go routeOriginal, prefix_cache_readme.md, load_balance.go
ApplyLoadImbalanceGate; saved to research/vidur-rl-prior-art/fulltext/), simplified to our views.

Gate (load_balance.go l.404-441): if max - min running requests >= MIN_GAP and the busiest pod exceeds
FACTOR x (mean of the others + 1), restrict the candidates to the pods with the minimum count.
Prefix cache (standard mode): match the request against the router's OWN prefix index (AddPrefix on
every route); matched candidates are sorted by match (desc) then load (asc), and the first with
load <= mean + LOAD_FACTOR x std (over the candidates) wins; otherwise the least-loaded candidate.
Defaults: FACTOR 2.0, MIN_GAP 8, LOAD_FACTOR 1. Not modelled: the default prefix-cache:5,load-balance:4
score blend (we route with prefix-cache alone, standard mode), 128-character blocks (we match token
blocks), index eviction. Outstanding requests stand in for running requests.
"""

import statistics

FACTOR = 2.0
MIN_GAP = 8
LOAD_FACTOR = 1.0


def _least(cands, loads, ctx):
    lo = min(loads[i] for i in cands)
    c = [i for i in cands if loads[i] == lo]
    return c[0] if len(c) == 1 else ctx.rng.choice(c)


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def choose(request, replicas, ctx):
    idx = ctx.state.setdefault("aibrix_index", [set() for _ in replicas])
    loads = [r.outstanding_requests for r in replicas]
    cands = list(range(len(replicas)))
    hi, lo = max(loads), min(loads)
    if hi - lo >= MIN_GAP and len(loads) > 2:
        others = (sum(loads) - hi) / (len(loads) - 1)
        if hi > FACTOR * (others + 1):
            cands = [i for i in cands if loads[i] == lo]
    pick = None
    if len(cands) > 1:
        m = {i: _match(idx[i], request.prefix_hashes) for i in cands}
        matched = sorted((i for i in cands if m[i] > 0), key=lambda i: (-m[i], loads[i]))
        if matched:
            cl = [loads[i] for i in cands]
            limit = statistics.fmean(cl) + LOAD_FACTOR * statistics.pstdev(cl)
            pick = next((i for i in matched if loads[i] <= limit), None)
    if pick is None:
        pick = _least(cands, loads, ctx)
    idx[pick].update(request.prefix_hashes)
    return pick
