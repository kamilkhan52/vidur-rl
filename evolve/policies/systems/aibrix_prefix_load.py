"""AIBrix prefix-cache routing with load awareness (AIBrix docs, "prefix-cache" policy), simplified.

If the outstanding-request imbalance across replicas exceeds a standard-deviation-based
threshold, fall back to least-request. Otherwise pick, among the replicas with the longest
prefix match (if any match exceeds MIN_MATCH of the prompt), the least-loaded one; with no
useful match, least-request.
"""

import statistics

MIN_MATCH = 0.5
IMBALANCE_STD = 2.0  # route by load if max load > mean + IMBALANCE_STD * std


def _least(idx, replicas, ctx):
    lo = min(replicas[i].outstanding_requests for i in idx)
    c = [i for i in idx if replicas[i].outstanding_requests == lo]
    return c[0] if len(c) == 1 else ctx.rng.choice(c)


def choose(request, replicas, ctx):
    loads = [r.outstanding_requests for r in replicas]
    mean = statistics.fmean(loads)
    sd = statistics.pstdev(loads)
    allidx = list(range(len(replicas)))
    if sd > 0 and max(loads) > mean + IMBALANCE_STD * sd:
        return _least(allidx, replicas, ctx)
    n = max(request.num_prefill_tokens, 1)
    best = max(request.cached_prefix_tokens)
    if best / n >= MIN_MATCH:
        return _least([i for i, v in enumerate(request.cached_prefix_tokens) if v == best], replicas, ctx)
    return _least(allidx, replicas, ctx)
