"""PROBE (2026-10-07): cache-aware least-outstanding-requests. Among replicas within 4 outstanding
requests of the least loaded, take the one with the longest cached (or in-flight) prefix; ties go to the
smaller uncached prefill backlog. Not a search output."""

K = 4


def choose(request, replicas, ctx):
    num = len(replicas)
    least = min(r.outstanding_requests for r in replicas)
    inflight = request.inflight_prefix_tokens or (0,) * num
    best, best_key = [], None
    for i, r in enumerate(replicas):
        if r.outstanding_requests > least + K:
            continue
        cached = max(request.cached_prefix_tokens[i], inflight[i])
        key = (-cached, r.pending_prefill_tokens, r.outstanding_requests)
        if best_key is None or key < best_key:
            best, best_key = [i], key
        elif key == best_key:
            best.append(i)
    return best[0] if len(best) == 1 else ctx.rng.choice(best)
