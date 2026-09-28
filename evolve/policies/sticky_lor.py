"""Session-sticky LOR, as a policy file. Must reproduce the built-in STICKY_LOR router exactly.

A request whose session was routed before goes to the same replica (prefix-cache
affinity); a new session goes to the replica with the fewest outstanding requests.
"""


def choose(request, replicas, ctx):
    if request.session_last_replica >= 0:
        return request.session_last_replica
    least = min(r.outstanding_requests for r in replicas)
    return ctx.rng.choice([r.index for r in replicas if r.outstanding_requests == least])
