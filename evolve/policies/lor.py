"""Least outstanding requests, as a policy file. Must reproduce the built-in LOR router exactly."""


def choose(request, replicas, ctx):
    least = min(r.outstanding_requests for r in replicas)
    return ctx.rng.choice([r.index for r in replicas if r.outstanding_requests == least])
