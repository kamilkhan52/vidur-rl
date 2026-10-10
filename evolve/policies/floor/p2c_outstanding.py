"""Power of two choices on outstanding requests (Mitzenmacher 2001; vllm-router `power_of_two`, AIBrix).

Sample two distinct replicas uniformly at random and route to the one with fewer outstanding requests;
ties go to the first sample. Prefix-cache blind. A floor baseline (readiness plan §5 item 1).
"""


def choose(request, replicas, ctx):
    a, b = ctx.rng.sample(range(len(replicas)), 2)
    return b if replicas[b].outstanding_requests < replicas[a].outstanding_requests else a
