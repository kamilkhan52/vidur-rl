"""Least outstanding uncached prefill (min load imbalance), as a policy file.

Reproduces the built-in LOP_UNCACHED router exactly, including under preemption
restarts (the built-in double-counted prefill completions until 2026-09-28).
"""


def choose(request, replicas, ctx):
    pending = [r.pending_prefill_tokens for r in replicas]
    imbalance = []
    for j in range(len(replicas)):
        extra = request.num_prefill_tokens - request.cached_prefix_tokens[j]
        loads = [p + (extra if i == j else 0) for i, p in enumerate(pending)]
        imbalance.append(max(loads) - min(loads))
    best = min(imbalance)
    return ctx.rng.choice([j for j in range(len(replicas)) if imbalance[j] == best])
