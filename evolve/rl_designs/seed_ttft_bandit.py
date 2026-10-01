"""Seed RL design: Lodestar-style contextual bandit on -log TTFT.

Per-replica features are what the evolved finalist found to matter: queued work,
the uncached work this request would add, decode occupancy (prefill chunks steal
decode steps), KV pressure, session affinity, and the replica relative to the
cluster mean. Scales are fixed constants so features are O(1).
"""

import math

TOKENS = 8192.0


def _lg(tokens):
    # log scale keeps knee backlogs (1e5+ tokens) in range
    return math.log1p(max(tokens, 0.0) / TOKENS)

CONFIG = {
    "algorithm": "bandit",
    "hidden": [64, 64],
    "lr": 1e-3,
    "epsilon_start": 0.2,
    "epsilon_end": 0.02,
    "epsilon_decay_decisions": 20000,
}


def replica_features(request, replica, replicas, ctx):
    n = request.num_prefill_tokens
    uncached = n - request.cached_prefix_tokens[replica.index]
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / len(replicas)
    mean_running = sum(r.running for r in replicas) / len(replicas)
    return [
        _lg(replica.pending_prefill_tokens),
        _lg(uncached),
        n / TOKENS,
        replica.running / 8.0,
        math.log1p(replica.waiting) / 3.0,
        math.log1p(replica.outstanding_requests) / 4.0,
        replica.kv_usage,
        1.0 if request.session_last_replica == replica.index else 0.0,
        _lg(replica.pending_prefill_tokens) - _lg(mean_pending),
        (replica.running - mean_running) / 8.0,
    ]


def reward(outcome):
    return -math.log1p(outcome.ttft)
