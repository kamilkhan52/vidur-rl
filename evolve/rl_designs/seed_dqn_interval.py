"""Seed RL design: Double DQN on the semi-MDP with the interval cost of Bradtke & Duff.

Reward between two decisions = -(requests in system) * dt, scaled per replica. Summed over
an episode this is total latency (Little's law), it is known at the next decision (no wait
for completions, no survivorship bias), and a decision that slows requests already on a
replica is charged for it through the bootstrap. A convex term on the busiest replica's
backlog adds tail pressure.
"""

import math

TOKENS = 8192.0


def _lg(tokens):
    # log scale keeps knee backlogs (1e5+ tokens) in range
    return math.log1p(max(tokens, 0.0) / TOKENS)

CONFIG = {
    "algorithm": "dqn",
    "beta": 0.05,  # value horizon ~20 simulated seconds
    "hidden": [64, 64],
    "lr": 5e-4,
    "target_update": 250,
    "epsilon_start": 0.1,
    "epsilon_end": 0.01,
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


def interval_reward(replicas_before, replicas_after, dt, ctx):
    # log-scaled so the per-second cost stays O(1) even with deep backlogs (v1 rewards reached -100
    # per transition and Q diverged to -1000)
    k = len(replicas_after)
    in_system = sum(r.outstanding_requests for r in replicas_after) / k
    worst = _lg(max(r.pending_prefill_tokens for r in replicas_after))
    return -0.25 * dt * (math.log1p(in_system) / 4.0 + 0.5 * worst * worst)
