"""Seed RL design: Double DQN on the semi-MDP with the interval cost of Bradtke & Duff.

Reward between two decisions = -(requests in system) * dt, scaled per replica. Summed over
an episode this is total latency (Little's law), it is known at the next decision (no wait
for completions, no survivorship bias), and a decision that slows requests already on a
replica is charged for it through the bootstrap. A convex term on the busiest replica's
backlog adds tail pressure.
"""

TOKENS = 8192.0

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
        replica.pending_prefill_tokens / TOKENS,
        uncached / TOKENS,
        n / TOKENS,
        replica.running / 8.0,
        replica.waiting / 8.0,
        replica.outstanding_requests / 16.0,
        replica.kv_usage,
        1.0 if request.session_last_replica == replica.index else 0.0,
        (replica.pending_prefill_tokens - mean_pending) / TOKENS,
        (replica.running - mean_running) / 8.0,
    ]


def interval_reward(replicas_before, replicas_after, dt, ctx):
    k = len(replicas_after)
    in_system = sum(r.outstanding_requests for r in replicas_after) / k
    worst = max(r.pending_prefill_tokens for r in replicas_after) / TOKENS
    return -dt * (in_system / 8.0 + 0.5 * worst * worst)
