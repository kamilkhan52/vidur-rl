"""Seed RL design: Double DQN with an interval cost on queued prompt work.

The EXP11 mechanism analysis (2026-10-01) showed that heuristic tails are dominated by
queueing (38-94 % of a tail request's time at the knee) while the evolved router's tail
is almost pure decode time. Requests-in-system is dominated by decoding requests, which
routing barely affects, so this design charges the interval cost on what routing does
control: uncached prompt tokens waiting to be prefilled, with a convex term on the worst
replica (tails come from the most backed-up replica).
"""

TOKENS = 8192.0

CONFIG = {
    "algorithm": "dqn",
    "beta": 0.05,
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
    pending = [r.pending_prefill_tokens / TOKENS for r in replicas_after]
    mean = sum(pending) / len(pending)
    worst = max(pending)
    return -dt * (mean + 0.5 * worst * worst)
