"""Seed RL design: Double DQN on the semi-MDP, warm-started from the evolved router.

Same per-replica features as seed_ttft_bandit. The teacher (evolved finalist) acts on a
third of the training decisions and a DQfD margin loss pulls the greedy action towards
it; the reward is a bounded soft deadline on both TTFT and E2E so that tail outcomes are
penalised without unbounded targets (the old policy-poisoning failure).
"""

TOKENS = 8192.0
TTFT_TARGET = 10.0  # seconds
E2E_TARGET = 40.0

CONFIG = {
    "algorithm": "dqn",
    "beta": 0.05,
    "hidden": [64, 64],
    "lr": 5e-4,
    "target_update": 500,
    "epsilon_start": 0.1,
    "epsilon_end": 0.01,
    "epsilon_decay_decisions": 20000,
    "teacher_mix": 0.3,
    "imitation_weight": 1.0,
    "imitation_margin": 0.5,
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


def _soft(latency, target):
    # 1 at zero latency, 0 at the target, saturating at -1 beyond twice the target
    return max(-1.0, 1.0 - latency / target)


def reward(outcome):
    return 0.5 * _soft(outcome.ttft, TTFT_TARGET) + 0.5 * _soft(outcome.e2e, E2E_TARGET)
