"""Seed RL design: Double DQN with an interval cost on queued prompt work.

The EXP11 mechanism analysis (2026-10-01) showed that heuristic tails are dominated by
queueing (38-94 % of a tail request's time at the knee) while the evolved router's tail
is almost pure decode time. Requests-in-system is dominated by decoding requests, which
routing barely affects, so this design charges the interval cost on what routing does
control: uncached prompt tokens waiting to be prefilled, with a convex term on the worst
replica (tails come from the most backed-up replica).

Features follow the EXP11 ablation (2026-10-01): the value of a cache hit grows with load,
so uncached work is given together with congestion; a session's previous turn that is
still queued on a replica will be cached there soon although the probe cannot see it yet
(in-flight credit, remembered in ctx.state); decode excess per replica (the knee stabiliser).
"""

import math

TOKENS = 8192.0


def _lg(tokens):
    # log scale keeps knee backlogs (1e5+ tokens) in range
    return math.log1p(max(tokens, 0.0) / TOKENS)

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


INFLIGHT_WINDOW_S = 30.0


def _previous_turn(request, ctx):
    """(seconds since, prompt tokens) of this session's previous turn, remembered in ctx.state."""
    if request.session_id is None:
        return None
    table = ctx.state.setdefault("turns", {})
    key = (request.session_id, request.arrived_at, request.num_prefill_tokens)
    if ctx.state.get("last_key") != key:  # first replica of this decision: update the memory once
        ctx.state["last_key"] = key
        ctx.state["prev"] = table.get(request.session_id)
        table[request.session_id] = (request.arrived_at, request.num_prefill_tokens)
        if len(table) > 20000:
            cutoff = request.arrived_at - INFLIGHT_WINDOW_S
            ctx.state["turns"] = {k: v for k, v in table.items() if v[0] > cutoff}
    prev = ctx.state.get("prev")
    if prev is None:
        return None
    return request.arrived_at - prev[0], prev[1]


def replica_features(request, replica, replicas, ctx):
    n = request.num_prefill_tokens
    cached = request.cached_prefix_tokens[replica.index]
    k = len(replicas)
    mean_pending = sum(r.pending_prefill_tokens for r in replicas) / k
    mean_running = sum(r.running for r in replicas) / k
    home = request.session_last_replica == replica.index
    credit = 0.0  # previous turn still queued here: its prefix will be cached soon
    prev = _previous_turn(request, ctx)
    if home and prev is not None and prev[0] <= INFLIGHT_WINDOW_S and replica.waiting > 0:
        credit = min(prev[1], n) * 0.75
    uncached = max(0.0, n - max(cached, credit))
    congestion = mean_pending / max(n, 1)
    return [
        _lg(replica.pending_prefill_tokens),
        _lg(uncached),
        n / TOKENS,
        replica.running / 8.0,
        math.log1p(replica.waiting) / 3.0,
        math.log1p(replica.outstanding_requests) / 4.0,
        replica.kv_usage,
        1.0 if home else 0.0,
        _lg(replica.pending_prefill_tokens) - _lg(mean_pending),
        (replica.running - mean_running) / 8.0,
        min(congestion, 8.0) / 8.0,
        _lg(uncached) * min(congestion, 8.0) / 8.0,  # miss cost grows with load
        mean_running / 8.0,
        _lg(credit),
    ]


def interval_reward(replicas_before, replicas_after, dt, ctx):
    pending = [_lg(r.pending_prefill_tokens) for r in replicas_after]
    mean = sum(pending) / len(pending)
    worst = max(pending)
    return -0.25 * dt * (mean + 0.5 * worst * worst)
