"""Decision tests for the floor baselines (power of two choices, JSQ on KV load)."""
import random
from pathlib import Path

from vidur.scheduler.global_scheduler.evolved_global_scheduler import (
    PolicyContext, ReplicaView, RequestView, load_policy)

FLOOR = Path(__file__).resolve().parents[1] / "evolve" / "policies" / "floor"


def _rep(i, out=0, kv=0.0, pending=0):
    return ReplicaView(i, out, pending, 0, 0, kv, 1000)


REQ = RequestView(0.0, 100, None, (0, 0, 0, 0), -1, 0, 1)


def _ctx(seed=0):
    return PolicyContext(random.Random(seed), 4)


def test_p2c_outstanding_picks_less_loaded_of_two_samples():
    pol = load_policy(str(FLOOR / "p2c_outstanding.py"))
    reps = [_rep(0, 5), _rep(1, 1), _rep(2, 3), _rep(3, 0)]
    for seed in range(50):
        a, b = random.Random(seed).sample(range(4), 2)
        want = b if reps[b].outstanding_requests < reps[a].outstanding_requests else a
        assert pol.choose(REQ, reps, _ctx(seed)) == want


def test_p2c_kv_counts_pending_prefill_as_kv_load():
    pol = load_policy(str(FLOOR / "p2c_kv.py"))
    # replica 0: 100 used + 0 pending; replica 1: 0 used + 500 pending -> 0 is lighter
    reps = [_rep(0, kv=0.1), _rep(1, pending=500)]
    ctx = PolicyContext(random.Random(0), 2)
    assert pol.choose(REQ, reps, ctx) == 0


def test_jsq_kv_ties_break_on_outstanding():
    pol = load_policy(str(FLOOR / "jsq_kv.py"))
    reps = [_rep(0, out=3, kv=0.2), _rep(1, out=1, kv=0.2), _rep(2, out=0, kv=0.5), _rep(3, out=0, kv=0.3)]
    assert pol.choose(REQ, reps, _ctx()) == 1
