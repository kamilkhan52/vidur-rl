"""Astrolabe port (evolve/priorwork/astrolabe_core.py): timing model, replica simulation and routing rule.

The predictor tests load the h100df prediction bundle (cache/prediction_bundle_4534b960326a7810.pkl, ~3 GB in
memory, a few seconds); they skip when it is absent.
"""
import copy
import math
import random
import sys
from types import SimpleNamespace

import pytest

from evolve.priorwork import astrolabe_core as A

SRC = (A.REPO / "evolve/policies/priorwork/astrolabe_h100df.py").read_text()
SPEC = eval(SRC[SRC.index("ReplicaSpec("):SRC.index("choose, observe")].strip(), {"ReplicaSpec": A.ReplicaSpec})


@pytest.fixture(scope="module")
def full_predictor():
    if not (A.REPO / "cache" / f"prediction_bundle_{SPEC.bundle_key}.pkl").exists():
        pytest.skip("h100df prediction bundle not available")
    return A.build_predictor(SPEC, prune=False)


@pytest.fixture(scope="module")
def pruned(full_predictor):
    full, rc, cc = full_predictor
    p = copy.copy(full)
    A._prune(p, SPEC)
    return p, rc, cc


def test_policy_spec_matches_the_h1_run_arguments():
    from evolve import evaluate as ev
    args = ev.DEVICE_ARGS["h100df"]
    assert ev.VIDUR_DEVICE["h100df"] == SPEC.device
    assert args[args.index("--replica_config_network_device") + 1] == SPEC.network_device
    assert int(args[args.index("--cache_config_num_blocks") + 1]) == SPEC.num_blocks
    flag = "--random_forest_execution_time_predictor_config_attention_input_file"
    assert args[args.index(flag) + 1] == SPEC.attention_input_file
    assert ev.MODEL_NAME["l3_8b"] == SPEC.model_name and ev.TP["l3_8b"] == SPEC.tensor_parallel_size


def test_argv_check_rejects_another_deployment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["vidur", "--replica_config_device", "a100"])
    with pytest.raises(RuntimeError, match="replica_config_device"):
        A._check_argv(SPEC)
    monkeypatch.setattr(sys, "argv", ["vidur", "--replica_config_device", "h100", "--cache_config_num_blocks", "30208"])
    A._check_argv(SPEC)


def test_pruned_table_equals_full_table(full_predictor, pruned):
    from vidur.entities.execution_time_predictor_request import ExecutionTimePredictorRequest as R
    full, p = full_predictor[0], pruned[0]
    rng = random.Random(0)
    n = 0
    for _ in range(5000):
        reqs, budget = [], SPEC.chunk_size
        for _ in range(rng.randint(0, SPEC.batch_size_cap)):
            reqs.append(R(rng.randint(1, 400_000), 1, True))
            budget -= 1
            if budget == 0:
                break
        while budget > 0 and len(reqs) < SPEC.batch_size_cap and rng.random() < 0.7:
            c = rng.randint(1, budget)
            reqs.append(R(rng.choice([0, rng.randint(0, 200_000)]), c, False))
            budget -= c
        if reqs:
            assert full.get_execution_time(reqs, 0).total_time == p.get_execution_time(reqs, 0).total_time
            n += 1
    assert n > 4000


def _ledger(rng, n_run, n_wait, max_prompt=20000):
    running = [(rng.randint(100, max_prompt), d, d + rng.randint(1, 300) if rng.random() < 0.8 else d + 10)
               for d in (rng.randint(1, 400) for _ in range(n_run))]
    waiting = [(0.0, rng.randint(100, max_prompt), rng.choice([0, 512, 4096]), rng.randint(1, 600))
               for _ in range(n_wait)]
    return running, waiting


@pytest.mark.parametrize("num_blocks", [SPEC.num_blocks, 1500])  # 1500 blocks: KV pressure and preemption
def test_counting_kv_gives_the_same_predictions_as_the_real_kv_manager(pruned, num_blocks):
    from vidur.config import CacheConfig
    p, rc, _ = pruned
    spec = A.ReplicaSpec(**{**SPEC.__dict__, "num_blocks": num_blocks})
    cc = CacheConfig(block_size=spec.block_size, num_blocks=num_blocks, enable_prefix_caching=False)
    fast = A.ReplicaSimulator(spec, p, rc, cc)
    exact = A.ReplicaSimulator(spec, p, rc, cc, exact_kv=True)
    rng = random.Random(num_blocks)
    for _ in range(8):
        running, waiting = _ledger(rng, rng.randint(0, 60), rng.randint(0, 8), max_prompt=6000)
        target = (1.0, rng.randint(100, 6000), rng.choice([0, 1024]), rng.randint(1, 400))
        assert fast.predict_e2e(running, waiting, target) == exact.predict_e2e(running, waiting, target)


def test_simulation_leaves_no_trace(pruned):
    from vidur.entities import Batch, Request
    sim = A.ReplicaSimulator(SPEC, *pruned)
    ids = (Request._id, Batch._id)
    running, waiting = _ledger(random.Random(3), 30, 4)
    e1 = sim.predict_e2e(running, waiting, (1.0, 3000, 0, 200))
    assert (Request._id, Batch._id) == ids
    assert sim.kv.get_num_free_blocks() == SPEC.num_blocks and not sim.rs._requests and not sim.rs._running
    assert sim.predict_e2e(running, waiting, (1.0, 3000, 0, 200)) == e1  # reusable, deterministic


def test_prediction_is_monotone_in_load_and_cache(pruned):
    sim = A.ReplicaSimulator(SPEC, *pruned)
    alone = sim.predict_e2e([], [], (0.0, 4000, 0, 200))
    cached = sim.predict_e2e([], [], (0.0, 4000, 3584, 200))
    busy = sim.predict_e2e([(2000, 5, 300)] * 40, [(0.0, 8000, 0, 100)] * 3, (0.0, 4000, 0, 200))
    assert 0 < cached < alone < busy < math.inf
    # 200 tokens, one per iteration after the prefill: at least 200 iterations of a positive time
    assert alone > 199 * sim.predict_e2e([], [], (0.0, 16, 0, 1)) / 2


# ----------------------------------------------------------------------------------------- routing rule
class _FakeSim:
    def __init__(self, table):
        self.table, self.calls = table, []
        self.timer = SimpleNamespace(calls=0, misses=0)

    def predict_e2e(self, running, waiting, target):
        self.calls.append((len(running), len(waiting), target))
        return self.table(len(running) + len(waiting), target)


def _router(table, n):
    r = A.Astrolabe.__new__(A.Astrolabe)
    r.spec, r.sim, r.oracle, r.n, r.diag_path, r.times, r.records = SPEC, _FakeSim(table), None, None, None, [], []
    r._init(n)
    return r


def _req(t, P=1000, n=4, cached=None):
    return SimpleNamespace(arrived_at=t, num_prefill_tokens=P, prefix_hashes=(), cached_prefix_tokens=cached or (0,) * n)


def _views(n, kv=None):
    return [SimpleNamespace(index=i, kv_usage=(kv or [0.0] * n)[i]) for i in range(n)]


def _ctx(seed=0, now=0.0, n=4):
    return SimpleNamespace(rng=random.Random(seed), now=now, num_replicas=n, state={})


def test_power_of_two_picks_the_lower_prediction_and_samples_with_ctx_rng():
    n = 8
    r = _router(lambda load, tgt: 1.0 + load, n)
    ctx = _ctx(5, n=n)
    expect_rng = random.Random(5)
    for k in range(50):
        pick = r.choose(_req(0.0, n=n), _views(n), ctx)
        s = expect_rng.sample(range(n), 2)
        loads = {i: len(r.entries[i]) - (i == pick) for i in s}
        assert pick in s and loads[pick] == min(loads.values())
        expect_rng.choice([i for i in s if loads[i] == loads[pick]])  # the tie-break draw (always made)
    assert len(r.sim.calls) == 100  # two probes per decision


def test_later_requests_of_a_burst_see_earlier_ones():
    r = _router(lambda load, tgt: 1.0 + load, 2)
    ctx = _ctx(1, n=2)
    picks = [r.choose(_req(0.0, n=2), _views(2), ctx) for _ in range(6)]
    assert sorted(picks) == [0, 0, 0, 1, 1, 1]  # load-balanced: each probe saw the earlier routed requests


def test_full_replica_is_dropped_and_all_full_falls_back_to_random():
    r = _router(lambda load, tgt: 1.0, 2)
    assert all(r.choose(_req(0.0, n=2), _views(2, [1.0, 0.5]), _ctx(s, n=2)) == 1 for s in range(10))
    picks = {r.choose(_req(0.0, n=2), _views(2, [1.0, 1.0]), _ctx(s, n=2)) for s in range(20)}
    assert picks == {0, 1}


def test_ledger_dead_reckoning_overrun_rule_and_observe_matching():
    r = _router(lambda load, tgt: 1.0, 1)
    ctx = _ctx(0, n=1)
    r.choose(_req(0.0, P=500, n=1, cached=(256,)), _views(1), ctx)
    r.choose(_req(0.0, P=500, n=1), _views(1), ctx)  # same key: matched FIFO
    assert r.sim.calls[-1][2] == (0.0, 500, 0, A.PRIOR_DECODE_TOKENS) and r.sim.calls[-1][1] == 1
    out = lambda ev, **k: SimpleNamespace(event=ev, replica=0, arrived_at=0.0, num_prefill_tokens=500, **k)
    r.observe(out("first_token", first_token_at=0.5, completed_at=None, num_cached_tokens=256,
                  num_decode_tokens=None), ctx)
    running, waiting = r._ledger(0, 0.5 + 100 * A.TPOT_PRIOR_S + 1e-9)
    assert running == [(500, 101, A.PRIOR_DECODE_TOKENS)] and waiting == [(0.0, 500, 0, A.PRIOR_DECODE_TOKENS)]
    running, _ = r._ledger(0, 0.5 + 300 * A.TPOT_PRIOR_S + 1e-9)  # past the estimate: ends within 10 tokens
    assert running == [(500, 301, 301 + A.OVERRUN_TOKENS)]
    r.observe(out("end", first_token_at=0.5, completed_at=2.5, num_cached_tokens=256, num_decode_tokens=101), ctx)
    assert len(r.entries[0]) == 1 and r._len_estimate() == 101
    assert r.tpot[0] == pytest.approx(2.0 / 100)
    assert r.records[0][7] == pytest.approx(2.5)
    r.choose(_req(1.0, P=700, n=1), _views(1), _ctx(0, now=1.0, n=1))
    assert r.sim.calls[-1][2][3] == 101  # new requests get the running mean


def test_port_completes_in_a_simulation(tiny_trace, tmp_path, full_predictor):
    """End to end on the tiny trace with the h100df deployment (the policy checks it against the run's argv)."""
    import pandas as pd
    from conftest import base_args, run_simulator
    from evolve import evaluate as ev
    args = base_args(tiny_trace, tmp_path / "out", num_replicas=3, write_metrics=True,
                     replica_config_device=SPEC.device, replica_config_model_name=SPEC.model_name,
                     global_scheduler_config_type="evolved",
                     evolved_global_scheduler_config_policy_path=str(A.REPO / "evolve/policies/priorwork/astrolabe_h100df.py"))
    args += ev.DEVICE_ARGS["h100df"]  # network device, 30,208 KV blocks, decode-fixed attention profile
    sim = run_simulator(args)
    sim._write_output()
    df = pd.read_csv(next((tmp_path / "out").rglob("request_metrics.csv")))
    assert len(df) == 24 and sim._scheduler.is_empty()
    inst = sim._scheduler._policy._inst
    s = inst.summary()
    assert s["decisions"] == 24 and s["matched"] >= 20 and s["inf_predictions"] == 0
