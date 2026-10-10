"""evolve.sim_fork: a forked simulation reproduces the unforked run byte for byte (forks at several times, stepped
alternately, taken inside a run() callback, bursty arrivals with prefix-cache hits), and a forced decision changes
only its own fork, routes the forced request where it was told and leaves the policy's memory as if the policy had
chosen that replica (review E r1, 2026-10-10)."""

import filecmp
import types
from pathlib import Path

import pandas as pd
import pytest

from evolve import evaluate as ev
from evolve import sim_fork as sf
from evolve.sim_fork import SimHandle, build_simulator, override_next_decision
from vidur.scheduler.global_scheduler.evolved_global_scheduler import EvolvedGlobalScheduler, PolicyError
from vidur.scheduler.global_scheduler.learned_global_scheduler import LearnedGlobalScheduler

POLICY = str(ev.REPO / "evolve/finalists/l3_48a067a7df63.py")  # keeps a per-replica FIFO of its own picks
# Exact replay (p0): 300 requests in 33 routing calls (bursts of 4..17), prefix-cache hits on ~296 of them.
SC = ev.Scenario("Conv", 4, 1.5, 6000, 300, 0)
OUT_FILES = ("request_metrics.csv", "token_gaps.npz", "predictor_info.json")


def _build(out) -> SimHandle:
    argv = ev.sim_command("evolved", POLICY, SC, ev.window_csv(SC), Path(out), ["--no-metrics_config_store_plots"])
    return build_simulator(argv[3:])


def _dir(d) -> Path:
    return next(Path(d).rglob("request_metrics.csv")).parent


def _same(d1, d2) -> bool:
    a, b = _dir(d1), _dir(d2)
    return all(filecmp.cmp(a / f, b / f, shallow=False) for f in OUT_FILES)


def _metrics(d) -> pd.DataFrame:
    return pd.read_csv(_dir(d) / "request_metrics.csv").set_index("Request Id").sort_index()


def _advance(h) -> bool:
    """Execute the next routing call; False when the run has ended."""
    n = h.decisions
    return h.run(lambda x: x.decisions > n)


def _finish(h, out=True):
    h.run()
    if out:
        h.write_output()


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref")
    a = _build(d)
    a.run()
    a.write_output()
    assert a.decisions == 33
    df = _metrics(d)
    assert df.index.is_unique and (df["request_num_prefill_tokens_cached"] > 0).sum() > 250
    return d


def test_interleaved_forks_at_two_times_reproduce_the_run(ref, tmp_path):
    o = _build(tmp_path / "O")
    assert o.run_to_decision(8)
    f1 = o.fork(str(tmp_path / "F1"))
    assert len(o.sim._scheduler._request_queue) >= 2  # a multi-request burst is pending at the fork point
    for _ in range(3):  # step the parent and the first fork alternately before the second fork
        assert _advance(o) and _advance(f1)
    assert o.run_to_decision(20)
    f2 = o.fork(str(tmp_path / "F2"))
    live = [o, f1, f2]
    while live:
        for h in list(live):
            if not _advance(h):
                live.remove(h)
    for h in (o, f1, f2):
        h.write_output()
    for x in ("O", "F1", "F2"):
        assert _same(ref, tmp_path / x), x


def test_fork_inside_run_callback(ref, tmp_path):
    o = _build(tmp_path / "O")
    forks = []

    def cb(h):
        if h.at_decision() and h.decisions == 15 and not forks:
            forks.append(h.fork(str(tmp_path / "F")))
            with pytest.raises(RuntimeError):  # a handle suspended under another one cannot be used
                x = h.fork()
                x.run(lambda _: bool(h.fork()))  # stops at once if fork() does not raise
        return False

    assert not o.run(cb)
    o.write_output()
    f = forks[0]
    assert f.decisions == 15 and not f.done()
    _finish(f)
    assert _metrics(tmp_path / "F").index.is_unique
    assert _same(ref, tmp_path / "O") and _same(ref, tmp_path / "F")


def test_forced_decisions(ref, tmp_path):
    k = 20
    o = _build(tmp_path / "O")
    assert o.run_to_decision(k)
    t_fork, gs = o.now, o.sim._scheduler
    burst = list(gs._request_queue)
    assert len(burst) >= 2
    first = min(burst, key=lambda r: (r.arrived_at, r.id))  # schedule() sorts the burst like this
    n = first.num_prefill_tokens

    p = o.fork()  # the policy's own decision
    assert p.run(lambda h: h.decisions >= k)
    chosen = p.sim._scheduler._assigned[first.id][0]
    a = (chosen + 2) % SC.replicas

    for bad in (-1, SC.replicas, True):  # range check (a negative index would silently wrap)
        with pytest.raises(PolicyError):
            override_next_decision(o, bad)
    with pytest.raises(RuntimeError):  # a fork without output_dir must not overwrite its parent's files
        p.write_output()

    # Forcing the policy's own choice is a no-op.
    e = o.fork(str(tmp_path / "E"))
    override_next_decision(e, chosen)

    # Force a different replica, then fork the forced handle: the pending override is bound to each copy.
    g = o.fork(str(tmp_path / "G"))
    override_next_decision(g, a)
    g2 = g.fork(str(tmp_path / "G2"))
    _finish(g2)  # the copy applies its override to itself; g and its pending override are untouched
    assert sf._FORCE_ATTR in g.sim._scheduler.__dict__ and "_call" in g.sim._scheduler.__dict__

    # Counterfactual by construction: the policy itself picks `a` for the first request (its _select returns a
    # after running as usual, so its rng draws are unchanged); no override.
    c = o.fork(str(tmp_path / "C"))
    mod = gs._policy
    orig, calls = mod._select, []

    def select_a(*args, **kw):
        r = orig(*args, **kw)
        if not calls:
            calls.append(r)
            return a
        return r

    mod._select = select_a
    try:
        assert c.run(lambda h: h.decisions >= k)
    finally:
        mod._select = orig
    assert calls == [chosen]

    # The parent continues untouched by the overrides of its forks.
    _finish(o)
    assert _same(ref, tmp_path / "O")
    assert "_call" not in gs.__dict__ and sf._FORCE_ATTR not in gs.__dict__

    # After the forced call, the policy's memory and the scheduler's ledger equal those of the run where the
    # policy chose `a` itself, and differ from those of its own choice.
    assert g.run(lambda h: h.decisions >= k)

    def snap(h):
        s = h.sim._scheduler
        st = {key: [list(q) for q in v] if key == "attn_queues" else list(v) for key, v in s._ctx.state.items()}
        return (st, dict(s._assigned), list(s._outstanding), list(s._pending_prefill), [dict(x) for x in s._inflight],
                dict(s._session_last), s._ctx.rng.getstate())

    assert snap(g) == snap(c) and snap(g)[0] != snap(p)[0]
    q_g = g.sim._scheduler._ctx.state["attn_queues"]
    assert q_g[a][-1] == n
    for h in (g, c, e):
        _finish(h)
    assert _same(ref, tmp_path / "E")
    assert _same(tmp_path / "G", tmp_path / "C") and _same(tmp_path / "G", tmp_path / "G2")
    assert not _same(ref, tmp_path / "G")

    R, G = _metrics(ref), _metrics(tmp_path / "G")
    assert R.loc[first.id, "replica"] == gs._replica_ids[chosen].id and G.loc[first.id, "replica"] == gs._replica_ids[a].id
    done_before = R.index[R["request_arrived_at"] + R["request_e2e_time"] <= t_fork]
    assert len(done_before) > 50
    pd.testing.assert_frame_equal(R.loc[done_before], G.loc[done_before])
    routed_before = R.index[R["request_arrived_at"] < first.arrived_at]
    assert (R.loc[routed_before, "replica"] == G.loc[routed_before, "replica"]).all()


def test_id_counters_are_per_handle(tmp_path):
    from vidur.entities import Batch, BatchStage, ExecutionTime

    lazy = (Batch, BatchStage, ExecutionTime)  # get a counter only on first use, during the run
    a = _build(tmp_path / "A")
    assert all(a.counters.get(c, sf.MISSING) is sf.MISSING for c in lazy)
    d = a.fork()  # forked before the first batch
    assert a.run_to_decision(5)
    at5 = {c: a.counters[c] for c in lazy}
    assert all(isinstance(v, int) and v >= 0 for v in at5.values())
    b = _build(tmp_path / "B")  # a second handle in the same process starts from fresh counters
    assert all(b.counters.get(c, sf.MISSING) is sf.MISSING for c in lazy)
    assert b.run_to_decision(5) and d.run_to_decision(5)
    assert {c: b.counters[c] for c in lazy} == at5 == {c: d.counters[c] for c in lazy}
    assert all("_id" not in c.__dict__ for c in lazy)  # nothing leaks into the process outside a handle

    # The predictor is shared; its clamp counts are per handle.
    pred = a._predictor()
    outside = pred.clamp_hits
    with d.active():
        pred.clamp_hits["probe"] = 1
    assert d.clamp_hits == {**a.clamp_hits, "probe": 1} and "probe" not in a.clamp_hits
    assert pred.clamp_hits is outside and "probe" not in outside
    assert "probe" in d.fork().clamp_hits


def test_unforkable_and_unreconcilable_cases():
    def stub(gs):
        h = SimHandle.__new__(SimHandle)
        h.sim = types.SimpleNamespace(_scheduler=gs)
        return h

    learned = object.__new__(LearnedGlobalScheduler)
    with pytest.raises(NotImplementedError, match="learned"):
        stub(learned).fork()
    with pytest.raises(NotImplementedError):
        override_next_decision(stub(learned), 0)

    evolved = object.__new__(EvolvedGlobalScheduler)
    evolved._policy = types.ModuleType("p")
    evolved._policy.TABLE = {}
    with pytest.raises(NotImplementedError, match="mutable module globals"):
        stub(evolved).fork()

    req = types.SimpleNamespace(num_prefill_tokens=7)
    ctx = types.SimpleNamespace(state={"other": [1]})
    with pytest.raises(NotImplementedError, match="cannot force"):
        sf._reconcile(types.ModuleType("p"), {"other": []}, req, None, 0, 1, ctx)

    from collections import deque
    before = {"attn_queues": [deque([3, 4]), deque([5])], "attn_sums": [7, 5]}
    good = {"attn_queues": [deque([4, 7]), deque([5])], "attn_sums": [11, 5]}  # trimmed 3, appended 7 at 0
    assert sf._fifo_move(before, good, req, 0, 1)
    assert [list(q) for q in good["attn_queues"]] == [[4], [5, 7]] and good["attn_sums"] == [4, 12]
    odd = {"attn_queues": [deque([3, 4, 7]), deque([9])], "attn_sums": [14, 9]}  # replica 1 changed otherwise
    assert not sf._fifo_move(before, odd, req, 0, 1)
