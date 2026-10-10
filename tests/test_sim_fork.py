"""evolve.sim_fork: a forked simulation reproduces the unforked run byte for byte (forks at several times, stepped
alternately, taken inside a run() callback, bursty arrivals with prefix-cache hits), and a forced decision changes
only its own fork, routes the forced request where it was told and leaves the policy's memory as if the policy had
chosen that replica (review E r1, 2026-10-10)."""

import copy
import filecmp
import textwrap
import types
from collections import deque
from pathlib import Path

import pandas as pd
import pytest

from evolve import evaluate as ev
from evolve import sim_fork as sf
from evolve.sim_fork import SimHandle, build_simulator, override_next_decision
from vidur.scheduler.global_scheduler.evolved_global_scheduler import EvolvedGlobalScheduler, PolicyError, load_policy
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

    p = o.fork()  # the policy's own decision, recorded without forcing (choice None)
    rec_p = {}
    override_next_decision(p, None, record=rec_p)
    assert p.run(lambda h: h.decisions >= k)
    chosen = p.sim._scheduler._assigned[first.id][0]
    assert rec_p == {"policy_choice": chosen, "executed": chosen} == p.last_override
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
    rec_g = {}
    override_next_decision(g, a, record=rec_g)
    g2 = g.fork(str(tmp_path / "G2"))
    _finish(g2)  # the copy applies its override to itself; g and its pending override are untouched
    assert sf._FORCE_ATTR in g.sim._scheduler.__dict__ and "_call" in g.sim._scheduler.__dict__

    # Counterfactual by construction: the policy itself picks `a` for the first request (its _select returns a
    # after running as usual, so its rng draws are unchanged); no override.
    c = o.fork(str(tmp_path / "C"))
    mod = load_policy(POLICY)  # c's own module instance: patching the shared one would (rightly) invalidate the family
    c.sim._scheduler._policy = mod
    c.family, c.policy_fp = {"forked": False}, sf._module_fingerprint(mod)
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
    assert rec_g == {"policy_choice": chosen, "executed": a} == g.last_override == g2.last_override
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
    # conftest.reset_id_counters (used by earlier tests in the same process) sets these class counters; a handle
    # must leave whatever the process had untouched.
    before = {c: c.__dict__.get("_id", sf.MISSING) for c in lazy}
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
    assert {c: c.__dict__.get("_id", sf.MISSING) for c in lazy} == before  # nothing leaks outside a handle

    # The predictor is shared; its clamp counts are per handle.
    pred = a._predictor()
    outside = pred.clamp_hits
    with d.active():
        pred.clamp_hits["probe"] = 1
    assert d.clamp_hits == {**a.clamp_hits, "probe": 1} and "probe" not in a.clamp_hits
    assert pred.clamp_hits is outside and "probe" not in outside
    assert "probe" in d.fork().clamp_hits


# ------------------------------------------------------------------ guards (unit level, no simulation)

def _stub(gs, policy_fp=None):
    h = SimHandle.__new__(SimHandle)
    h.sim = types.SimpleNamespace(_scheduler=gs)
    h.policy_fp = policy_fp
    return h


def _evolved(mod):
    gs = object.__new__(EvolvedGlobalScheduler)
    gs._policy = mod
    return gs


def _load(tmp_path, name, src):
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(src))
    return load_policy(str(f))


def test_learned_scheduler_refused():
    learned = object.__new__(LearnedGlobalScheduler)
    with pytest.raises(NotImplementedError, match="learned"):
        _stub(learned).fork()
    with pytest.raises(NotImplementedError):
        override_next_decision(_stub(learned), 0)


# Ways a policy module can keep state outside ctx.state (review E r2), each mutated by a call to choose().
STATEFUL = {
    "global_counter": """
        _CALLS = 0
        def choose(request, replicas, ctx):
            global _CALLS
            _CALLS += 1
            return 0
        """,
    "numpy_array": """
        import numpy as np
        _W = np.zeros(4)
        def choose(request, replicas, ctx):
            _W[0] += 1
            return 0
        """,
    "module_rng": """
        import random
        _R = random.Random(0)
        def choose(request, replicas, ctx):
            return _R.randrange(len(replicas))
        """,
    "namespace": """
        import types
        _NS = types.SimpleNamespace(n=0)
        def choose(request, replicas, ctx):
            _NS.n += 1
            return 0
        """,
    "class_attribute": """
        class _S:
            hits = 0
        def choose(request, replicas, ctx):
            _S.hits += 1
            return 0
        """,
    "class_container": """
        class _S:
            seen = []
        def choose(request, replicas, ctx):
            _S.seen.append(1)
            return 0
        """,
    "closure": """
        def _make():
            seen = []
            def choose(request, replicas, ctx):
                seen.append(1)
                return 0
            return choose
        choose = _make()
        """,
    "lru_cache": """
        import functools
        @functools.lru_cache(maxsize=None)
        def _f(x):
            return 0
        def choose(request, replicas, ctx):
            return _f(request)
        """,
    "mutable_default": """
        def choose(request, replicas, ctx, _seen=[]):
            _seen.append(1)
            return 0
        """,
    "classmethod_counter": """
        class _T:
            n = 0
            @classmethod
            def bump(cls):
                cls.n += 1
        def choose(request, replicas, ctx):
            _T.bump()
            return 0
        """,
    "alias_store": """
        class _C:
            n = 0
        def choose(request, replicas, ctx):
            c = _C
            c.n += 1
            return 0
        """,
    "alias_function_dict": """
        def choose(request, replicas, ctx):
            f = choose
            f.__dict__["calls"] = f.__dict__.get("calls", 0) + 1
            return 0
        """,
    "class_via_instance": """
        class _C:
            n = 0
            def bump(self):
                type(self).n += 1
        def choose(request, replicas, ctx):
            _C().bump()
            return 0
        """,
    "function_attribute": """
        def choose(request, replicas, ctx):
            choose.calls = getattr(choose, "calls", 0) + 1
            return 0
        """,
}


@pytest.mark.parametrize("case", sorted(STATEFUL))
def test_stateful_policy_module_refused(tmp_path, case):
    mod = _load(tmp_path, case, STATEFUL[case])
    assert sf._module_state_issues(mod), case
    fp = sf._module_fingerprint(mod)
    with pytest.raises(NotImplementedError, match="outside ctx.state"):
        _stub(_evolved(mod), fp).fork()
    assert sf._unforkable_reason(_stub(_evolved(mod), fp), trust_module_state=True) is None
    mod.choose(1, [0, 0, 0, 0], types.SimpleNamespace(state={}))
    assert sf._module_fingerprint(mod) != fp, case  # the runtime check sees the mutation too


def test_policy_fingerprint_checked_at_fork(tmp_path):
    mod = _load(tmp_path, "clean", """
        LIMIT = 3
        def choose(request, replicas, ctx):
            return 0
        """)
    assert not sf._module_state_issues(mod)
    h = _stub(_evolved(mod), sf._module_fingerprint(mod))
    assert sf._unforkable_reason(h) is None
    mod.LIMIT = 4  # state changed by a path the static scan cannot see
    with pytest.raises(NotImplementedError, match="changed its module-level state"):
        h.fork()


@pytest.mark.parametrize("path", sorted(str(p.relative_to(ev.REPO)) for p in
                                        list((ev.REPO / "evolve/finalists").glob("*.py"))
                                        + list((ev.REPO / "evolve/policies").rglob("*.py"))
                                        if not p.name.startswith("rl_")))
def test_repo_policies_pass_module_scan(path):
    assert sf._module_state_issues(load_policy(str(ev.REPO / path), require_router=False)) == []


HARMLESS = {  # E r4 minor 2: stores of module constants into locals/ctx.state are not aliases of shared objects
    "local_item_const": """
        INELIGIBLE = 1e30
        def choose(request, replicas, ctx):
            costs = [0.0] * len(replicas)
            costs[0] = INELIGIBLE
            return costs.index(min(costs))
        """,
    "ctx_state_const": """
        INIT = 0.5
        def choose(request, replicas, ctx):
            ctx.state["n"] = ctx.state.get("n", 0) + 1
            ctx.state["ewma"] = INIT
            return 0
        """,
    "local_dict_math": """
        import math
        def choose(request, replicas, ctx):
            d = {}
            d["k"] = math.pi
            d["j"] = d["k"]
            return 0
        """,
}


@pytest.mark.parametrize("case", sorted(HARMLESS))
def test_harmless_stores_pass_module_scan(tmp_path, case):
    assert sf._module_state_issues(_load(tmp_path, case, HARMLESS[case])) == []


def test_reviewed_candidate_passes_module_scan():
    f = ev.REPO / "evolve/runs/t4_local_r3/programs/33c06df4220e.py"
    if not f.exists():
        pytest.skip("candidate not present")
    assert sf._module_state_issues(load_policy(str(f), require_router=False)) == []


def _views(*outstanding):
    return [types.SimpleNamespace(outstanding_requests=o) for o in outstanding]


def test_reconcile_requires_an_exact_mechanism(tmp_path):
    req = types.SimpleNamespace(num_prefill_tokens=7, prefix_hashes=(1, 2))
    ctx = types.SimpleNamespace(num_replicas=4, state={})

    # SGLang: the prefix index is unchanged at its own pick (hashes already indexed there), but forcing replica 0
    # would have indexed them at 0. "Unchanged" is not exact: raise; and it is not a reviewed "independent" policy.
    sglang = load_policy(str(ev.REPO / "evolve/policies/systems/sglang_cache_aware.py"))
    ctx.state = {"sglang_index": [set(), {1, 2}, set(), set()]}
    before = copy.deepcopy(ctx.state)
    for mode in ("auto", "independent"):
        with pytest.raises(NotImplementedError, match="cannot force"):
            sf._reconcile(sglang, before, req, _views(0, 1, 0, 0), 1, 0, ctx, mode)

    clean = _load(tmp_path, "nostate", "def choose(request, replicas, ctx):\n    return 0\n")
    ctx.state = {}
    sf._reconcile(clean, {}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto")  # no state anywhere: exact
    ctx.extra = 1  # state on a ctx attribute (outside ctx.state) is never reconciled
    with pytest.raises(NotImplementedError, match="ctx attributes"):
        sf._reconcile(clean, {}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto")
    del ctx.extra
    with pytest.raises(NotImplementedError, match="ctx attributes"):  # present before the call only
        sf._reconcile(clean, {}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto", {"last_pick": 1})

    # "independent": only for reviewed policies (allowlist by source sha256)
    ctx.state = {"sessions": {5: (1.0, 7)}}
    with pytest.raises(NotImplementedError, match="accepted only for reviewed policies"):
        sf._reconcile(clean, {"sessions": {}}, req, _views(0, 0, 0, 0), 1, 0, ctx, "independent")
    with pytest.raises(NotImplementedError):
        sf._reconcile(clean, {"sessions": {}}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto")
    claude = load_policy(str(ev.REPO / "evolve/finalists/claude_r1_76be082eb0f5.py"))
    sf._reconcile(claude, {"sessions": {}}, req, _views(0, 0, 0, 0), 1, 0, ctx, "independent")
    # early session ids look like replica indices (E r3 nit): no false refusal
    ctx.state = {"sessions": {0: (0.0, 900), 1: (0.5, 1200), 2: (1.0, 4000), 3: (1.5, 800)}}
    sf._reconcile(claude, {"sessions": {0: (0.0, 900), 1: (0.5, 1200), 2: (1.0, 4000)}}, req,
                  _views(0, 0, 0, 0), 1, 0, ctx, "independent")
    ctx.state = {"sessions": {5: (1.0, 7)}, "affinity": {5: 1}}  # second guard: a value equal to the pick
    with pytest.raises(NotImplementedError, match="involves the pick"):
        sf._reconcile(claude, {"sessions": {}, "affinity": {}}, req, _views(0, 0, 0, 0), 1, 0, ctx, "independent")

    hooked = _load(tmp_path, "hooked", """
        def choose(request, replicas, ctx):
            ctx.state.setdefault("idx", [set() for _ in replicas])[1].update(request.prefix_hashes)
            return 1
        def on_forced(request, replicas, chosen, forced, ctx):
            ctx.state["forced"] = (chosen, forced)
        """)
    ctx.state = {"idx": [set(), {1, 2}, set(), set()]}
    sf._reconcile(hooked, {}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto")
    assert ctx.state["forced"] == (1, 0)


def test_pick_dependent_change_guard():
    pick, num = 1, 4
    cases = {  # E r3 major: state shapes whose update records the pick (each before -> after)
        "mooncake_burst": ((0.0, [set(), set(), set(), set()]), (0.0, [set(), {7}, set(), set()])),
        "nested_per_replica": ({"q": [0, 0, 0, 0]}, {"q": [0, 5, 0, 0]}),
        "session_affinity": ({}, {12: 1}),
        "last_pick": (0, 1),
        "global_fifo": ([(0, 300)], [(0, 300), (1, 700)]),
        "pick_history": (deque([0], maxlen=16), deque([0, 1], maxlen=16)),
        "replica_set": (set(), {1}),
    }
    for name, (old, new) in cases.items():
        assert sf._pick_dependent_change({"k": old}, {"k": new}, pick, num), name
    # pick-independent updates (claude_r1 / completion_r2 / t4_70b shapes) pass
    assert sf._pick_dependent_change({"sessions": {1: (0.5, 800)}}, {"sessions": {1: (2.0, 900), 3: (2.0, 50)}},
                                     pick, num) is None
    assert sf._pick_dependent_change({}, {"session_prompt": {1: 900.0}, "scarcity_ewma": 0.3}, pick, num) is None
    assert sf._pick_dependent_change({"prompt_ewma": 1.0}, {"prompt_ewma": 1.5}, pick, num) is None


TAIL4 = (ev.REPO / "evolve/finalists/tail4_ni.py").read_text()
FIFO_VARIANTS = {  # E r2 minor 3: FIFO-shaped policies whose commit is not "move n from pick to forced"
    "uncached_entry": TAIL4.replace("    queues[pick].append(n)\n    attended[pick] += n\n",
                                    "    u = n - request.cached_prefix_tokens[pick]\n"
                                    "    queues[pick].append(u)\n    attended[pick] += u\n"),
    "conditional_append": TAIL4.replace("    queues[pick].append(n)\n    attended[pick] += n\n",
                                        "    if not queues[pick] or queues[pick][-1] != n:\n"
                                        "        queues[pick].append(n)\n        attended[pick] += n\n"),
    "pick_dependent_trim": TAIL4.replace("    queues[pick].append(n)\n",
                                         "    while len(queues[pick]) > 2:\n"
                                         "        attended[pick] -= queues[pick].popleft()\n"
                                         "    queues[pick].append(n)\n"),
}


def test_fifo_commit_recognition(tmp_path):
    assert sf._fifo_policy(load_policy(str(ev.REPO / "evolve/finalists/tail4_ni.py")))
    req = types.SimpleNamespace(num_prefill_tokens=1000)
    ctx = types.SimpleNamespace(num_replicas=2, state=None)
    for name, src in FIFO_VARIANTS.items():
        assert src != TAIL4, name
        mod = _load(tmp_path, name, src)
        assert not sf._module_state_issues(mod) and not sf._fifo_policy(mod), name
        # a state change that the shape check alone would accept: trimmed nothing, appended n at 0
        ctx.state = {"attn_queues": [deque([500, 1000]), deque([700])], "attn_sums": [1500, 700]}
        with pytest.raises(NotImplementedError, match="cannot force"):
            sf._reconcile(mod, {"attn_queues": [deque([500]), deque([700])], "attn_sums": [500, 700]},
                          req, _views(5, 5), 0, 1, ctx, "auto")

    def move(q0, q1, views, chosen=0, forced=1):
        def sums(qs):
            return [sum(x) if all(isinstance(y, int) for y in x) else 0 for x in qs]
        before = {"attn_queues": [deque(x) for x in q0], "attn_sums": sums(q0)}
        after = {"attn_queues": [deque(x) for x in q1], "attn_sums": sums(q1)}
        ok = sf._fifo_move(before, after, req, views, chosen, forced)
        return ok, [list(q) for q in after["attn_queues"]], after["attn_sums"]

    # trimmed to min(len, outstanding) everywhere, then n appended at the pick: moved
    assert move([[300, 400], [600, 700]], [[400, 1000], [700]], _views(1, 1)) == (
        True, [[400], [700, 1000]], [400, 1700])
    assert not move([[300], [1000]], [[300], [1000]], _views(5, 5))[0]  # no append: an old entry would move
    assert not move([[400], [600, 700]], [[400, 1000], [700]], _views(5, 5))[0]  # trim not explained by the views
    assert not move([[(1, 2)], []], [[(1, 2), 1000], []], _views(5, 5))[0]  # tuple entries: False, not TypeError


def test_route_bursts_and_record_only(tmp_path):
    pol = tmp_path / "route_first.py"
    pol.write_text("def route(burst, replicas, ctx):\n    return [0] * len(burst)\n")
    argv = ev.sim_command("evolved", str(pol), SC, ev.window_csv(SC), tmp_path / "R",
                          ["--no-metrics_config_store_plots"])[3:]
    h = build_simulator(argv)
    assert h.run_to_decision(2) and len(h.sim._scheduler._request_queue) > 1
    with pytest.raises(NotImplementedError, match="route_first_only"):
        override_next_decision(h, 1)
    rec = {}
    override_next_decision(h, 1, record=rec, route_first_only=True)
    burst = sorted(h.sim._scheduler._request_queue, key=lambda r: (r.arrived_at, r.id))
    assert h.run(lambda x: x.decisions >= 2)
    gs = h.sim._scheduler
    assert rec == {"policy_choice": 0, "executed": 1}
    assert [gs._assigned[r.id][0] for r in burst] == [1] + [0] * (len(burst) - 1)


def test_sglang_forced_decision_raises(tmp_path):
    pol = ev.REPO / "evolve/policies/systems/sglang_cache_aware.py"
    argv = ev.sim_command("evolved", str(pol), SC, ev.window_csv(SC), tmp_path / "S",
                          ["--no-metrics_config_store_plots"])[3:]
    h = build_simulator(argv)
    assert h.run_to_decision(3)
    p = h.fork()
    rec = {}
    override_next_decision(p, None, record=rec)
    assert p.run(lambda x: x.decisions >= 3)
    override_next_decision(h, (rec["policy_choice"] + 1) % SC.replicas)
    with pytest.raises(NotImplementedError, match="cannot force"):
        h.run(lambda x: x.decisions >= 3)


def test_module_state_change_after_fork_is_caught(tmp_path):
    # Module state mutated through a path the static scan does not see, starting only after t = 30 s: the fork at
    # decision 2 is allowed, and the change is caught when a run of the family returns.
    pol = tmp_path / "lazy.py"
    pol.write_text(textwrap.dedent("""
        class _C:
            n = 0
        def choose(request, replicas, ctx):
            if ctx.now > 30.0:
                type.__setattr__(_C, "n", _C.n + 1)
            loads = [r.outstanding_requests for r in replicas]
            return loads.index(min(loads))
        """))
    mod = load_policy(str(pol))
    assert not sf._module_state_issues(mod)
    argv = ev.sim_command("evolved", str(pol), SC, ev.window_csv(SC), tmp_path / "L",
                          ["--no-metrics_config_store_plots"])[3:]
    h = build_simulator(argv)
    assert h.run_to_decision(2)
    f = h.fork()
    with pytest.raises(RuntimeError, match="changed its module-level state during a run"):
        f.run(lambda x: x.decisions >= 12)
    with pytest.raises(NotImplementedError, match="changed its module-level state"):
        h.fork()
    # a forced decision is refused too once the module state differs from the build's
    req = types.SimpleNamespace(num_prefill_tokens=7)
    ctx = types.SimpleNamespace(num_replicas=4, state={})
    with pytest.raises(NotImplementedError, match="changed since the handle was built"):
        sf._reconcile(h.sim._scheduler._policy, {}, req, _views(0, 0, 0, 0), 1, 0, ctx, "auto", {}, h.policy_fp)


def test_os_fork_inside_run_callback_keeps_global_rng(tmp_path):
    import hashlib
    import json
    import os
    import pickle
    import random

    h = _build(tmp_path / "O")
    seen = {}

    def cb(x):
        if x.at_decision() and x.decisions == 4:
            seen["parent"] = hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    raw = hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()
                    sf.after_os_fork(x)
                    out = {"raw": raw, "restored": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                           "rec": None}
                    rec = {}
                    override_next_decision(x, None, record=rec)
                    out["cancel"] = [sf.cancel_override(x), sf.cancel_override(x)]
                    override_next_decision(x, 0, record=rec)
                    with os.fdopen(w, "w") as fw:
                        json.dump(out, fw)
                finally:
                    os._exit(0)
            os.close(w)
            with os.fdopen(r) as fr:
                seen["child"] = json.load(fr)
            os.waitpid(pid, 0)
            return True
        return False

    assert h.run(cb)
    c = seen["child"]
    assert c["raw"] == seen["parent"] == c["restored"]  # restored automatically (register_at_fork) and by the helper
    assert c["cancel"] == [True, False]


def test_pending_override_passes_observe_calls_through(tmp_path):
    # v4.1: the scheduler calls observe() with observe=True; a pending override must pass it through untouched.
    pol = str(ev.REPO / "evolve/golden/policies/feedback.py")
    argv = ev.sim_command("evolved", pol, SC, ev.window_csv(SC), tmp_path / "F", ["--no-metrics_config_store_plots"])
    h = build_simulator(argv[3:])
    for _ in range(8):
        assert _advance(h)
    gs = h.sim._scheduler
    n = gs.observe_calls
    rec = {}
    override_next_decision(h, None, record=rec)
    assert _advance(h)
    assert gs.observe_calls > n  # observe() ran while the override was pending
    assert rec["executed"] == rec["policy_choice"]
    _finish(h, out=False)
