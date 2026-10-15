"""LLMServingSim 2.0 adapter (evolve/lss2): converter and router logic, without the simulator.

The router is exercised against fake LLMServingSim schedulers (waiting/running lists, an NPU pool, a prefix-cache
probe, ``add_request``/``add_done``), so these tests need neither LLMServingSim nor ASTRA-Sim.
"""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evolve.lss2 import convert as cv
from evolve.lss2 import router as rt

REPO = Path(__file__).resolve().parents[1]
WINDOWS = REPO / "evolve" / ".cache_v4" / "windows"


def _small_window(tmp_path, rows):
    df = pd.DataFrame(rows)
    df["block_hash_ids"] = df["block_hash_ids"].map(json.dumps)
    df["block_size"] = 16
    p = tmp_path / "Tool_0_4_test.csv"
    df.to_csv(p, index=False)
    return p


# ------------------------------------------------------------------ converter

def test_tokens_preserve_prefix_sharing():
    a = {"row": 0, "num_prefill_tokens": 40, "block_hash_ids": [7, 8, 100]}
    b = {"row": 1, "num_prefill_tokens": 37, "block_hash_ids": [7, 9, 101]}
    ta = cv.expand_tokens(cv.tok_block_ids(a), 40)
    tb = cv.expand_tokens(cv.tok_block_ids(b), 37)
    assert len(ta) == 40 and len(tb) == 37
    assert ta[:16] == tb[:16] == [7 * 16 + j for j in range(16)]  # shared first block
    assert ta[16:32] != tb[16:32]
    assert len(set(ta)) == 40  # ids within a request distinct
    # block ids missing from the trace become negative ids unique to the row
    c = {"row": 5, "num_prefill_tokens": 50, "block_hash_ids": [7]}
    ids = cv.tok_block_ids(c)
    assert ids[0] == 7 and len(ids) == 4 and all(h < 0 for h in ids[1:]) and len(set(ids)) == 4
    assert cv.prompt_hashes(a) == [7, 8]  # full prompt blocks only (Vidur's prompt_block_hashes)


def test_replay_matches_vidur_generator(tmp_path):
    from vidur.config import TraceRequestGeneratorConfig
    from vidur.request_generator.trace_request_generator import TraceRequestGenerator
    rng = np.random.default_rng(3)
    rows = []
    for k in range(60):
        n, d = int(rng.integers(20, 300)), int(rng.integers(1, 50))
        rows.append({"arrived_at": float((k // 8) * 3.0), "num_prefill_tokens": n, "num_decode_tokens": d,
                     "block_hash_ids": list(range(1000 * k, 1000 * k + (n + d) // 16)), "session_id": k % 7})
    p = _small_window(tmp_path, rows)
    for scale, pert in ((0.75, 0), (0.75, 1), (1.3, 4)):
        jit = cv.vidur_jitter("Tool", pert)
        cfg = TraceRequestGeneratorConfig(trace_file=str(p), time_scale_factor=scale, arrival_jitter=jit,
                                          seed=100 * pert + 1, max_tokens=cv.MAX_TOKENS_L3_8B)
        gen = TraceRequestGenerator(cfg)
        ref = []
        while (r := gen.get_next_request()) is not None:
            ref.append((r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens, r.session_id))
        ours = cv.replay(cv.read_window(str(p)), scale, pert, jit)
        assert [(r["arrival_s"], r["num_prefill_tokens"], r["num_decode_tokens"], r["session_id"])
                for r in ours] == ref


def test_jsonl_formats_agree(tmp_path):
    rows = [{"arrived_at": 0.0, "num_prefill_tokens": 40, "num_decode_tokens": 5, "block_hash_ids": [1, 2, 3],
             "session_id": 4},
            {"arrived_at": 1.5, "num_prefill_tokens": 20, "num_decode_tokens": 3, "block_hash_ids": [1, 9],
             "session_id": None}]
    p = _small_window(tmp_path, rows)
    rr = cv.replay(cv.read_window(str(p)), 2.0)
    cv.write_jsonl(rr, str(tmp_path / "f.jsonl"), "full")
    cv.write_jsonl(rr, str(tmp_path / "c.jsonl"), "compact")
    full = [json.loads(x) for x in open(tmp_path / "f.jsonl")]
    comp = [json.loads(x) for x in open(tmp_path / "c.jsonl")]
    assert [f["arrival_time_ns"] for f in full] == [0, 3_000_000_000]
    for f, c in zip(full, comp):
        assert f["input_tok_ids"] == cv.expand_tokens(c["tok_block_ids"], c["input_toks"])
    assert full[0]["prompt_hashes"] == [1, 2] and full[1]["session_id"] is None


def test_cluster_config_shape():
    c = cv.cluster_config(8, 0.77)
    inst = c["nodes"][0]["instances"]
    assert c["nodes"][0]["num_instances"] == 8 and len(inst) == 8
    assert {i["hardware"] for i in inst} == {"RTXPRO6000"} and inst[0]["npu_mem"]["mem_util"] == 0.77


@pytest.mark.skipif(not WINDOWS.exists(), reason="no window cache")
def test_real_window_ids_are_prefix_unique():
    ws = sorted(WINDOWS.glob("Tool_*_200_*.csv")) + sorted(WINDOWS.glob("Conv_*_200_*.csv"))
    if not ws:
        pytest.skip("no small windows")
    for w in ws[:2]:
        assert cv.check_prefix_unique(cv.read_window(str(w))) == 0


# ------------------------------------------------------------------ fakes for the router

class FakePool:
    def __init__(self, blocks=1000, used=0):
        self.num_blocks, self.block_size, self.used = blocks, 16, used

    def usage(self):
        return self.used / self.num_blocks


class FakeReq:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None,
                 is_init=True):
        self.id, self.input, self.output, self.arrival = id, input, output, arrival
        self.original_input = input
        self.input_hash_ids = input_hash_ids or []
        self.block_hashes = None
        self.ttft, self.latency, self.itl = -1, -1, []
        self.npu_cache_hit = 0
        self.num_preemptions = 0


class FakeKV:
    """Prefix cache = set of cached token-block tuples (per replica)."""

    def __init__(self):
        self.npu_pool = FakePool()
        self.cached = set()

    def get_computed_blocks(self, req):
        toks = req.input_hash_ids
        n = 0
        maxb = (req.input - 1) // 16
        while n < maxb and tuple(toks[16 * n:16 * n + 16]) in self.cached:
            n += 1
        return [None] * n, 16 * n, 0


class FakeBatch:
    def __init__(self, bid, reqs):
        self.batch_id, self.requests = bid, reqs


class FakeSched:
    def __init__(self, i):
        self.instance_id, self.model, self.enable_prefix_caching = i, "m", True
        self.waiting, self.running, self.inflight, self.done = [], [], [], []
        self.kv = FakeKV()

    def add_request(self, req, is_init=True):
        self.waiting.append(FakeReq(*req))

    def add_done(self, id, sys_, finish):
        """Complete batch id-1: every request in it gets its first token; requests whose output target is reached
        (``output`` = prompt + 1 here) finish."""
        b = next(b for b in self.inflight if b.batch_id == id - 1)
        self.inflight.remove(b)
        ended = []
        for r in b.requests:
            if r.ttft < 0:
                r.ttft = finish - r.arrival
                r.npu_cache_hit = 16
            if r.output == r.input + 1:
                r.latency = finish - r.arrival
                ended.append(r)
                self.running.remove(r)
        return 0, 0, ended

    def run_batch(self, bid):
        reqs = list(self.waiting)
        self.waiting.clear()
        self.running += reqs
        self.inflight.append(FakeBatch(bid, reqs))
        return reqs


class FakeBaseRouter:
    """The parts of serving.core.router.Router the policy router relies on."""

    def __init__(self, num_instances, schedulers, req_num, routing_policy="RR", seed=42):
        self.schedulers = schedulers
        self.prefill_schedulers = schedulers
        self.decode_schedulers = []
        self.req_num = req_num
        self._pending_requests, self._pending_idx, self._next_request_id = [], 0, 0


class Recorder:
    def __init__(self, pick):
        self.pick, self.calls, self.outcomes = pick, [], []

    def choose(self, request, replicas, ctx):
        self.calls.append((request, replicas, ctx.now))
        return self.pick(request, replicas)

    def observe(self, outcome, ctx):
        self.outcomes.append((outcome, ctx.now))


def _router(policy, R=3, seed=1):
    cls = rt.make_router_class(FakeBaseRouter, FakeReq, policy, seed)
    scheds = [FakeSched(i) for i in range(R)]
    return cls(R, scheds, 0), scheds


def _jsonl(tmp_path, recs):
    p = tmp_path / "w.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(p)


def _rec(t_s, n, out, blocks, session=None, row=0):
    return {"input_toks": n, "output_toks": out, "arrival_time_ns": int(t_s * 1e9), "arrival_s": t_s, "row": row,
            "session_id": session, "prompt_hashes": blocks[: n // 16], "tok_block_ids": blocks}


def test_views_counters_and_feedback(tmp_path):
    rec = Recorder(lambda req, reps: 1 if req.position_in_burst == 0 else 2)
    r, scheds = _router(rec)
    # replica 1 already holds block 5 in its prefix cache
    scheds[1].kv.cached.add(tuple(5 * 16 + j for j in range(16)))
    scheds[0].running = [object(), object()]
    scheds[2].kv.npu_pool.used = 250
    path = _jsonl(tmp_path, [_rec(1.0, 40, 1, [5, 6, 7], session=11, row=0),
                             _rec(1.0, 33, 1, [5, 8, 9], session=11, row=1),
                             _rec(2.0, 20, 1, [5, 10], session=11, row=2)])
    r.load_requests(path, enable_prefix_caching=True)
    assert r.route_arrived_requests(int(1.5e9)) == 2  # the burst at t = 1 s
    (q0, v0, now0), (q1, v1, now1) = rec.calls
    assert now0 == now1 == 1.0
    assert (q0.position_in_burst, q0.burst_size, q1.position_in_burst) == (0, 2, 1)
    assert q0.cached_prefix_tokens == (0, 16, 0) and q1.cached_prefix_tokens == (0, 16, 0)
    assert q0.prefix_hashes == (5, 6) and q0.block_tokens == 16 and q0.session_id == 11
    assert q0.session_last_replica == -1 and q1.session_last_replica == 1  # refreshed within the burst
    # the burst's ledgers are taken before any of it is placed (as Vidur)
    assert q1.inflight_prefix_tokens == (0, 0, 0) and q1.active_prefix_tokens == (0, 0, 0)
    assert [v.running for v in v0] == [2, 0, 0] and v0[2].kv_usage == 0.25 and v0[0].kv_capacity_tokens == 16000
    assert [v.outstanding_requests for v in v1] == [0, 1, 0]
    assert [v.pending_prefill_tokens for v in v1] == [0, 40 - 16, 0]
    assert [v.waiting for v in v1] == [0, 0, 0]  # waiting excludes the burst being routed
    assert len(scheds[1].waiting) == 1 and len(scheds[2].waiting) == 1
    assert scheds[1].waiting[0].block_hashes is None  # fakes do not hash; the router only shares what exists
    # prefill + end of request 0 on replica 1 (batch 0), then route the t = 2 s request
    reqs = scheds[1].run_batch(0)
    scheds[1].add_done(1, 0, int(1.2e9))
    (o_ft, now_ft), (o_end, now_end) = rec.outcomes
    assert o_ft.event == "first_token" and o_ft.replica == 1 and abs(now_ft - 1.2) < 1e-12
    assert o_ft.num_cached_tokens == 16 and o_ft.num_decode_tokens is None and o_ft.completed_at is None
    assert o_end.event == "end" and o_end.num_decode_tokens == 1 and o_end.arrived_at == 1.0
    assert r._outstanding == [0, 0, 1] and r._pending_prefill == [0, 0, 33]
    rec.pick = lambda req, reps: 0
    assert r.route_arrived_requests(int(2e9)) == 1
    q2, v2, _ = rec.calls[-1]
    assert q2.session_last_replica == 2 and q2.burst_size == 1
    # request 1 (blocks 5, 8) is still in flight on replica 2: block 5 counts there
    assert q2.inflight_prefix_tokens == (0, 0, 16) and q2.active_prefix_tokens == (0, 0, 16)
    assert [v.outstanding_requests for v in v2] == [0, 0, 1]
    st = r.stats()
    assert st["n_routed"] == 3 and st["policy_calls"] == 3 and st["observe_calls"] == 2
    assert r.records[0]["e2e_s"] == pytest.approx(0.2) and r.records[0]["probe_cached"] == 16


def test_route_policy_and_errors(tmp_path):
    class RoutePolicy:
        @staticmethod
        def route(burst, replicas, ctx):
            return [len(replicas) - 1] * len(burst)

    r, scheds = _router(RoutePolicy())
    r.load_requests(_jsonl(tmp_path, [_rec(0.0, 20, 1, [1, 2]), _rec(0.0, 20, 1, [3, 4])]), True)
    r.route_arrived_requests(0)
    assert len(scheds[2].waiting) == 2 and r._outstanding == [0, 0, 2]

    r2, _ = _router(Recorder(lambda req, reps: 7))
    r2.load_requests(_jsonl(tmp_path, [_rec(0.0, 20, 1, [1, 2])]), True)
    with pytest.raises(rt.PolicyError):
        r2.route_arrived_requests(0)


def test_builtins_match_policy_files():
    import importlib.util
    rng = np.random.default_rng(0)
    for name, fname in (("rr", "round_robin"), ("lor", "lor"), ("sticky_lor", "sticky_lor"),
                        ("lop_uncached", "lop_uncached")):
        spec = importlib.util.spec_from_file_location("p", REPO / "evolve" / "policies" / f"{fname}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import random
        ca, cb = rt.PolicyContext(random.Random(5), 4), rt.PolicyContext(random.Random(5), 4)
        for _ in range(50):
            reps = [rt.ReplicaView(i, int(rng.integers(0, 3)), int(rng.integers(0, 3)) * 100, 0, 0, 0.0, 1000)
                    for i in range(4)]
            req = rt.RequestView(0.0, 300, 1, tuple(int(x) * 16 for x in rng.integers(0, 3, 4)),
                                 int(rng.integers(-1, 4)), 0, 1)
            assert rt.BUILTINS[name](req, reps, ca) == mod.choose(req, reps, cb)


def test_view_copies_match_vidur():
    from vidur.scheduler.global_scheduler import evolved_global_scheduler as egs
    for ours, theirs in ((rt._ReplicaViewCopy, egs.ReplicaView), (rt._RequestViewCopy, egs.RequestView),
                         (rt._RequestOutcomeCopy, egs.RequestOutcome)):
        fo = [(f.name, f.type, f.default) for f in dataclasses.fields(ours)]
        ft = [(f.name, f.type, f.default) for f in dataclasses.fields(theirs)]
        assert fo == ft
        assert ours.__dataclass_params__.frozen and theirs.__dataclass_params__.frozen
    ctx = rt._PolicyContextCopy("rng", 3)
    ref = egs.PolicyContext("rng", 3)
    assert vars(ctx) == vars(ref)


ES_CHECK = r'''
import json, random, sys
import numpy as np
sys.path.insert(0, {repo!r})
stub = sys.argv[1] == "stub"
from evolve.lss2 import router as rt
if stub:
    rt._install_es_stubs()
pol = rt._exec_policy({policy!r})
rng = np.random.default_rng(1)
ctx = rt.PolicyContext(random.Random(3), 4)
picks = []
for k in range(200):
    reps = [rt.ReplicaView(i, int(rng.integers(0, 20)), int(rng.integers(0, 30000)), int(rng.integers(0, 5)),
            int(rng.integers(0, 20)), float(rng.random()), 483328) for i in range(4)]
    n = int(rng.integers(100, 20000))
    h = tuple(int(x) for x in rng.integers(0, 5, n // 16))
    req = rt.RequestView(float(k), n, k % 9, tuple(int(x) for x in rng.integers(0, n, 4)), int(rng.integers(-1, 4)),
                         0, 1, h, 16, (0,) * 4, (0,) * 4)
    c = pol.choose(req, reps, ctx)
    picks.append(c)
    if k % 3 == 0:
        pol.observe(rt.RequestOutcome("end", c, float(k), float(k) + 1, float(k) + 2, n, 0, int(rng.integers(1, 900)),
                    k % 9), ctx)
print(json.dumps({{"picks": picks, "moved": ctx.state["es"]["moved"],
                  "stubbed": bool(getattr(sys.modules.get("evolve.sim_fork"), "__lss2_stub__", False))}}))
'''


@pytest.mark.parametrize("name", ["es_lin9_20", "es_mlp9_10"])
def test_es_policy_identical_with_stubs(name):
    """The ES policies give the same decisions when their Vidur-stack imports are stubbed (the lss2 env)."""
    policy = str(REPO / "evolve" / "finalists" / "rl_es" / f"{name}.py")
    code = ES_CHECK.format(repo=str(REPO), policy=policy)
    out = {}
    for mode in ("native", "stub"):
        p = subprocess.run([sys.executable, "-c", code, mode], capture_output=True, text=True, cwd=str(REPO))
        assert p.returncode == 0, p.stderr[-2000:]
        out[mode] = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["stub"]["stubbed"] and not out["native"]["stubbed"]
    assert out["stub"]["picks"] == out["native"]["picks"]
    assert out["stub"]["moved"] > 0  # the learned correction is active, not plain 4873
