"""v4 robustness: trace/cache block-size assert (KV-10), side-effect-free router probe (GS-15), explicit output
writing with an end-state file and predictor provenance (SIM-3 atexit, KV-7 clamp counter), trace lint (KV-1)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO_ROOT, base_args, build_simulator, run_simulator

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lint_trace import lint  # noqa: E402


def _rehash_at_32(tiny_trace, path):
    """The tiny trace made internally consistent at 32-token blocks (KV-10's case: valid on its own terms, so the
    per-request Request assertion passes; only the trace/cache block-size check can catch it)."""
    df = pd.read_csv(tiny_trace)
    df["block_hash_ids"] = [str(json.loads(ids)[:(p + d) // 32])
                            for ids, p, d in zip(df.block_hash_ids, df.num_prefill_tokens, df.num_decode_tokens)]
    df["block_size"] = 32
    df.to_csv(path, index=False)
    return path


def test_trace_block_size_must_match_cache(tiny_trace, tmp_path):
    bad = _rehash_at_32(tiny_trace, tmp_path / "bs32.csv")
    with pytest.raises(ValueError, match=r"Trace block_size \[32\] does not match the KV cache block size 16"):
        build_simulator(base_args(bad, tmp_path / "out"))
    # Without the validator the same trace builds silently (it would replay with wrong prefix matches), so the
    # error above comes from _validate_block_size, not from a per-request assertion.
    from vidur.simulator import Simulator
    orig = Simulator._validate_block_size
    Simulator._validate_block_size = lambda self: None
    try:
        build_simulator(base_args(bad, tmp_path / "out2"))
    finally:
        Simulator._validate_block_size = orig


def test_block_size_check_runs_before_requests_are_built(tiny_trace, tmp_path):
    """An inconsistent trace (ids hashed at 16, block_size column 32) would trip Request's own assertion when the
    first request is built; the block-size check must fire first, with its specific error."""
    df = pd.read_csv(tiny_trace)
    df["block_size"] = 32
    bad = tmp_path / "bs32_inconsistent.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="does not match the KV cache block size"):
        build_simulator(base_args(bad, tmp_path / "out"))


def test_router_probe_leaves_no_state(conv_session_trace, tmp_path):
    sim = run_simulator(base_args(conv_session_trace, tmp_path / "out", num_replicas=2,
                                  global_scheduler_config_type="lop_uncached"))
    from vidur.entities import Request
    req = Request(arrived_at=0.0, num_prefill_tokens=64, num_decode_tokens=16,
                  block_hash_ids=[1, 2, 3, 4, 5], block_size=16)
    def pool_state(kv):  # the block pool is only read by the probe; guard that too
        pool = kv.block_pool
        free, b = [], pool.free_block_queue.free_list_head
        while b is not None:
            free.append(b.block_id)
            b = b.next_free_block
        return free, sorted(map(repr, pool.cached_block_hash_to_block)), [blk.ref_cnt for blk in pool.blocks]

    for rs in sim._scheduler._replica_schedulers.values():
        kv = rs._kv_cache_manager
        before = (dict(kv.req_to_block_hashes), repr(kv.prefix_cache_stats), pool_state(kv))
        rs.get_cached_prefill_length(req)
        assert (dict(kv.req_to_block_hashes), repr(kv.prefix_cache_stats), pool_state(kv)) == before
    # nothing left behind for requests that ran elsewhere
    for rs in sim._scheduler._replica_schedulers.values():
        assert not dict(rs._kv_cache_manager.req_to_block_hashes)


def test_write_output_records_predictor_provenance(tiny_trace, tmp_path):
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", write_metrics=True))
    sim.write_output()
    info = json.loads(next((tmp_path / "out").rglob("predictor_info.json")).read_text())
    assert len(info["bundle_key"]) == 16 and isinstance(info["clamp_hits"], dict)
    assert not list((tmp_path / "out").rglob("stuck_requests.json"))
    files = sorted(p for p in (tmp_path / "out").rglob("*") if p.is_file())
    snap = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    sim.write_output()  # idempotent: nothing rewritten
    assert sorted(p for p in (tmp_path / "out").rglob("*") if p.is_file()) == files
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files} == snap


def test_lint_flags_shared_private_ids(tmp_path):
    rows = [  # prompt 32 tokens = 2 full blocks; decode 16 -> 3 ids
        {"num_prefill_tokens": 32, "num_decode_tokens": 16, "block_hash_ids": "[1, 2, 3]", "block_size": 16},
        {"num_prefill_tokens": 32, "num_decode_tokens": 16, "block_hash_ids": "[1, 2, 3]", "block_size": 16},
    ]
    p = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert lint(str(p))["violations"] == {"R2": 2}
    rows[1]["block_hash_ids"] = "[1, 2, 4]"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert lint(str(p))["violations"] == {}


OBSERVING_POLICY = '''
SEEN = []
SIM = None  # set by the test: the simulator clock at each call is the time of the event being handled

def choose(request, replicas, ctx):
    return min(range(len(replicas)), key=lambda i: (replicas[i].outstanding_requests, i))

def observe(outcome, ctx):
    SEEN.append((outcome, SIM._time))
'''


def test_observe_hook_reports_first_token_then_end(tiny_trace, tmp_path):
    pol = tmp_path / "obs.py"
    pol.write_text(OBSERVING_POLICY)
    sim = build_simulator(base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                                    evolved_global_scheduler_config_policy_path=str(pol)))
    sim._scheduler._policy.SIM = sim
    sim.run()
    seen = sim._scheduler._policy.SEEN
    n = len(pd.read_csv(tiny_trace))
    by_req = {}
    for k, (o, t) in enumerate(seen):
        # identify a request by its arrival and prompt (unique in this slice)
        by_req.setdefault((o.arrived_at, o.num_prefill_tokens, o.session_id), []).append((k, o, t))
    assert len(by_req) == n
    for events in by_req.values():
        assert [o.event for _, o, _ in events] == ["first_token", "end"]  # once each, in order
        (_, ft, t_ft), (_, end, t_end) = events
        assert ft.first_token_at >= ft.arrived_at
        assert t_ft == ft.first_token_at  # fired at the prefill-end event, not later
        assert t_end == end.completed_at and end.first_token_at == ft.first_token_at
        assert ft.completed_at is None and ft.num_decode_tokens is None  # output length stays hidden
        assert end.completed_at >= end.first_token_at and end.num_decode_tokens > 0


def _metrics(sim):
    sim.write_output()
    out = Path(sim._config.metrics_config.output_dir)
    return pd.read_csv(next(out.rglob("request_metrics.csv"))).sort_values("Request Id").reset_index(drop=True)


def test_route_delay_shifts_every_request(tiny_trace, tmp_path):
    base = _metrics(run_simulator(base_args(tiny_trace, tmp_path / "a", write_metrics=True,
                                            global_scheduler_config_type="round_robin")))
    late = _metrics(run_simulator(base_args(tiny_trace, tmp_path / "b", write_metrics=True,
                                            global_scheduler_config_type="round_robin",
                                            cluster_config_route_delay_s="0.5")))
    col = next(c for c in base.columns if "e2e" in c.lower() and "normalized" not in c.lower())
    assert len(base) == len(late)
    assert (late[col] >= base[col] + 0.5 - 1e-6).mean() > 0.9  # light load: each request pays the hop


STALE_POLICY = '''
SEEN = []

def choose(request, replicas, ctx):
    SEEN.append((ctx.now, tuple((r.waiting, r.running, r.kv_usage) for r in replicas)))
    return min(range(len(replicas)), key=lambda i: (replicas[i].outstanding_requests, i))
'''


def test_view_refresh_freezes_polled_metrics(tiny_trace, tmp_path):
    pol = tmp_path / "stale.py"
    pol.write_text(STALE_POLICY)
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                                  evolved_global_scheduler_config_policy_path=str(pol),
                                  evolved_global_scheduler_config_view_refresh_s="1e9"))
    seen = sim._scheduler._policy.SEEN
    assert len({v for _, v in seen}) == 1  # polled once at the first decision, never refreshed
