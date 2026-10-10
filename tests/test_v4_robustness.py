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


def test_trace_block_size_must_match_cache(tiny_trace, tmp_path):
    df = pd.read_csv(tiny_trace)
    df["block_size"] = 32
    bad = tmp_path / "bs32.csv"
    df.to_csv(bad, index=False)
    with pytest.raises((ValueError, AssertionError)):
        build_simulator(base_args(bad, tmp_path / "out"))


def test_router_probe_leaves_no_state(conv_session_trace, tmp_path):
    sim = run_simulator(base_args(conv_session_trace, tmp_path / "out", num_replicas=2,
                                  global_scheduler_config_type="lop_uncached"))
    from vidur.entities import Request
    req = Request(arrived_at=0.0, num_prefill_tokens=64, num_decode_tokens=16,
                  block_hash_ids=[1, 2, 3, 4, 5], block_size=16)
    for rs in sim._scheduler._replica_schedulers.values():
        kv = rs._kv_cache_manager
        before = (dict(kv.req_to_block_hashes), repr(kv.prefix_cache_stats))
        rs.get_cached_prefill_length(req)
        assert (dict(kv.req_to_block_hashes), repr(kv.prefix_cache_stats)) == before
    # nothing left behind for requests that ran elsewhere
    for rs in sim._scheduler._replica_schedulers.values():
        assert not dict(rs._kv_cache_manager.req_to_block_hashes)


def test_write_output_records_predictor_provenance(tiny_trace, tmp_path):
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", write_metrics=True))
    sim.write_output()
    info = json.loads(next((tmp_path / "out").rglob("predictor_info.json")).read_text())
    assert len(info["bundle_key"]) == 16 and isinstance(info["clamp_hits"], dict)
    assert not list((tmp_path / "out").rglob("stuck_requests.json"))
    sim.write_output()  # idempotent


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

def choose(request, replicas, ctx):
    return min(range(len(replicas)), key=lambda i: (replicas[i].outstanding_requests, i))

def observe(outcome, ctx):
    SEEN.append(outcome)
'''


def test_observe_hook_reports_first_token_then_end(tiny_trace, tmp_path):
    pol = tmp_path / "obs.py"
    pol.write_text(OBSERVING_POLICY)
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                                  evolved_global_scheduler_config_policy_path=str(pol)))
    seen = sim._scheduler._policy.SEEN
    n = len(pd.read_csv(tiny_trace))
    assert [o.event for o in seen].count("first_token") == n and [o.event for o in seen].count("end") == n
    for o in seen:
        assert o.first_token_at >= o.arrived_at
        if o.event == "first_token":
            assert o.completed_at is None and o.num_decode_tokens is None  # output length stays hidden
        else:
            assert o.completed_at >= o.first_token_at and o.num_decode_tokens > 0
