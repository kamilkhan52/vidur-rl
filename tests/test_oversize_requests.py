"""A trace request larger than a replica's KV cache must be refused, not wedge the replica."""

import pytest

from conftest import KV_CAPACITY_7B, base_args, build_simulator, run_simulator
from vidur.entities import Request


def test_trace_with_oversize_request_is_refused(tiny_trace_with_oversize, tmp_path):
    argv = base_args(
        tiny_trace_with_oversize, tmp_path / "out", global_scheduler_config_type="round_robin"
    )
    with pytest.raises(ValueError, match="trace_request_generator_config_max_tokens"):
        build_simulator(argv)


def test_max_tokens_filter_makes_the_same_trace_runnable(tiny_trace_with_oversize, tmp_path):
    argv = base_args(
        tiny_trace_with_oversize,
        tmp_path / "out",
        global_scheduler_config_type="round_robin",
        trace_request_generator_config_max_tokens=str(KV_CAPACITY_7B),
    )
    sim = run_simulator(argv)
    assert sim._scheduler.is_empty(), "all admitted requests should complete"
    assert sim._request_generator.num_requests == 24


def test_replica_scheduler_rejects_impossible_request(tiny_trace, tmp_path):
    sim = build_simulator(
        base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="round_robin")
    )
    rs = next(iter(sim._scheduler._replica_schedulers.values()))
    assert rs.max_request_total_tokens == KV_CAPACITY_7B
    too_big = Request(arrived_at=0.0, num_prefill_tokens=KV_CAPACITY_7B, num_decode_tokens=1,
                      block_hash_ids=None, block_size=None, session_id=0)
    with pytest.raises(ValueError, match="can hold at most"):
        rs.add_request(too_big)
