"""The evolved-router harness must be a faithful stand-in for hand-written routers.

Built-in routers ported to policy files must produce byte-identical per-request
metrics on the same trace and seed. Also checks the contract guards and that
the policy never sees information a real router could not have.
"""

from pathlib import Path

import pandas as pd
import pytest

from conftest import base_args, build_simulator, run_simulator
from vidur.scheduler.global_scheduler.evolved_global_scheduler import (
    PolicyError,
    ReplicaView,
    RequestView,
)

POLICIES = Path(__file__).resolve().parents[1] / "evolve" / "policies"


def metrics_of(sim) -> pd.DataFrame:
    sim._write_output()
    out = Path(sim._config.metrics_config.output_dir)
    df = pd.read_csv(next(out.rglob("request_metrics.csv")))
    return df.sort_values("Request Id").reset_index(drop=True)


def run(trace, out, scheduler, num_replicas=3, **kw):
    return run_simulator(
        base_args(trace, out, num_replicas=num_replicas, write_metrics=True,
                  global_scheduler_config_type=scheduler, **kw)
    )


@pytest.mark.parametrize(
    "builtin,policy",
    [("round_robin", "round_robin"), ("lor", "lor"), ("sticky_lor", "sticky_lor"),
     ("lop_uncached", "lop_uncached")],
)
def test_ported_policy_reproduces_builtin(tiny_trace, tmp_path, builtin, policy):
    ref = metrics_of(run(tiny_trace, tmp_path / "ref", builtin))
    evo_sim = run(tiny_trace, tmp_path / "evo", "evolved",
                  evolved_global_scheduler_config_policy_path=str(POLICIES / f"{policy}.py"))
    evo = metrics_of(evo_sim)
    assert len(ref) == 24 and evo_sim._scheduler.is_empty()
    pd.testing.assert_frame_equal(ref, evo)


@pytest.mark.parametrize("name", ["sticky_lor", "lop_uncached"])
def test_ported_policy_reproduces_builtin_with_sessions_and_cache_hits(conv_session_trace, tmp_path, name):
    ref_sim = run(conv_session_trace, tmp_path / "ref", name)
    ref = metrics_of(ref_sim)
    evo = metrics_of(run(conv_session_trace, tmp_path / "evo", "evolved",
                         evolved_global_scheduler_config_policy_path=str(POLICIES / f"{name}.py")))
    assert len(ref) >= 30 and ref_sim._scheduler.is_empty()
    assert ref["request_num_prefill_tokens_cached"].sum() > 0, "slice must exercise prefix caching"
    pd.testing.assert_frame_equal(ref, evo)


def test_lop_uncached_port_matches_builtin_under_preemption(tmp_path):
    """Overloaded window with preemption restarts: the built-in used to double-count
    prefill completions after a restart and diverge from the port (P99 361 s vs 188 s)."""
    import evolve.evaluate as ev

    sc = ev.Scenario("Tool", 4, 1.5, 12000, 200, 1)
    ev.CACHE_DIR = tmp_path / "cache"
    trace = ev.window_csv(sc)
    kw = dict(trace_request_generator_config_time_scale_factor="1.5",
              trace_request_generator_config_arrival_jitter="3.0",
              trace_request_generator_config_seed="101", seed="101")
    ref = metrics_of(run(trace, tmp_path / "ref", "lop_uncached", num_replicas=4,
                         l_o_p_uncached_global_scheduler_config_seed="101", **kw))
    evo = metrics_of(run(trace, tmp_path / "evo", "evolved", num_replicas=4,
                         evolved_global_scheduler_config_seed="101",
                         evolved_global_scheduler_config_policy_path=str(POLICIES / "lop_uncached.py"), **kw))
    assert (ref["request_num_restarts"] > 0).sum() > 0, "scenario must exercise preemption restarts"
    pd.testing.assert_frame_equal(ref, evo)


def test_views_withhold_output_length():
    assert "num_decode_tokens" not in RequestView.__dataclass_fields__
    for f in RequestView.__dataclass_fields__:
        assert "decode" not in f and "output" not in f
    assert set(ReplicaView.__dataclass_fields__) == {
        "index", "outstanding_requests", "pending_prefill_tokens", "waiting", "running",
        "kv_usage", "kv_capacity_tokens"}


def write_policy(tmp_path, body):
    p = tmp_path / "policy.py"
    p.write_text(body)
    return str(p)


@pytest.mark.parametrize(
    "body,match",
    [("def choose(request, replicas, ctx):\n    return len(replicas)\n", "out of range"),
     ("def choose(request, replicas, ctx):\n    return 0.0\n", "must be an int"),
     ("def choose(request, replicas, ctx):\n    return True\n", "must be an int"),
     ("def route(burst, replicas, ctx):\n    return [0]\n", "must return"),
     ("def choose(request, replicas, ctx):\n    raise KeyError('x')\n", "policy raised KeyError"),
     ("x = 1\n", "neither route")],
)
def test_contract_violations_fail_loudly(tiny_trace, tmp_path, body, match):
    argv = base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                     evolved_global_scheduler_config_policy_path=write_policy(tmp_path, body))
    with pytest.raises(PolicyError, match=match):
        run_simulator(argv)


def test_route_interface_and_burst_views(tiny_trace, tmp_path):
    body = (
        "def route(burst, replicas, ctx):\n"
        "    ctx.state.setdefault('bursts', []).append(len(burst))\n"
        "    assert all(r.position_in_burst == k and r.burst_size == len(burst) for k, r in enumerate(burst))\n"
        "    assert all(len(r.cached_prefix_tokens) == len(replicas) for r in burst)\n"
        "    return [k % len(replicas) for k in range(len(burst))]\n"
    )
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                                  evolved_global_scheduler_config_policy_path=write_policy(tmp_path, body)))
    sched = sim._scheduler
    assert sim._scheduler.is_empty()
    assert sum(sched._ctx.state["bursts"]) == 24
    # all bookkeeping returns to zero once every request has completed
    assert sched._outstanding == [0] * len(sched._outstanding)
    assert sched._pending_prefill == [0] * len(sched._pending_prefill)
    assert sched.policy_calls == len(sched._ctx.state["bursts"])
