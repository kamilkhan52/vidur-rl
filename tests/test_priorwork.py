"""EXP33 prior-work routers: the router-side hold (dispatch() policies) and the ports' contracts.

A dispatch() policy may leave requests at the router; the harness re-offers them on every arrival and after every
batch that completed a request (Glia's Vidur fork). Routers without dispatch() are unaffected (the identity checks in
evolve/priorwork/PREREG.md cover that on the comparison scenarios; test_evolved_scheduler covers the ports).
"""

from pathlib import Path

import pandas as pd
import pytest

from conftest import base_args, run_simulator

PRIOR = Path(__file__).resolve().parents[1] / "evolve" / "policies" / "priorwork"

HOLD_ALL_UNTIL_THREE = '''
def dispatch(held, replicas, ctx):
    ctx.state["calls"] = ctx.state.get("calls", 0) + 1
    if len(held) < 3 and not ctx.state.get("open"):
        return []
    ctx.state["open"] = True  # after the first release, send everything on arrival
    return [(k, k % len(replicas)) for k in range(len(held))]
'''


def _metrics(sim):
    sim._write_output()
    out = Path(sim._config.metrics_config.output_dir)
    return pd.read_csv(next(out.rglob("request_metrics.csv"))).sort_values("Request Id").reset_index(drop=True)


def _run(trace, out, policy, num_replicas=3):
    return run_simulator(base_args(trace, out, num_replicas=num_replicas, write_metrics=True,
                                   global_scheduler_config_type="evolved",
                                   evolved_global_scheduler_config_policy_path=str(policy)))


def test_held_requests_are_released_and_wait_counts_in_latency(tiny_trace, tmp_path):
    pol = tmp_path / "hold.py"
    pol.write_text(HOLD_ALL_UNTIL_THREE)
    sim = _run(tiny_trace, tmp_path / "out", pol)
    df = _metrics(sim)
    assert len(df) == 24 and sim._scheduler.is_empty()
    first_two = df.sort_values("request_arrived_at").head(2)
    assert (first_two["request_scheduling_delay"] > 0).all(), "the first two requests waited at the router"
    assert sim._scheduler.holds_requests and sim._scheduler.policy_calls >= 3


@pytest.mark.parametrize("name", ["glia", "engram", "gorgo", "llumnix_dispatch", "dualmap"])
def test_prior_work_port_completes(tiny_trace, tmp_path, name):
    sim = _run(tiny_trace, tmp_path / name, PRIOR / f"{name}.py")
    df = _metrics(sim)
    assert len(df) == 24 and sim._scheduler.is_empty()
