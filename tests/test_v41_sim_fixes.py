"""v4.1 simulator fixes (review 2026-10-10 chunk A): the stuck-request end state (exit 3, stuck_requests.json),
route_delay_s as a pure hop delay that keeps same-instant bursts together, view_refresh_s as a periodic poller on a
fixed grid, ctx.now and call counting around observe(), bundle provenance, the shared side-effect-free probe."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import REPO_ROOT, base_args, build_simulator, reset_id_counters, run_simulator


def _metrics(sim):
    sim.write_output()
    out = Path(sim._config.metrics_config.output_dir)
    return pd.read_csv(next(out.rglob("request_metrics.csv"))).sort_values("Request Id").reset_index(drop=True)


# ------------------------------------------------------------------------------------------- stuck requests

def test_stuck_requests_exit_3_with_end_state_and_metrics(tiny_trace, tmp_path, monkeypatch):
    """A replica that can never schedule (wedged) leaves requests pending: vidur.main must still write the metrics,
    write a parseable stuck_requests.json listing them, and exit 3."""
    import vidur.main
    from vidur.simulator import Simulator

    orig_run = Simulator.run
    wedged = {}

    def run_with_wedged_replica(self):
        rid, rs = next(iter(self._scheduler._replica_schedulers.items()))
        rs.can_schedule = lambda: False  # requests routed here can never be scheduled
        wedged["rid"] = rid.id
        orig_run(self)

    monkeypatch.setattr(Simulator, "run", run_with_wedged_replica)
    out = tmp_path / "out"
    argv = base_args(tiny_trace, out, write_metrics=True, global_scheduler_config_type="round_robin")
    monkeypatch.setattr(sys, "argv", argv)
    reset_id_counters()
    with pytest.raises(SystemExit) as e:
        vidur.main.main()
    assert e.value.code == 3

    stuck = json.loads(next(out.rglob("stuck_requests.json")).read_text())
    n = len(pd.read_csv(tiny_trace))
    assert stuck and all(s == {"where": "waiting", "replica": wedged["rid"], "request_id": s["request_id"]}
                         for s in stuck)
    assert len({s["request_id"] for s in stuck}) == len(stuck) == n // 2  # round robin over 2 replicas
    metrics = pd.read_csv(next(out.rglob("request_metrics.csv")))
    assert len(metrics) == n - len(stuck)  # the completed half is still reported
    assert not set(metrics["Request Id"]) & {s["request_id"] for s in stuck}
    assert list(out.rglob("predictor_info.json"))
    assert not list(out.rglob("*.tmp*"))  # atomic writes leave no temp files


def test_failed_end_state_write_leaves_no_partial_file(tiny_trace, tmp_path):
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", write_metrics=True))
    sim.stuck_requests = [{"where": "waiting", "replica": object(), "request_id": 1}]  # not serialisable
    with pytest.raises(TypeError):
        sim.write_output()
    out = tmp_path / "out"
    assert not list(out.rglob("stuck_requests.json")) and not list(out.rglob("*.tmp*"))
    sim.stuck_requests = [{"where": "waiting", "replica": 0, "request_id": 1}]
    sim.write_output()  # the failed call did not mark the outputs as written
    assert json.loads(next(out.rglob("stuck_requests.json")).read_text()) == sim.stuck_requests


# ------------------------------------------------------------------------------------------- route delay

def _burst_trace(path: Path, times, per_burst=4, prompt=64, decode=16) -> Path:
    rows, nid = [], 0
    for t in times:
        for _ in range(per_burst):
            n_ids = (prompt + decode) // 16
            rows.append({"arrived_at": t, "num_prefill_tokens": prompt, "num_decode_tokens": decode,
                         "block_hash_ids": str(list(range(nid, nid + n_ids))), "block_size": 16, "session_id": nid})
            nid += n_ids
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("delay", [0.5, 0.125])
def test_route_delay_is_a_pure_hop_for_bursts(tmp_path, delay):
    """Same-instant bursts at light load: with delay d every request reaches its replica d after arrival, the burst
    still shares one batch per replica, and every latency is the zero-delay one plus exactly d."""
    trace = _burst_trace(tmp_path / "bursts.csv", times=[1.0, 3.0, 5.0])
    base = _metrics(run_simulator(base_args(trace, tmp_path / "a", num_replicas=2, write_metrics=True,
                                            global_scheduler_config_type="round_robin",
                                            trace_request_generator_config_time_scale_factor="1.0")))
    sim = run_simulator(base_args(trace, tmp_path / "b", num_replicas=2, write_metrics=True,
                                  global_scheduler_config_type="round_robin",
                                  trace_request_generator_config_time_scale_factor="1.0",
                                  cluster_config_route_delay_s=str(delay)))
    late = _metrics(sim)
    assert len(base) == len(late) == 12
    shifted = [c for c in base.columns if c.startswith(("request_e2e_time", "prefill_e2e_time",
                                                         "request_scheduling_delay"))
               and "normalized" not in c]
    assert {"request_e2e_time", "prefill_e2e_time", "request_scheduling_delay"} <= set(shifted)
    for c in shifted:
        assert (late[c] - base[c] - delay).abs().max() < 1e-9, c
    # same batches: execution times, cache hits, restarts and placement unchanged
    for c in ("request_execution_time", "request_model_execution_time", "prefill_time_execution_plus_preemption",
              "decode_time_execution_plus_preemption_normalized", "request_num_prefill_tokens_cached",
              "request_num_restarts", "replica"):
        assert (late[c] - base[c]).abs().max() < 1e-9, c
    # scheduled exactly d after arrival (light load: zero-delay scheduling delay is 0)
    assert base["request_scheduling_delay"].abs().max() < 1e-9
    assert (late["request_scheduling_delay"] - delay).abs().max() < 1e-9


def test_route_delay_keeps_burst_in_one_batch(tmp_path):
    """Review repro (4 requests at t=1, one replica): the delayed burst used to be split into a 1 + 3 batch."""
    trace = _burst_trace(tmp_path / "bursts.csv", times=[1.0], per_burst=4)
    sim = run_simulator(base_args(trace, tmp_path / "b", num_replicas=1, write_metrics=True,
                                  global_scheduler_config_type="round_robin",
                                  trace_request_generator_config_time_scale_factor="1.0",
                                  cluster_config_route_delay_s="0.5"))
    m = _metrics(sim)
    assert (m["request_scheduling_delay"] - 0.5).abs().max() < 1e-9  # all four scheduled at 1.5
    assert m["prefill_e2e_time"].nunique() == 1  # in one batch: their prefills complete together


# ------------------------------------------------------------------------------------------- view refresh

STALE_POLICY = '''
SEEN = []

def choose(request, replicas, ctx):
    SEEN.append((ctx.now, tuple((r.waiting, r.running, r.kv_usage) for r in replicas)))
    return min(range(len(replicas)), key=lambda i: (replicas[i].waiting + replicas[i].running, replicas[i].kv_usage, i))
'''


def _stale_sim(trace, out, pol, refresh, **kw):
    return build_simulator(base_args(trace, out, num_replicas=3, write_metrics=True,
                                     global_scheduler_config_type="evolved",
                                     evolved_global_scheduler_config_policy_path=str(pol),
                                     evolved_global_scheduler_config_view_refresh_s=str(refresh), **kw))


def test_view_refresh_views_are_snapshots_of_the_last_grid_tick(tiny_trace, tmp_path):
    """Each decision at t sees exactly the replica state at tick floor(t/R)*R, taken independently of decisions."""
    import math

    pol = tmp_path / "stale.py"
    pol.write_text(STALE_POLICY)
    R = 0.05
    sim = _stale_sim(tiny_trace, tmp_path / "out", pol, R)
    gs = sim._scheduler
    trace = []  # (event time, exact replica state before the event is handled)
    orig = sim._set_time

    def recording_set_time(t):
        trace.append((t, tuple(gs._poll())))
        orig(t)

    sim._set_time = recording_set_time
    sim.run()
    seen = gs._policy.SEEN
    assert len(seen) == len(pd.read_csv(tiny_trace))

    def state_at(tick_time):  # state after every event before tick_time = state before the first event >= it
        return next(s for t, s in trace if t >= tick_time)

    stale = 0
    for now, views in seen:
        k = math.floor(now / R)
        if k * R > now:
            k -= 1
        assert views == state_at(k * R), (now, k)
        stale += views != state_at(now)
    assert stale > 0  # the period actually hides fresher state at some decisions


KV_FIRST_POLICY = '''
SEEN = []

def choose(request, replicas, ctx):
    SEEN.append(tuple((r.waiting, r.running, r.kv_usage) for r in replicas))
    return min(range(len(replicas)), key=lambda i: (round(replicas[i].kv_usage, 2),
                                                    replicas[i].waiting + replicas[i].running, i))
'''


def test_view_refresh_below_burst_gap_changes_outcomes(tmp_path):
    """Review repro: Conv v4 bursts arrive every 3 s x scale (1.5 s here). A period of 1 s used to re-poll at every
    burst, so views were exact and outputs identical to refresh 0. On the fixed grid the views are 0.5 s old."""
    conv = REPO_ROOT / "data" / "processed_traces" / "mooncake_conversation_trace_v4.csv"
    if not conv.exists():
        pytest.skip(f"trace not available: {conv}")
    df = pd.read_csv(conv, nrows=120)
    df = df[(df.num_prefill_tokens + df.num_decode_tokens) <= 120_000]
    trace = tmp_path / "conv_v4.csv"
    df.to_csv(trace, index=False)
    pol = tmp_path / "kv_first.py"
    pol.write_text(KV_FIRST_POLICY)
    outs = {}
    for refresh in ("0", "1.0"):
        sim = build_simulator(base_args(trace, tmp_path / refresh, num_replicas=4, write_metrics=True,
                                        global_scheduler_config_type="evolved",
                                        evolved_global_scheduler_config_policy_path=str(pol),
                                        evolved_global_scheduler_config_view_refresh_s=refresh,
                                        trace_request_generator_config_time_scale_factor="0.5"))
        sim.run()
        outs[refresh] = (_metrics(sim), list(sim._scheduler._policy.SEEN))
        sim._scheduler._policy.SEEN.clear()
    assert outs["0"][1] != outs["1.0"][1]  # the views differ
    assert not outs["0"][0].equals(outs["1.0"][0])  # and so do the outcomes


def test_view_refresh_zero_and_negative_are_exact(tiny_trace, tmp_path):
    pol = tmp_path / "stale.py"
    pol.write_text(STALE_POLICY)
    sim = _stale_sim(tiny_trace, tmp_path / "out", pol, "-1")
    assert not sim._scheduler.needs_clock and not sim._clock_hook


# ------------------------------------------------------------------------------------------- observe

OBSERVING_POLICY = '''
SEEN = []
SIM = None

def choose(request, replicas, ctx):
    return min(range(len(replicas)), key=lambda i: (replicas[i].outstanding_requests, i))

def observe(outcome, ctx):
    SEEN.append((outcome, ctx.now, SIM._time))
'''


def test_observe_sets_ctx_now_to_the_event_time_and_is_not_a_policy_call(tiny_trace, tmp_path):
    pol = tmp_path / "obs.py"
    pol.write_text(OBSERVING_POLICY)
    sim = build_simulator(base_args(tiny_trace, tmp_path / "out", global_scheduler_config_type="evolved",
                                    evolved_global_scheduler_config_policy_path=str(pol)))
    gs = sim._scheduler
    gs._policy.SIM = sim
    routed_at = []
    orig_schedule = gs.schedule

    def schedule():
        mapping = orig_schedule()
        if mapping:
            routed_at.append(gs._ctx.now)
        return mapping

    gs.schedule = schedule
    sim.run()
    n = len(pd.read_csv(tiny_trace))
    seen = gs._policy.SEEN
    assert len(seen) == 2 * n
    for o, now, sim_time in seen:
        assert now == sim_time == (o.completed_at if o.event == "end" else o.first_token_at)
    assert gs.observe_calls == 2 * n and gs.policy_calls == n  # choose() once per request
    assert gs._ctx.now == routed_at[-1]  # restored after observe


# ------------------------------------------------------------------------------------------- provenance, probe

def test_provenance_records_bundle_source(tiny_trace, tmp_path):
    sim = run_simulator(base_args(tiny_trace, tmp_path / "out", write_metrics=True))
    sim.write_output()
    info = json.loads(next((tmp_path / "out").rglob("predictor_info.json")).read_text())
    src = info["bundle_source"]
    assert src["source"] == "bundle"  # the test cache ships the prediction bundle
    import hashlib
    bundle = Path(sim._scheduler._execution_time_predictor._config.cache_dir) / \
        f"prediction_bundle_{info['bundle_key']}.pkl"
    assert src["bundle_sha"] == hashlib.sha256(bundle.read_bytes()).hexdigest()[:16]


def test_disk_scheduler_uses_the_side_effect_free_probe(monkeypatch):
    from types import SimpleNamespace

    from vidur.scheduler.replica_scheduler import vllm_v1_disk_replica_scheduler as disk

    calls = []
    monkeypatch.setattr(disk, "probe_cached_prefill_tokens", lambda kv, req: calls.append((kv, req)) or 48)
    fake = SimpleNamespace(_kv_cache_manager="kv")
    assert disk.VLLMV1DiskReplicaScheduler.get_cached_prefill_length(fake, "req") == 48
    assert calls == [("kv", "req")]
