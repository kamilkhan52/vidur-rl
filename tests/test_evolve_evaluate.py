"""The cascaded evaluator must pair candidates and baselines on identical workloads."""

from pathlib import Path

import pytest

import evolve.evaluate as ev

POLICIES = Path(__file__).resolve().parents[1] / "evolve" / "policies"
SMALL = [ev.Scenario("Tool", 4, 1.5, 12000, 150, 1), ev.Scenario("Conv", 4, 2.0, 6000, 150, 0)]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    for t in ev.TRACES.values():
        if not t.exists():
            pytest.skip(f"trace not available: {t}")
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "cache")


def test_windows_are_rebased_and_deterministic():
    a = ev.window_csv(SMALL[0])
    b = ev.window_csv(SMALL[0])
    assert a == b
    import pandas as pd
    df = pd.read_csv(a)
    assert len(df) == 150 and df.arrived_at.min() == 0.0
    assert (df.num_prefill_tokens + df.num_decode_tokens).max() <= ev.KV_CAPACITY["7b"]


def test_policy_port_scores_exactly_like_builtin_baseline():
    res = ev.evaluate_policy(str(POLICIES / "lor.py"), SMALL, workers=4,
                             baselines=("lor", "round_robin"))
    assert res["valid"], res
    base = ev.ensure_baselines(SMALL, ("lor", "round_robin"))
    for d in res["details"]:
        # candidate == LOR on the same workload, so it can never beat the per-scenario best
        assert d["log_ratio"] >= -1e-12
    # and where LOR is the best baseline its ratio is exactly 0
    for sc in SMALL:
        for m in ev.METRICS:
            if base[("lor", sc)][f"p99_{m}"] <= base[("round_robin", sc)][f"p99_{m}"]:
                row = next(d for d in res["details"] if d["scenario"] == sc.tag and d["metric"] == m)
                assert row["log_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_broken_policy_is_invalid_not_crash(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def choose(request, replicas, ctx):\n    return -1\n")
    res = ev.evaluate_policy(str(p), SMALL[:1], workers=1, baselines=("lor",))
    assert res["valid"] is False and res["combined_score"] < -1e8
    assert "exit" in next(iter(res["failures"].values()))["reason"]


def test_search_and_heldout_sets_are_disjoint():
    assert not set(ev.STAGE1) & set(ev.HELDOUT)
    assert {sc.perturbation for sc in ev.STAGE1} == {1}
    assert all(sc.window_start < 0 for sc in ev.STAGE1 + ev.HELDOUT)  # full traces
    assert {sc.trace for sc in ev.HELDOUT} >= {"Synth"} and {sc.replicas for sc in ev.HELDOUT} >= {16}


def test_objectives_weight_metrics_and_balanced_is_equal_weighting():
    from evolve import evaluate as ev

    for w in ev.OBJECTIVES.values():
        assert abs(sum(w.values()) - 1.0) < 1e-12 and set(w) == set(ev.METRICS)
    sc = ev.Scenario("Tool", 8, 1.0, -1, 0, 1)
    base = {(b, sc): {"p99_e2e": 10.0, "p99_ttft": 4.0, "p99_tpot": 0.05, "mean_e2e": 3.0}
            for b in ev.DEFAULT_BASELINES}
    cand = {sc: {"p99_e2e": 9.0, "p99_ttft": 2.0, "p99_tpot": 0.06, "mean_e2e": 3.0}}
    import math

    logs = {"e2e": math.log(0.9), "ttft": math.log(0.5), "tpot": math.log(1.2)}
    for name, w in ev.OBJECTIVES.items():
        s = sum(w[m] * logs[m] for m in logs)
        assert abs(ev.score(cand, base, objective=name)["combined_score"] - (-(s + 0.5 * s))) < 1e-12
    # a TTFT-heavy win scores higher under "interactive" than under "completion"
    assert ev.score(cand, base, objective="interactive")["combined_score"] > \
        ev.score(cand, base, objective="completion")["combined_score"]
