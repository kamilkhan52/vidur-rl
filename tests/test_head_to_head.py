import math

from evolve import evaluate as ev
from evolve.analysis.head_to_head import head_to_head


def test_ratio_to_each_baseline_recovers_policy_over_baseline():
    scs = [ev.Scenario("Tool", 8, 1.0, -1, 0, p, model="l3_8b") for p in (1, 2)]
    bar = ("lor", "sticky_lor")
    p99 = {"lor": 10.0, "sticky_lor": 5.0}  # bar = sticky_lor on every metric
    cache = {ev.baseline_key(b, sc): {f"p99_{m}": p99[b] for m in ("e2e", "ttft", "tpot", "tbt")}
             for b in bar for sc in scs}
    per = {sc.tag: {m: 0.8 for m in ("e2e", "ttft", "tpot", "tbt")} for sc in scs}  # policy P99 = 4.0
    out = head_to_head(per, scs, cache, bar)
    assert math.isclose(out["sticky_lor"]["e2e"]["geo"], 0.8)
    assert math.isclose(out["lor"]["tbt"]["geo"], 0.4)
    assert out["lor"]["e2e"]["wins"] == 2


def _bar_eval_json(tmp_path, monkeypatch, **over):
    """A bar_eval-style JSON scored against a 3-member bar whose best member (round_robin) is not in BAR_V2's
    head, plus the matching mock cache. The policy's true P99 is 4.0 everywhere."""
    import json
    scs = ev.STAGES["search3b"]
    bar = ["lor", "sticky_lor", "round_robin"]
    p99 = {"lor": 10.0, "sticky_lor": 5.0, "round_robin": 2.5}
    cache = {ev.baseline_key(b, sc): {f"p99_{m}": p99[b] for m in ("e2e", "ttft", "tpot", "tbt")}
             for b in bar for sc in scs}
    monkeypatch.setattr(ev, "load_baseline_cache", lambda: cache)
    per = {sc.tag: {m: 4.0 / 2.5 for m in ("e2e", "ttft", "tpot", "tbt")} for sc in scs}
    d = {"bar": bar, "bar_name": "x", "code_version": ev.code_version(), "stage": "search3b",
         "provenance": ev.provenance(), "policies": {"pol.py": {"per_scenario": per}}}
    d.update(over)
    d = {k: v for k, v in d.items() if v is not None}
    f = tmp_path / "be.json"
    f.write_text(json.dumps(d))
    return ["--json", str(f), "--stage", "search3b", "--policy", "pol.py"]


def test_main_uses_the_bar_the_json_was_scored_against(tmp_path, monkeypatch):
    from evolve.analysis import head_to_head as hh
    out = hh.main(_bar_eval_json(tmp_path, monkeypatch))["pol.py"]
    assert sorted(out) == ["lor", "round_robin", "sticky_lor"]
    assert math.isclose(out["sticky_lor"]["e2e"]["geo"], 0.8)  # was reported as 1.6 against BAR_V2's head
    assert math.isclose(out["round_robin"]["tbt"]["geo"], 1.6) and out["lor"]["ttft"]["wins"] == 10


def test_main_refuses_a_json_without_a_bar_unless_given(tmp_path, monkeypatch):
    import pytest
    from evolve.analysis import head_to_head as hh
    argv = _bar_eval_json(tmp_path, monkeypatch, bar=None)
    with pytest.raises(SystemExit, match="--bar"):
        hh.main(argv)
    monkeypatch.setitem(ev.BARS, "mock3", ("lor", "sticky_lor", "round_robin"))
    out = hh.main(argv + ["--bar", "mock3"])["pol.py"]
    assert math.isclose(out["sticky_lor"]["e2e"]["geo"], 0.8)


def test_main_refuses_another_code_version_unless_allowed(tmp_path, monkeypatch, capsys):
    import pytest
    from evolve.analysis import head_to_head as hh
    argv = _bar_eval_json(tmp_path, monkeypatch, code_version="c6fceae7bfa8")
    with pytest.raises(SystemExit, match="code_version c6fceae7bfa8"):
        hh.main(argv)
    out = hh.main(argv + ["--allow-code-version-mismatch"])["pol.py"]
    assert math.isclose(out["sticky_lor"]["e2e"]["geo"], 0.8)
    assert "WARNING" in capsys.readouterr().err


def test_main_refuses_other_traces(tmp_path, monkeypatch):
    import pytest
    from evolve.analysis import head_to_head as hh
    prov = dict(ev.provenance(), traces={"Tool": "0" * 16, "Conv": "1" * 16, "Synth": "2" * 16})
    with pytest.raises(SystemExit, match="trace shas"):
        hh.main(_bar_eval_json(tmp_path, monkeypatch, provenance=prov))
