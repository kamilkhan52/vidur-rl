"""v4 harness: context cap, device tags, bundle consistency, code_version coverage, split disjointness."""
import pytest

from evolve import evaluate as ev


def test_context_cap_applies_to_llama31_only():
    if ev.V3_DATA:
        pytest.skip("EVOLVE_V3_DATA=1 disables the cap")
    assert ev.max_request_tokens(ev.Scenario("Synth", 8, 1.0, -1, 0, 1, model="l3_8b")) == 131072
    assert ev.max_request_tokens(ev.Scenario("Synth", 8, 1.0, -1, 0, 1, model="7b")) == ev.KV_CAPACITY["7b"]


def test_device_appears_in_tag_only_off_a100():
    a = ev.Scenario("Tool", 8, 1.0, -1, 0, 1, model="l3_8b")
    h = ev.Scenario("Tool", 8, 1.0, -1, 0, 1, model="l3_8b", device="h100")
    assert a.tag == "l3_8b-8R-Tool-s1.0-full-p1" and h.tag == "l3_8b-h100-8R-Tool-s1.0-full-p1"


def test_score_refuses_mixed_prediction_bundles():
    sc = ev.Scenario("Tool", 8, 1.0, -1, 0, 1, model="l3_8b")
    res = {f"p99_{m}": 1.0 for m in ("e2e", "ttft", "tpot", "tbt")} | {"mean_e2e": 1.0}
    base = {("lor", sc): res | {"bundle": "k:aaa"}}
    ev.score({sc: res | {"bundle": "k:aaa"}}, base, ("lor",), "tail4")
    with pytest.raises(ValueError, match="prediction bundles"):
        ev.score({sc: res | {"bundle": "k:bbb"}}, base, ("lor",), "tail4")


def _fresh_version():
    ev._VERSION.clear()
    v = ev.code_version()
    ev._VERSION.clear()
    return v


def test_code_version_covers_profiles(monkeypatch, tmp_path):
    """Behavioural: editing or adding a profiling CSV changes code_version (file_sha memo cleared)."""
    for f, text in (("vidur/a.py", "x = 1\n"), ("scripts/tail_metrics.py", "y = 2\n"), ("data/profiling/m/x.csv", "a,b\n1,2\n")):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text)
    monkeypatch.setattr(ev, "REPO", tmp_path)
    monkeypatch.setattr(ev, "_SHA", {})
    monkeypatch.setattr(ev, "_VERSION", {})
    v0 = _fresh_version()
    assert v0 != "unknown" and _fresh_version() == v0
    (tmp_path / "data/profiling/m/x.csv").write_text("a,b\n1,3\n")
    ev._SHA.clear()
    v1 = _fresh_version()
    assert v1 != v0
    (tmp_path / "data/profiling/h100").mkdir()
    (tmp_path / "data/profiling/h100/y.csv").write_text("a\n1\n")
    ev._SHA.clear()
    assert _fresh_version() not in (v0, v1)


@pytest.mark.parametrize("edit", [
    lambda m: m.setitem(ev.MAX_CONTEXT, "l3_8b", 32768),
    lambda m: m.setitem(ev.KV_CAPACITY, "l3_8b", 1),
    lambda m: m.setitem(ev.KV_CAPACITY_DEVICE, ("l3_8b", "h100"), 999),
    lambda m: m.setitem(ev.MODEL_NAME, "l3_8b", "meta-llama/Llama-3.1-8B"),
    lambda m: m.setitem(ev.TP, "l3_8b", 2),
    lambda m: m.setattr(ev, "BUCKETED", {"Tool"}),
])
def test_code_version_follows_table_values(monkeypatch, edit):
    """Editing a table a run reads changes code_version; restoring it restores the original value."""
    monkeypatch.setattr(ev, "_VERSION", {})
    v0 = _fresh_version()
    edit(monkeypatch)
    assert _fresh_version() != v0
    monkeypatch.undo()
    monkeypatch.setattr(ev, "_VERSION", {})
    assert _fresh_version() == v0


def test_v4_splits_are_disjoint():
    """No (trace, draw) pair is shared by two of the v4 splits that exist (search, held-out and test sets)."""
    names = [n for n in ("search3b", "heldout3", "test3", "test3b", "test3c", "test4") if n in ev.STAGES]
    assert {"search3b", "heldout3", "test3", "test3b"} <= set(names)
    draws = {n: {(sc.trace, sc.perturbation) for sc in ev.STAGES[n]} for n in names}
    for i, a in enumerate(names):
        assert len(set(ev.STAGES[a])) == len(ev.STAGES[a]), a
        for b in names[i + 1:]:
            assert not draws[a] & draws[b], (a, b, sorted(draws[a] & draws[b]))
    assert {sc.model for n in names for sc in ev.STAGES[n]} == {"l3_8b"}
