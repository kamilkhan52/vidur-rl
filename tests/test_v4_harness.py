"""v4 harness: context cap, device tags, bundle consistency, profile hashing."""
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


def test_code_version_covers_profiles(monkeypatch, tmp_path):
    import inspect
    assert "data\" / \"profiling" in inspect.getsource(ev.code_version)
