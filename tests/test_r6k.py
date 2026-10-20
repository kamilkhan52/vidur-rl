"""r6k (evolve/r6k/PREREG.md): the hardware windows are reproduced exactly, and no search or validation window touches
them; search and validation windows are disjoint from each other and from the excluded ranges they must avoid."""
import pandas as pd
import pytest

from evolve import evaluate as ev
from evolve.rl_plan.rl_residual import N_REQ

HW = {"ToolHW": "data/r6k/Tool_4000_2000_3281d502d2bbca0f.csv", "ConvHW": "data/r6k/Conv_0_2000_2fe221fad356221c.csv"}


def _win(trace, start, n):
    return pd.read_csv(ev.window_csv(ev.Scenario(trace, 4, 2.0, start, n, 1, model="l3_8b", device="rtxpro6000")))


def _same(a, b):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return a.shape == b.shape and all((a[c].astype(str) == b[c].astype(str)).all() for c in a if c != "arrived_at") \
        and ((a.arrived_at - b.arrived_at).abs() < 1e-9).all()


@pytest.mark.parametrize("trace", sorted(HW))
def test_hw_window_is_realsys_first_1000(trace):
    assert _same(_win(trace, 0, 1000), pd.read_csv(ev.REPO / HW[trace]).head(1000))


def test_excluded_ranges_cover_the_hardware_requests():
    assert _same(_win("Tool3", 4000, 2000), pd.read_csv(ev.REPO / HW["ToolHW"]))
    c = _win("Conv3", 0, 2004)
    assert _same(c[(c.num_prefill_tokens + c.num_decode_tokens) <= 122752], pd.read_csv(ev.REPO / HW["ConvHW"]))
    assert (4000, 2000) in ev.R6K_EXCLUDE["Tool3"] and (0, 2004) in ev.R6K_EXCLUDE["Conv3"]


def test_search_and_validation_windows_avoid_what_they_must():
    def overlaps(s, n, s0, n0):
        return not (s + n <= s0 or s >= s0 + n0)
    for t in ("Tool3", "Conv3"):
        for w in ev.R6K_SEARCH_WINDOWS[t]:  # search: clear of the hardware and validation ranges
            assert not any(overlaps(w, 1000, s0, n0) for s0, n0 in ev.R6K_EXCLUDE[t])
        hw = ev.R6K_EXCLUDE[t][0]
        for w in ev.R6K_VAL_WINDOWS[t]:  # validation: clear of the hardware range, inside its own range
            assert not overlaps(w, 1000, *hw)
            assert any(s0 <= w and w + 1000 <= s0 + n0 for s0, n0 in ev.R6K_EXCLUDE[t][1:])
            assert w + 1000 <= N_REQ[t]


def test_final_test_uses_only_hardware_traces():
    assert {sc.trace for sc in ev.STAGES["r6k_test"]} == {"ToolHW", "ConvHW"}
    assert all(sc.replicas == 4 and sc.device == "rtxpro6000" and sc.window_len == 1000 for sc in ev.STAGES["r6k_test"])
    # amendment 3: draws 1-6 at the candidate high loads, the first 1000 requests of each hardware window
    assert len(ev.STAGES["r6k_test"]) == 2 * 3 * 6
    assert {sc.perturbation for sc in ev.STAGES["r6k_test"]} == set(range(1, 7))
    assert {sc.scale for sc in ev.STAGES["r6k_test"]} == {3.0, 4.0, 4.8}
    assert all(sc.window_start == 0 for sc in ev.STAGES["r6k_test"])
