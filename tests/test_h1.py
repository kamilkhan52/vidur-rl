"""h1 campaign (evolve/h1/PREREG.md): stages, hold-out and the training script agree with the pre-registration."""
import re
from pathlib import Path

import pandas as pd

from evolve import evaluate as ev

ROWS = {"Tool3": 23608, "Conv3": 12031}


def _inside(t, w, n):
    return any(s0 <= w and w + n <= s0 + n0 for s0, n0 in ev.H1_EXCLUDE[t])


def test_stage_sizes():
    assert len(ev.STAGES["h1_val"]) == 36 and len(ev.STAGES["h1_test"]) == 36 and len(ev.STAGES["h1_test_mod"]) == 18
    assert {sc.device for k in ("h1_val", "h1_test", "h1_test_mod") for sc in ev.STAGES[k]} == {"h100df"}
    assert {(sc.replicas, sc.scale) for sc in ev.STAGES["h1_test"]} == {(8, 0.5982), (8, 0.8973), (16, 0.2991),
                                                                        (16, 0.4487)}
    assert {sc.perturbation for sc in ev.STAGES["h1_test"]} == {1, 2, 3}
    assert {sc.perturbation for sc in ev.STAGES["h1_val"]} == {2, 3}


def test_val_and_test_windows_are_held_out_and_fit():
    for stage in ("h1_val", "h1_test"):
        for sc in ev.STAGES[stage]:
            assert sc.window_start + sc.window_len <= ROWS[sc.trace]
            assert _inside(sc.trace, sc.window_start, sc.window_len), sc.tag
    test = {(sc.trace, sc.window_start) for sc in ev.STAGES["h1_test"]}
    val = {(sc.trace, sc.window_start) for sc in ev.STAGES["h1_val"]}
    for t, w in val:
        assert all(w + ev.H1_LEN <= w2 or w >= w2 + ev.H1_LEN for t2, w2 in test if t2 == t)


def test_training_script_matches_exclusions():
    sh = (ev.REPO / "evolve/h1/es_seed.sh").read_text()
    ex = re.search(r"--exclude (\S+)", sh).group(1)
    got = {}
    for x in ex.split(","):
        t, s0, n0 = x.split(":")
        got.setdefault(t, []).append((int(s0), int(n0)))
    assert got == ev.H1_EXCLUDE
    for x in re.search(r"--eval-starts (\S+)", sh).group(1).split(","):
        t, ss = x.split(":")
        for s in map(int, ss.split("/")):
            assert not any(not (s + ev.H1_LEN <= s0 or s >= s0 + n0) for s0, n0 in ev.H1_EXCLUDE[t]), (t, s)


def test_trace_rows():
    for t, n in ROWS.items():
        assert len(pd.read_csv(ev.TRACES[t])) == n


def test_device_command():
    sc = ev.STAGES["h1_test"][0]
    cmd = ev.sim_command("evolved", "p.py", sc, Path("t.csv"), Path("o"))
    i = cmd.index("--replica_config_device")
    assert cmd[i + 1] == "h100"
    assert "compute_decodefix" in " ".join(cmd) and "h100_pairwise_nvlink" in cmd
