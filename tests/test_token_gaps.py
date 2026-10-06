"""Exact time-between-tokens log: per request, the gaps sum to E2E - TTFT and there is one gap per output token
after the first, preempted requests included (2026-10-06)."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evolve import evaluate as ev


def test_token_gaps_reconcile_with_request_metrics():
    sc = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1)
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "out"
        subprocess.run(ev.sim_command("lor", None, sc, ev.window_csv(sc), out), cwd=ev.REPO, check=True,
                       capture_output=True)
        df = pd.read_csv(next(out.rglob("request_metrics.csv")))
        z = np.load(next(out.rglob("token_gaps.npz")))
    g = pd.DataFrame({"r": z["request_id"], "g": z["gap"]}).groupby("r")["g"].agg(["sum", "count"])
    m = df.set_index("Request Id").join(g).fillna(0)
    assert (m["request_num_restarts"] > 0).any()  # the window exercises preemption
    err = (m["sum"] - (m["request_e2e_time"] - m["prefill_e2e_time"])).abs().max()
    assert err < 1e-5
    assert ((m["count"] - (m["request_num_decode_tokens"] - 1)) == 0).all()
    r = ev.run_one("lor", sc)
    # run_one scores only requests after the warm-up cut of a windowed scenario
    assert r["ok"] and r["p99_tbt"] >= r["p50_tbt"] > 0 and 0 < r["n_tbt"] <= int(z["gap"].size)
