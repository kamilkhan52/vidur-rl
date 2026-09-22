"""Tail-metric definitions used by the analysis scripts (scripts/tail_metrics.py)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp09_tail_leverage as lev  # noqa: E402
from exp09_tail_extract import parse_run_path  # noqa: E402
from tail_metrics import is_missing, percentile, summarize_rows, tpot  # noqa: E402

from vidur.entities import Request


def test_percentile_matches_numpy_linear():
    rng = np.random.default_rng(0)
    v = list(rng.exponential(size=101))
    for q in (0.0, 0.5, 0.95, 0.99, 1.0):
        assert percentile(v, q) == pytest.approx(np.percentile(v, q * 100))
    assert math.isnan(percentile([], 0.99))


def test_tpot_excludes_first_token():
    # A request whose first token arrives at t=2 and whose remaining n-1 tokens
    # each take exactly 1 s has TPOT 1.0 under the simulator's accounting.
    n = 6
    r = Request(arrived_at=0.0, num_prefill_tokens=10, num_decode_tokens=n,
                block_hash_ids=None, block_size=None, session_id=0)
    r.on_batch_schedule(0.5)
    r.on_batch_end(2.0, 10)  # prefill done: first output token at t=2
    for i in range(n - 1):
        r.on_batch_end(3.0 + i, 1)
    assert r.completed
    ttft = r.prefill_completed_at - r.arrived_at
    assert tpot(r.e2e_time, ttft, n) == pytest.approx(1.0)
    # The old definition divided by n and would report 5/6.
    assert (r.e2e_time - ttft) / n == pytest.approx(5 / 6)


def test_tpot_undefined_for_single_token_outputs():
    assert tpot(5.0, 5.0, 1) is None
    assert tpot(5.0, 5.0, 0) is None


def test_summarize_rows():
    rows = [
        {"request_e2e_time": 10, "prefill_e2e_time": 2, "request_scheduling_delay": 1,
         "request_num_prefill_tokens": 100, "request_num_prefill_tokens_cached": 50,
         "request_num_decode_tokens": 5},
        {"request_e2e_time": 4, "prefill_e2e_time": 4, "request_scheduling_delay": 3,
         "request_num_prefill_tokens": 100, "request_num_prefill_tokens_cached": 0,
         "request_num_decode_tokens": 1},
    ]
    s = summarize_rows([{k: str(v) for k, v in r.items()} for r in rows])
    assert s["n_requests"] == 2 and s["n_tpot"] == 1
    assert s["p50_tpot"] == pytest.approx(2.0)  # (10 - 2) / (5 - 1)
    assert s["mean_e2e"] == pytest.approx(7.0)
    assert s["queue_fraction"] == pytest.approx(2.0 / 7.0)
    assert s["prefill_hit_ratio"] == pytest.approx(0.25)


def test_parse_run_path_any_model_size():
    p = "/scratch/x/runs/STICKY_LOR/70b-16R-Conv/scale3.0/20260422_231913/run1/2026-04-22_23-19-13-1"
    assert parse_run_path(p) == ("STICKY_LOR", "70b", "16R", "Conv", "3.0")


def test_is_missing_catches_nan():
    assert is_missing(float("nan")) and is_missing("nan") and is_missing("") and is_missing(None)
    assert not is_missing(0.0) and not is_missing("1.5")


def test_leverage_ignores_nan_and_flags_completion_gaps():
    items = [("A", {"p99_e2e": 10.0, "n_requests": 100.0}),
             ("B", {"p99_e2e": float("nan"), "n_requests": 100.0}),
             ("C", {"p99_e2e": 5.0, "n_requests": 80.0})]
    s, worst, best = lev.sli(items, "p99_e2e")
    assert (worst, best) == ("A", "C") and s == pytest.approx(0.5)
    assert lev.completion_mismatch(items) == pytest.approx(0.2)
