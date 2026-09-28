"""Seeded arrival jitter for bucketed traces (Mooncake conv/tool timestamps are 3 s buckets)."""

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTH_TRACE, _require_data
from vidur.config import TraceRequestGeneratorConfig
from vidur.request_generator.trace_request_generator import TraceRequestGenerator


@pytest.fixture(scope="module")
def bucketed_trace(tmp_path_factory):
    """40 small requests in 3-second buckets of 8, like the Mooncake conv/tool traces."""
    _require_data()
    df = pd.read_csv(SYNTH_TRACE)
    df = df[(df.num_prefill_tokens + df.num_decode_tokens) <= 20000].head(40).reset_index(drop=True)
    df["arrived_at"] = (np.arange(len(df)) // 8) * 3.0
    path = tmp_path_factory.mktemp("trace") / "bucketed.csv"
    df.to_csv(path, index=False)
    return path


def replay(path, **kw):
    cfg = TraceRequestGeneratorConfig(trace_file=str(path), time_scale_factor=2.0, **kw)
    gen = TraceRequestGenerator(cfg)
    out = []
    while (r := gen.get_next_request()) is not None:
        out.append((r.arrived_at, r.num_prefill_tokens, r.num_decode_tokens))
    return out


def test_zero_jitter_replays_exactly(bucketed_trace):
    a = replay(bucketed_trace)
    b = replay(bucketed_trace, arrival_jitter=0.0, seed=7)
    assert a == b
    assert sorted({t for t, _, _ in a}) == [0.0, 6.0, 12.0, 18.0, 24.0]


def test_jitter_stays_in_bucket_sorted_and_paired(bucketed_trace):
    a = replay(bucketed_trace, arrival_jitter=3.0, seed=101)
    b = replay(bucketed_trace, arrival_jitter=3.0, seed=101)
    c = replay(bucketed_trace, arrival_jitter=3.0, seed=201)
    base = replay(bucketed_trace)
    assert a == b                      # same seed -> identical workload (paired comparison)
    assert a != c                      # different seed -> different arrivals
    times = [t for t, _, _ in a]
    assert times == sorted(times)      # generator requires monotone arrivals
    assert len(set(times)) == len(times)
    # same requests, and each stays inside its scaled bucket [6k, 6k + 6)
    assert sorted((p, d) for _, p, d in a) == sorted((p, d) for _, p, d in base)
    bucket_of = {}
    for t0, p, d in base:
        bucket_of.setdefault((p, d), []).append(t0)
    for t, p, d in a:
        assert any(t0 <= t < t0 + 6.0 for t0 in bucket_of[(p, d)])
