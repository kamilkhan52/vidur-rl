"""Per-run latency and tail metrics from a Vidur ``request_metrics.csv``.

Single source of truth for the tail analysis (EXP08/EXP09 re-cuts, RL evals,
and the upcoming frontier re-measurement). Definitions:

- E2E  = ``request_e2e_time`` (arrival -> last token).
- TTFT = ``prefill_e2e_time`` (arrival -> first token). Vidur emits the first
  output token when prefill completes, and records this time only once, so it
  is unaffected by preemption restarts.
- TPOT = (E2E - TTFT) / (n_decode - 1), over requests with n_decode > 1.
  The time after the first token covers n_decode - 1 tokens. The pre-2026-09-22
  extractors divided by n_decode and kept single-token requests (TPOT = 0),
  which biased TPOT low, strongly so for short outputs (24 % of Tool requests
  have <= 5 output tokens). This matches the vLLM benchmark definition.
- Percentiles use linear interpolation (numpy's default).

``summarize_run`` also reports how many requests completed. Tail statistics
over completed requests only are biased whenever completion counts differ
between the runs being compared; callers should check ``n_requests``.
"""

import csv
import math
from typing import Dict, Iterable, List, Optional, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 1]; NaN for an empty input."""
    if not values:
        return float("nan")
    v = sorted(values)
    k = (len(v) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return v[int(k)]
    return v[f] * (c - k) + v[c] * (k - f)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def tpot(e2e: float, ttft: float, num_decode_tokens: float) -> Optional[float]:
    """Time per output token after the first; None when undefined (n_decode <= 1)."""
    n = int(num_decode_tokens)
    if n <= 1:
        return None
    return (e2e - ttft) / (n - 1)


def read_request_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


SUMMARY_FIELDS = [
    "n_requests",
    "mean_e2e", "p50_e2e", "p95_e2e", "p99_e2e",
    "mean_ttft", "p50_ttft", "p95_ttft", "p99_ttft",
    "mean_tpot", "p50_tpot", "p95_tpot", "p99_tpot",
    "queue_delay_mean", "queue_delay_p99", "queue_fraction",
    "prefill_hit_ratio", "n_tpot",
]


def summarize_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    rows = list(rows)
    e2e = [float(r["request_e2e_time"]) for r in rows]
    ttft = [float(r["prefill_e2e_time"]) for r in rows]
    sd = [float(r["request_scheduling_delay"]) for r in rows]
    pf = [float(r["request_num_prefill_tokens"]) for r in rows]
    ca = [float(r["request_num_prefill_tokens_cached"]) for r in rows]
    tp = [
        t
        for t in (
            tpot(float(r["request_e2e_time"]), float(r["prefill_e2e_time"]),
                 float(r["request_num_decode_tokens"]))
            for r in rows
        )
        if t is not None
    ]
    me, qd = mean(e2e), mean(sd)
    out = {"n_requests": len(rows)}
    for name, vals in (("e2e", e2e), ("ttft", ttft), ("tpot", tp)):
        out[f"mean_{name}"] = mean(vals)
        for q in (50, 95, 99):
            out[f"p{q}_{name}"] = percentile(vals, q / 100)
    out["queue_delay_mean"] = qd
    out["queue_delay_p99"] = percentile(sd, 0.99)
    out["queue_fraction"] = qd / me if me and not math.isnan(me) else float("nan")
    out["prefill_hit_ratio"] = sum(ca) / sum(pf) if sum(pf) else float("nan")
    out["n_tpot"] = len(tp)
    return out


def summarize_run(request_metrics_csv: str) -> Dict[str, float]:
    return summarize_rows(read_request_rows(request_metrics_csv))


def is_missing(x) -> bool:
    """True for None, '' or NaN (float or numeric string)."""
    if x is None or x == "":
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True
