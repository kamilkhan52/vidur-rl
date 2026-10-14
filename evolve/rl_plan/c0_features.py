"""Step C0 (SELECTION amendment 2026-10-13): the router's view as features, shared by the logger (``c0.py``) and the
deployed router (``c0_policy.py``), so training and deployment compute them with the same code.

Inputs are only what an evolved router is given at a routing call (``RequestView`` and the ``ReplicaView`` list, as
plain dicts; never output lengths or future arrivals), the base router's pick ``b``, and the router's own arrival
history (times and sizes of earlier bursts, which a router observes). One row per candidate replica ``c``; the gain
model predicts the score gain of routing to ``c`` instead of ``b``.
"""
import math
from typing import Dict, List, Sequence

HIST_WINDOWS = (10.0, 60.0)  # s: requests that arrived in the last 10 and 60 seconds
DT_CAP = 60.0  # s: time since the previous burst, capped (also the value before any burst)

FEATURES = (
    # the request
    "log_prompt", "burst_size", "has_session",
    # candidate c
    "outstanding", "log_pending", "waiting", "running", "kv_usage", "cached_frac", "inflight_frac", "active_frac",
    "log_uncached", "is_session_last",
    # c relative to the base pick b
    "d_outstanding", "d_log_pending", "d_waiting", "d_running", "d_kv_usage", "d_cached_frac", "d_log_uncached",
    # the cluster
    "mean_outstanding", "mean_running", "mean_kv_usage", "log_mean_pending", "max_cached_frac", "rank_outstanding",
    # arrivals the router has seen
    "log_n_10s", "log_n_60s", "dt_prev",
)


def hist_features(hist: Sequence, t: float) -> Dict[str, float]:
    """``hist``: (time, burst size) of earlier bursts, oldest first, all before ``t``."""
    n = {w: 0 for w in HIST_WINDOWS}
    for t0, k in reversed(hist):
        age = t - t0
        if age > max(HIST_WINDOWS):
            break
        for w in HIST_WINDOWS:
            if age <= w:
                n[w] += k
    dt = min(DT_CAP, t - hist[-1][0]) if hist else DT_CAP
    return {"log_n_10s": math.log1p(n[10.0]), "log_n_60s": math.log1p(n[60.0]), "dt_prev": dt}


def prune_hist(hist: list, t: float) -> None:
    """Drop bursts older than every window (in place; ``hist`` is a list or deque, oldest first)."""
    horizon = max(HIST_WINDOWS)
    while len(hist) > 1 and t - hist[0][0] > horizon:
        hist.pop(0) if isinstance(hist, list) else hist.popleft()


def _per_replica(req: dict, reps: List[dict]) -> List[Dict[str, float]]:
    n = max(int(req["num_prefill_tokens"]), 1)
    R = len(reps)
    cached = list(req.get("cached_prefix_tokens") or (0,) * R)
    inflight = list(req.get("inflight_prefix_tokens") or (0,) * R)
    active = list(req.get("active_prefix_tokens") or (0,) * R)
    last = int(req.get("session_last_replica", -1))
    out = []
    for i, r in enumerate(reps):
        unc = max(0, n - max(cached[i], inflight[i]))
        out.append({"outstanding": float(r["outstanding_requests"]),
                    "log_pending": math.log1p(max(0, r["pending_prefill_tokens"])),
                    "waiting": float(r["waiting"]), "running": float(r["running"]), "kv_usage": float(r["kv_usage"]),
                    "cached_frac": cached[i] / n, "inflight_frac": inflight[i] / n, "active_frac": active[i] / n,
                    "log_uncached": math.log1p(unc), "is_session_last": float(i == last)})
    return out


def features(req: dict, reps: List[dict], b: int, hist: Dict[str, float]) -> List[List[float]]:
    """Feature rows (in ``FEATURES`` order) for every replica index 0..R-1 (the row of ``b`` is included; the gain
    model is trained on the others)."""
    per = _per_replica(req, reps)
    R = len(per)
    n = max(int(req["num_prefill_tokens"]), 1)
    pend = [r["pending_prefill_tokens"] for r in reps]
    outs = [p["outstanding"] for p in per]
    cl = {"mean_outstanding": sum(outs) / R, "mean_running": sum(p["running"] for p in per) / R,
          "mean_kv_usage": sum(p["kv_usage"] for p in per) / R, "log_mean_pending": math.log1p(max(0, sum(pend) / R)),
          "max_cached_frac": max(p["cached_frac"] for p in per)}
    base = per[b]
    rq = {"log_prompt": math.log1p(n), "burst_size": float(req.get("burst_size", 1)),
          "has_session": float(req.get("session_last_replica", -1) >= 0)}
    rows = []
    for i, p in enumerate(per):
        f = dict(rq)
        f.update(p)
        for k in ("outstanding", "log_pending", "waiting", "running", "kv_usage", "cached_frac", "log_uncached"):
            f["d_" + k] = p[k] - base[k]
        f.update(cl)
        f["rank_outstanding"] = sum(o < p["outstanding"] for o in outs) / max(R - 1, 1)
        f.update(hist)
        rows.append([float(f[k]) for k in FEATURES])
    return rows
