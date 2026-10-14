"""Step C0's learned router: the base router (finalist 4873) plus a learned correction (SELECTION amendment
2026-10-13, C3).

At the first request of each burst (the decision C0 logs and scores), the gain model predicts each other replica's
score gain over the base router's pick; the router overrides the pick only when the best predicted gain exceeds the
margin. Every other request is routed by the base router alone. An override is booked exactly as ``sim_fork``'s
forced decision books it for the base router (the attended-context FIFO entry moves from the base pick to the
override), so deployment and the training rollouts take the same action.

A generated policy file (``c0.write_policy``) holds the paths and calls ``build``::

    from evolve.rl_plan.c0_policy import build as _build
    choose = _build("<base router .py>", "<model .pkl>")
"""
import importlib.util
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evolve.rl_plan import c0_features as cf

HIST_KEY = "c0_hist"  # ctx.state key of the arrival history (the base router uses attn_queues / attn_sums)
STATS_KEY = "c0_stats"


def view_dicts(request, replicas):
    """The views as the plain dicts the logger stores (prompt block hashes dropped)."""
    req = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(request).items() if k != "prefix_hashes"}
    return req, [asdict(r) for r in replicas]


def load_base(path: str):
    spec = importlib.util.spec_from_file_location(f"c0_base_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "choose") or hasattr(mod, "route"):
        raise ValueError(f"{path}: the C0 router wraps a choose() base router")
    return mod


def move_fifo(state: dict, n: int, b: int, c: int) -> None:
    """Move the attended-context entry the base router appended at ``b`` (its last call) to ``c``."""
    q, s = state["attn_queues"], state["attn_sums"]
    if not q[b] or q[b][-1] != n:
        raise RuntimeError(f"the base router's last FIFO entry at replica {b} is not the request's {n} tokens")
    q[b].pop()
    s[b] -= n
    q[c].append(n)
    s[c] += n


def build(base_path: str, model_path: str):
    base = load_base(base_path)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    if tuple(bundle["features"]) != cf.FEATURES:
        raise ValueError(f"{model_path}: trained on other features")
    model, margin = bundle["model"], float(bundle["margin"])
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)  # one prediction on 7 rows per burst: OpenMP threads would only add overhead

    def choose(request, replicas, ctx):
        b = base.choose(request, replicas, ctx)
        if request.position_in_burst != 0:
            return b
        hist = ctx.state.setdefault(HIST_KEY, [])
        st = ctx.state.setdefault(STATS_KEY, {"decisions": 0, "overrides": 0})
        req, reps = view_dicts(request, replicas)
        rows = cf.features(req, reps, b, cf.hist_features(hist, ctx.now))
        hist.append((ctx.now, request.burst_size))
        cf.prune_hist(hist, ctx.now)
        st["decisions"] += 1
        alts = [c for c in range(len(replicas)) if c != b]
        pred = model.predict(np.asarray([rows[c] for c in alts]))
        j = int(np.argmax(pred))
        if pred[j] > margin:
            c = alts[j]
            move_fifo(ctx.state, request.num_prefill_tokens, b, c)
            st["overrides"] += 1
            return c
        return b

    return choose
