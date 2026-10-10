"""Closed-loop headroom pilot (RL design discussion, 2026-10-10; reviews G r1, r2).

Question: with the true future (hindsight), does picking at every routing decision the candidate replica whose
rollout scores best, with pi_e (the evolved rule l3_48a067a7df63) as the rollout policy, improve the window's P99
tails over pi_e alone? The method must be able to answer either way, so the rollout objective has to see what forms
window tails (sustained load after the decision, and the tails of the routed request and of later arrivals).

Loop (``--mode closed``, default). At each decision after the warm-up, every candidate replica is tried in an
``os.fork`` child: the child forces the arriving request onto that replica (``sim_fork.override_next_decision``),
routes everything after it with pi_e and scores the affected set. The parent takes the best-scoring candidate (ties
and gains <= ``min_gain`` keep pi_e's choice), executes it through the same forcing mechanism and continues. Only the
first request of a burst is planned (with jittered arrivals, p >= 1, bursts are single requests; ``route()``
policies with bursts are refused). Early decisions' affected sets include requests that arrived before the warm-up
cut, which the window metrics do not score.

Horizons (``--horizon``; the affected set is identical for all candidates of a decision):
- ``T=<t>+drain`` (default ``T=1x+drain``; t in seconds, or ``<f>x`` = f times pi_e's window P99 E2E): arrivals
  continue for t seconds after the decision, then stop, and every affected request (in flight at the decision,
  including the routed one, plus every arrival up to the cutoff) runs to completion. Nothing is censored, so the
  routed request's and late arrivals' tails are scored.
- ``T=<t>`` (pure cutoff; an option only): the rollout stops at the cutoff and an unfinished request is censored
  (``--censor elapsed``: its age at the cutoff, a lower bound; ``remaining``: plus remaining decode tokens times
  pi_e's mean TPOT). BLIND SPOT (review G r2): every request arriving at or after the decision has E2E <= t, so at
  ``T=1x`` its E2E hinge is identically 0 and the tail term sees only requests already in flight; censoring also
  credits candidates that push work past the cutoff. Probe on Conv 4R s1.5 w6000+200 (K=120): a candidate scored
  210 better than pi_e censored and 229 worse uncensored. ``test_headroom.test_censoring_probe`` checks the blind
  spot and that the drained scores bound the censored ones.
- ``H=<n>``: the next n routing decisions, then arrivals stop and the system drains (the r1 method; on Tool H=64 is
  ~8 s of arrivals against a 95 s P99 E2E).
- ``full``: the rest of the window (reference horizon of ``--mode openloop``).

Horizon selection (``--mode openloop``): pi_e runs alone; at every ``--every``-th decision after the cut, all
candidates are rolled out under each ``--horizon`` and under ``--reference`` (default ``full``), and the summary
reports per horizon how well its candidate ranking agrees with the reference: mean Spearman correlation, top-1
agreement, agreement on whether to deviate from pi_e, pairwise sign agreement against pi_e's choice, and the
reference-measured gain of the horizon's choice over pi_e's (``ref_gain``; the reference's own is the upper bound).
One child per candidate serves every horizon: it runs once to the longest horizon and, at each shorter horizon's
stop point, forks a grandchild that drains and scores it, so a decision costs about one ``full`` rollout plus one
drain per horizon per candidate. Run this on the window used for the closed loop, then run the closed loop at the
horizon that ranks like the reference.

Cost (CPU-s per planned decision = R rollouts; measured 2026-10-10 on the Mac at load ~6; every run records
``cpu_s_children`` and ``cpu_s_per_decision``):
- Tool 8R s0.75 w0+2000 (a full pi_e run of the window: ~25 s). Closed loop: ``H=64`` 16.4 (r1 run; 2.05 per
  rollout); ``T=1x`` (94.8 s) 30.4 (3.8 per rollout). Open-loop smoke at the first decision after the cut (the
  costliest point: the longest remainder), standalone estimates per decision: ``H=64`` 9.6, ``T=0.5x+drain`` 25.7,
  ``T=1x+drain`` 44.7 (5.6 per rollout), ``T=2x+drain`` 77, ``T=1x`` 32.6, ``full`` 81; the nested open-loop point
  cost 121 CPU-s for all six horizons (100 s wall with 2 workers). Later decisions are cheaper (horizons hit the
  window end), so a closed loop at ``T=1x+drain`` is <= 20 CPU-h (~5 h wall at 4 workers, ~2.5 h at 8) and an
  open-loop selection at ``--every 10`` (165 points) <= 5.5 CPU-h, ~3 CPU-h expected.
- ``--plan-every K`` divides the closed-loop cost by K but ANSWERS A DIFFERENT QUESTION: the headroom of lookahead at
  1/K of the decisions with pi_e elsewhere. A null at K > 1 does not bound the K = 1 headroom (gains need not add up)
  and a positive is not a per-decision estimate. ``--max-decisions M`` plans only the first M (biased to the start).

Scores (``--score``), from per-request E2E e_i and TTFT f_i over the affected set; tau_e, tau_f = pi_e's window P99
E2E and P99 TTFT, m = pi_e's window mean E2E:
- ``tail`` (default): ``sum_i e_i + w * (sum_i (e_i/tau_e - 1)+ + sum_i (f_i/tau_f - 1)+)``, w = 25 m unless
  ``--tail-weight``. The hinges are the tail terms: by Rockafellar-Uryasev, ``tau + sum_j (x_j - tau)+ / (0.01 N)``
  with tau = the window P99 bounds the window CVaR_0.99 and is tight at tau, so a hinge change d moves log CVaR_0.99
  by about d / (0.01 N tau). The weight is a DESIGN CHOICE, a mean regulariser, not the objective's marginal rate:
  evaluate.score on one scenario puts 0.375 (0.25 x (1 + 0.5)) on each log P99, and its mean-E2E term
  log(mean / best baseline mean) is one-sided, active only above 1.05 x the best baseline's mean. When it is inactive
  (likely at pi_e's operating point) the objective's weight on the summed E2E is 0; when active, the same
  linearisation gives w = 37.5 m. w = 25 m keeps the score tail-dominated (one normalised hinge unit = 25 mean
  E2Es) while the sum term orders candidates whose hinges tie and guards the mean. TTFT gets its own hinge because
  placements trade TTFT against E2E (the r1 lex run moved P99 TTFT by +73 %). TPOT and TBT are not scored (per-token
  pace; P99 TBT is a tail of token gaps that a per-request sum dilutes, and LOR's P99 TBT is +9.8 % over pi_e's on
  Tool, so placement does move it): every lookahead row reports ``vs_pi_e``, the log ratios of the four P99s and the
  mean E2E against pi_e plus their tail4 combination, so a gain bought with token pace shows in the row.
  Logged per decision: each candidate's score and terms (sum, hinge, hinge_ttft, the routed request's E2E, the
  post-decision E2E hinge, unfinished counts), the per-term gain of the chosen candidate over pi_e's choice and
  ``decided_by`` = the term with the largest gain (``keep`` when pi_e's choice is kept).
- ``lex``: (sum (e_i/tau_e - 1)+, sum e_i) lexicographically, ties within 1e-9 (the r1 rule; closed loop only).
- ``sum``, ``hinge``: one term each.

Resampled output lengths (``--future sampled-lengths``; a PARTIAL information gap): in each decision's children
(common random numbers: one seed per decision), only OUTPUT LENGTHS are resampled: not-yet-arrived requests from the
empirical output-length distribution, in-flight requests (including the one being routed) from the same
distribution conditioned on exceeding the tokens they have already generated, capped at ``max_request_tokens``.
Arrival times, prompt lengths, prefix hashes and sessions of future arrivals stay TRUE, so this is not the full
"no hindsight" gap of the plan (resampled arrivals and lengths). The distribution is the capacity-filtered trace
outside the window (a full-trace scenario falls back to the whole trace). The parent continues on the true trace.

Reuse and resume. ``<out>/summary.json`` carries ``meta`` (scenario, policy and Mooncake sha, code_version,
prediction bundle); a file with a different meta or format is refused (use a new --out). pi_e and baseline rows
depend only on that meta (the in-process pi_e is checked against ``evaluate.run_one`` unless ``--skip-baselines``).
Each lookahead / open-loop row is keyed by a hash of all its parameters INCLUDING the sha256 of headroom.py and
sim_fork.py as loaded, so rows from other code are never reused. Rows that fail the completion gate are not stored.
Each run appends one JSON line per decision to ``<key>.jsonl`` after a header line with the parameters; an
interrupted run resumes from it (the header must match; the logged decisions are replayed with ``apply_decision``,
checking pi_e's choice at each, then planning continues).

Usage:
  .venv/bin/python -m evolve.rl_plan.headroom --mode openloop --trace Tool --every 10 \\
      --horizon H=64,T=0.5x+drain,T=1x+drain,T=2x+drain,T=1x --out evolve/rl_plan/out/tool_s075_w2000_v3
  .venv/bin/python -m evolve.rl_plan.headroom --trace Tool --horizon T=1x+drain \\
      --out evolve/rl_plan/out/tool_s075_w2000_v3
"""

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import select
import signal
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from evolve import evaluate as ev
from evolve import sim_fork
from evolve.sim_fork import after_os_fork, build_simulator, override_next_decision
from tail_metrics import summarize_rows  # evolve.evaluate puts scripts/ on sys.path

POLICY = str(ev.REPO / "evolve/finalists/l3_48a067a7df63.py")
MOONCAKE = str(ev.REPO / "evolve/policies/systems/mooncake_min_ttft.py")
FORMAT = 2
# sha256 of the planner and fork code as loaded (part of every lookahead row's key)
CODE_SHA = {"headroom": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "sim_fork": hashlib.sha256(Path(sim_fork.__file__).read_bytes()).hexdigest()[:16]}


def argv_for(sc, out_dir, policy=POLICY):
    return ev.sim_command("evolved", policy, sc, ev.window_csv(sc), Path(out_dir))[3:]


def score_output(out_dir, sc):
    """``evaluate.run_one``'s post-processing on an output directory written in process: completion gate, warm-up
    cut, ``summarize_rows``, exact TBT over the scored requests' tokens, prediction bundle and clamp hits. Returns
    ``{"ok": False, "reason": ...}`` when the gate fails, as ``run_one`` does. (``run_one`` inlines this code and is
    hashed into code_version, so it cannot be factored out; ``test_headroom`` checks the two agree.)"""
    out_dir = Path(out_dir)
    f = next(out_dir.rglob("request_metrics.csv"))
    df = pd.read_csv(f)
    n_expected = len(pd.read_csv(ev.window_csv(sc), usecols=["arrived_at"]))
    completed = len(df)
    if completed != n_expected:
        return {"ok": False, "reason": f"completion gate: {completed}/{n_expected} requests completed",
                "completed": completed, "n_expected": n_expected}
    cut = None
    if sc.window_start >= 0 and sc.warmup_frac > 0:
        span = df["request_arrived_at"].max() - df["request_arrived_at"].min()
        cut = float(df["request_arrived_at"].min() + sc.warmup_frac * span)
        df = df[df["request_arrived_at"] >= cut]
    s = summarize_rows(df.to_dict("records"))
    gaps_file = f.parent / "token_gaps.npz"
    if gaps_file.exists():
        gaps = np.load(gaps_file)
        g = gaps["gap"][np.isin(gaps["request_id"], df["Request Id"].to_numpy())]
        if len(g):
            s.update({"p50_tbt": float(np.percentile(g, 50)), "p95_tbt": float(np.percentile(g, 95)),
                      "p99_tbt": float(np.percentile(g, 99)), "p999_tbt": float(np.percentile(g, 99.9)),
                      "mean_tbt": float(g.mean()), "n_tbt": int(len(g))})
    s.update({"ok": True, "completed": completed, "n_expected": n_expected, "scored": len(df), "cut": cut})
    info_files = list(out_dir.rglob("predictor_info.json"))
    if info_files:
        info = json.loads(info_files[0].read_text())
        s.update({"bundle": ev.bundle_fingerprint(info["bundle_key"]), "clamp_hits": info.get("clamp_hits", {})})
    return s


def bundle_of(h) -> str:
    return ev.bundle_fingerprint(h.sim._scheduler._execution_time_predictor._bundle_key())


# ------------------------------------------------------------------ horizons and scores

@dataclass(frozen=True)
class Horizon:
    mode: str  # "time" | "decisions" | "full"
    value: float = 0.0
    rel: bool = False  # time given as a multiple of pi_e's window P99 E2E
    drain: bool = True  # time mode: stop arrivals at the cutoff and drain (False: censor at the cutoff)

    @staticmethod
    def parse(spec: str) -> "Horizon":
        s = spec.strip()
        if s.lower() == "full":
            return Horizon("full")
        k, _, v = s.partition("=")
        if k.upper() == "H" and v.isdigit() and int(v) > 0:
            return Horizon("decisions", int(v))
        if k.upper() == "T" and v:
            drain = v.endswith("+drain")
            v = v[:-len("+drain")] if drain else v
            rel = v.endswith("x")
            try:
                x = float(v[:-1] if rel else v)
            except ValueError:
                x = 0.0
            if x > 0 and math.isfinite(x):
                return Horizon("time", x, rel, drain)
        raise ValueError(f"bad horizon {spec!r}: use H=<decisions>, T=<s>[+drain], T=<multiple>x[+drain] or full")

    def resolve(self, p99_e2e: float) -> "Horizon":
        return replace(self, value=self.value * p99_e2e, rel=False) if self.rel else self

    @property
    def label(self) -> str:
        if self.mode == "full":
            return "full"
        if self.mode == "decisions":
            return f"H{int(self.value)}"
        t = f"T{self.value:g}x" if self.rel else f"T{self.value:.4g}s"
        return t + ("+drain" if self.drain else "")


@dataclass(frozen=True)
class Plan:
    score: str = "tail"  # tail | lex | sum | hinge
    tau_e: float = 1.0  # pi_e's window P99 E2E
    tau_f: float = 1.0  # pi_e's window P99 TTFT
    weight: float = 0.0  # w of the tail score
    censor: str = "elapsed"  # elapsed | remaining
    tpot_ref: float = 0.0  # pi_e's window mean TPOT (censor="remaining")
    min_gain: float = 0.0  # a deviation needs a score gain above this (scalar scores)


def request_costs(reqs, cutoff: Optional[float], censor: str = "elapsed", tpot_ref: float = 0.0):
    """Per-request (E2E, TTFT, finished) at the cutoff. Finished requests: actual values. Unfinished: age at the
    cutoff (lower bound); ``censor="remaining"`` adds remaining decode tokens * ``tpot_ref`` to the E2E. Without a
    cutoff (drained rollouts) an unfinished request is an error."""
    e2e, ttft, fin = [], [], []
    for r in reqs:
        a = r.arrived_at
        if r._completed:
            e2e.append(r._completed_at - a)
            fin.append(True)
        else:
            if cutoff is None:
                raise RuntimeError(f"drained rollout left request {r.id} unfinished")
            e = cutoff - a
            if censor == "remaining":
                e += (r.total_tokens - max(r.num_processed_tokens, r.num_prefill_tokens)) * tpot_ref
            elif censor != "elapsed":
                raise ValueError(f"unknown censor rule {censor!r}")
            e2e.append(e)
            fin.append(False)
        ttft.append(r._prefill_completed_at - a if r._prefill_completed_at > 0 else cutoff - a)
    return np.array(e2e, float), np.array(ttft, float), np.array(fin, bool)


def rollout_terms(e2e, ttft, tau_e, tau_f) -> dict:
    return {"sum": float(e2e.sum()), "hinge": float(np.maximum(e2e / tau_e - 1.0, 0.0).sum()),
            "hinge_ttft": float(np.maximum(ttft / tau_f - 1.0, 0.0).sum())}


def score_parts(t: dict, plan: Plan) -> dict:
    """The additive parts of a scalar score (their sum is the score)."""
    if plan.score == "tail":
        return {"mean": t["sum"], "e2e_tail": plan.weight * t["hinge"], "ttft_tail": plan.weight * t["hinge_ttft"]}
    if plan.score in ("sum", "hinge"):
        return {plan.score: t[plan.score]}
    raise ValueError(plan.score)


def select_candidate(by: dict, pe: int, plan: Plan):
    """Best candidate and the decision record. ``by``: candidate -> rollout terms. Ties and gains within
    ``min_gain`` (plus 1e-9 relative) keep pi_e's choice ``pe``."""
    if plan.score == "lex":
        key = lambda c: (by[c]["hinge"], by[c]["sum"])  # noqa: E731
        best = min(by, key=lambda c: (key(c), c != pe, c))
        if all(x >= y - 1e-9 for x, y in zip(key(best), key(pe))):
            best = pe
        decided = "keep" if best == pe else ("hinge" if by[best]["hinge"] < by[pe]["hinge"] - 1e-9 else "sum")
        return best, {"decided_by": decided, "scores": {c: list(key(c)) for c in by}}
    parts = {c: score_parts(by[c], plan) for c in by}
    s = {c: sum(p.values()) for c, p in parts.items()}
    best = min(s, key=lambda c: (s[c], c != pe, c))
    gain = s[pe] - s[best]
    if gain <= plan.min_gain + 1e-9 * max(1.0, abs(s[pe])):
        best, gain = pe, 0.0
    term_gain = {k: parts[pe][k] - parts[best][k] for k in parts[pe]}
    decided = "keep" if best == pe else max(term_gain, key=term_gain.get)
    return best, {"decided_by": decided, "gain": gain, "term_gain": term_gain, "scores": s,
                  "parts": parts}


# ------------------------------------------------------------------ rollouts

def _inflight(h):
    gs = h.sim._scheduler
    out = list(gs._request_queue)
    for rs in gs._replica_schedulers.values():
        wq = rs._waiting_queue
        out += (wq.to_list() if hasattr(wq, "to_list") else list(wq)) + list(getattr(rs, "_running", []))
    return out


def _next_time(h) -> float:
    s = h.sim
    t = s._request_generator.get_next_request_arrival_time()
    q = s._event_queue[0]._time if s._event_queue else None
    ts = [x for x in (t, q) if x is not None]
    return min(ts) if ts else float("inf")


def apply_decision(h, choice: Optional[int]) -> dict:
    """Execute the pending routing decision with ``choice`` (None: pi_e's own) through sim_fork's forcing path;
    returns ``{"policy_choice", "executed"}``."""
    rec = {}
    override_next_decision(h, choice, record=rec)
    n = h.decisions
    h.run(lambda x: x.decisions > n)
    if "executed" not in rec:
        raise RuntimeError("the forced decision did not happen")
    return rec


def output_pool(sc) -> np.ndarray:
    """Sorted output lengths of the capacity-filtered trace outside the window (whole trace for a full-trace
    scenario)."""
    df = pd.read_csv(ev.TRACES[sc.trace], usecols=["num_prefill_tokens", "num_decode_tokens"])
    df = df[(df.num_prefill_tokens + df.num_decode_tokens) <= ev.max_request_tokens(sc)].reset_index(drop=True)
    d = df["num_decode_tokens"].to_numpy().astype(int).clip(min=1)
    if sc.window_start >= 0:
        d = np.concatenate([d[:sc.window_start], d[sc.window_start + sc.window_len:]])
    return np.sort(d)


def _draw(pool, above, hi, rng):
    """Draws from ``pool`` (sorted) conditioned on above < L <= hi (vectorised); L = above + 1 when empty."""
    above, hi = np.asarray(above), np.asarray(hi)
    lo_i = np.searchsorted(pool, above, "right")
    hi_i = np.searchsorted(pool, hi, "right")
    u = rng.random(len(above))
    idx = np.minimum(lo_i + (u * (hi_i - lo_i)).astype(int), len(pool) - 1)
    return np.where(hi_i > lo_i, pool[idx], np.minimum(above + 1, np.maximum(hi, 1)))


def generated_tokens(r) -> int:
    """Output tokens a request has generated (preemption restarts fold them into its prompt)."""
    folded = r.num_prefill_tokens - r.original_num_prefill_tokens
    return folded + max(r.num_processed_tokens - r.num_prefill_tokens, 0)


def resample_outputs(h, rng, pool, cap):
    """Replace every output length the router cannot know (in-flight requests: conditioned on what they have
    generated; future arrivals: unconditioned), capped at ``cap`` total tokens. Child-only state."""
    inflight = _inflight(h)
    if inflight:
        gen = np.array([generated_tokens(r) for r in inflight])
        hi = np.array([cap - r.original_num_prefill_tokens for r in inflight])
        new = _draw(pool, gen, np.maximum(hi, gen + 1), rng)
        for r, L in zip(inflight, new):
            L = int(L)
            r._num_decode_tokens = r.original_num_prefill_tokens + L - r.num_prefill_tokens
            r._original_num_decode_tokens = L
    g = h.sim._request_generator
    i = g.num_requests_generated
    arr = np.array(g._num_decode_tokens, copy=True)
    if len(arr) > i:
        pre = np.asarray(g._num_prefill_tokens[i:])
        arr[i:] = _draw(pool, np.zeros(len(pre), int), np.maximum(cap - pre, 1), rng)
    g._num_decode_tokens = arr


def _due(hz: Horizon, h, n0: int, cutoff: Optional[float]) -> bool:
    if hz.mode == "decisions":
        return h.at_decision() and h.decisions >= n0 + hz.value
    if hz.mode == "time":
        return _next_time(h) > cutoff
    return False  # full: the end of the run


def horizon_terms(reqs, n_aff: int, cutoff: Optional[float], plan: Plan) -> dict:
    """Rollout terms over the affected set ``reqs`` (``reqs[0]`` = the routed request, ``reqs[n_aff:]`` = arrivals
    after the decision), plus audit fields: the routed request's E2E and whether it was censored, and the E2E hinge
    over the routed request and later arrivals (``post_hinge``)."""
    e2e, ttft, fin = request_costs(reqs, cutoff, plan.censor, plan.tpot_ref)
    t = rollout_terms(e2e, ttft, plan.tau_e, plan.tau_f)
    post = np.zeros(len(e2e), bool)
    post[0] = True
    post[n_aff:] = True
    t.update({"n": int(len(e2e)), "n_unfinished": int((~fin).sum()),
              "post_hinge": float(np.maximum(e2e[post] / plan.tau_e - 1.0, 0.0).sum()),
              "routed_e2e": float(e2e[0]), "routed_censored": bool(not fin[0])})
    return t


def child_rollouts(h, choice, horizons: List[Horizon], plan: Plan, sample=None, timeout: float = 1800.0) -> dict:
    """One candidate's rollout under every horizon (run inside an ``os.fork`` child; mutates ``h``).

    The child runs once to the longest horizon. At each horizon's stop point it scores that horizon: a pure time
    cutoff is scored in place (censored); a horizon that drains is drained in a forked grandchild (or in place when
    it is the last one), so the shared prefix is simulated once. ``cpu_s`` per horizon estimates its standalone cost
    (the prefix up to its stop plus its own drain)."""
    after_os_fork(h)
    gc.disable()  # the child's GC would touch (and copy) every page of the parent
    c0 = time.process_time()
    if sample is not None:
        seed, pool, cap = sample
        resample_outputs(h, np.random.default_rng(seed), pool, cap)
    affected = _inflight(h)  # [0] is the request being routed
    n_aff = len(affected)
    g = h.sim._request_generator
    made = []
    orig_next = g.get_next_request

    def wrapped():
        r = orig_next()
        if r is not None:
            made.append(r)
        return r
    g.get_next_request = wrapped
    t_dec = _next_time(h)
    rec = apply_decision(h, choice)
    n0 = h.decisions - 1
    cut = [t_dec + hz.value if hz.mode == "time" else None for hz in horizons]

    def finish(i, drain, prefix, forked=False):
        if forked:
            after_os_fork(h)
        c1 = time.process_time()
        if drain:
            g.num_requests = g.num_requests_generated  # stop arrivals, then drain
            h.run()
        cutoff = None if (drain or h.done()) else cut[i]
        t = horizon_terms(affected + made, n_aff, cutoff, plan)
        t.update({"horizon": horizons[i].label, "cpu_s": prefix + time.process_time() - c1})
        return t

    out: List[Optional[dict]] = [None] * len(horizons)
    pending = list(range(len(horizons)))
    while pending:
        ended = not h.run(lambda x: any(_due(horizons[i], x, n0, cut[i]) for i in pending))
        due = list(pending) if ended else [i for i in pending if _due(horizons[i], h, n0, cut[i])]
        for i in due:
            pending.remove(i)
        due.sort(key=lambda i: horizons[i].mode == "time" and not horizons[i].drain, reverse=True)  # censored first
        for j, i in enumerate(due):
            hz = horizons[i]
            drain = not ended and (hz.mode == "decisions" or (hz.mode == "time" and hz.drain))
            prefix = time.process_time() - c0
            if not drain or (not pending and j == len(due) - 1):
                out[i] = finish(i, drain, prefix)
            else:
                kid = _fork_child(lambda i=i, p=prefix: finish(i, True, p, forked=True))
                out[i] = _collect_all([kid], timeout)[0]
    return {"choice": choice, "policy_choice": rec["policy_choice"], "executed": rec["executed"], "h": out}


# ------------------------------------------------------------------ os.fork children

@dataclass
class _Child:
    pid: int
    rfd: int
    err: object  # temporary file holding the child's stderr
    buf: bytearray
    status: Optional[int] = None
    result: Optional[dict] = None


def _fork_child(fn: Callable[[], dict]) -> _Child:
    err = tempfile.TemporaryFile()
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: never return into the caller's stack
        code = 0
        try:
            os.close(rfd)
            os.dup2(err.fileno(), 2)
            try:
                payload = {"ok": fn()}
            except BaseException:
                payload = {"error": traceback.format_exc()}
            data = json.dumps(payload).encode()
            while data:
                data = data[os.write(wfd, data):]
        except BaseException:
            code = 3
        finally:
            os._exit(code)
    os.close(wfd)
    return _Child(pid, rfd, err, bytearray())


def _stderr_tail(c: _Child, n=2000) -> str:
    try:
        c.err.seek(0, 2)
        size = c.err.tell()
        c.err.seek(max(0, size - n))
        return c.err.read().decode(errors="replace")
    except (OSError, ValueError):
        return ""


def _describe(c: _Child) -> str:
    if c.status is None:
        return f"pid {c.pid} (still running)"
    if os.WIFSIGNALED(c.status):
        return f"pid {c.pid} killed by signal {os.WTERMSIG(c.status)} ({signal.Signals(os.WTERMSIG(c.status)).name})"
    return f"pid {c.pid} exit status {os.WEXITSTATUS(c.status)}"


def _collect_all(kids: List[_Child], timeout: float) -> List[dict]:
    """Each child's result, read concurrently with a deadline. On the first failure (error payload, a child killed
    or exiting non-zero, a timeout) every other child is killed, all are reaped and the error names the child's pid,
    exit status or signal, its traceback and the tail of its stderr."""
    deadline = time.monotonic() + timeout
    open_ = {c.rfd: c for c in kids}
    try:
        while open_:
            ready, _, _ = select.select(list(open_), [], [], max(0.0, deadline - time.monotonic()))
            if not ready:
                raise RuntimeError(f"rollout child(ren) {[c.pid for c in open_.values()]} timed out after "
                                   f"{timeout:g} s (killed)")
            for fd in ready:
                c = open_[fd]
                chunk = os.read(fd, 1 << 16)
                if chunk:
                    c.buf += chunk
                    continue
                del open_[fd]
                os.close(fd)
                c.rfd = -1
                _, c.status = os.waitpid(c.pid, 0)
                try:
                    res = json.loads(bytes(c.buf)) if c.buf else None
                except ValueError:
                    res = None
                if res is None or "error" in res or c.status != 0:
                    why = res["error"] if res and "error" in res else "no result"
                    raise RuntimeError(f"rollout child {_describe(c)}: {why}\n--- child stderr (tail) ---\n"
                                       f"{_stderr_tail(c)}")
                c.result = res["ok"]
        return [c.result for c in kids]
    finally:
        for c in kids:
            if c.status is None:
                try:
                    os.kill(c.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, c.status = os.waitpid(c.pid, 0)
            if c.rfd >= 0:
                os.close(c.rfd)
                c.rfd = -1
            c.err.close()


def evaluate_candidates(h, cands, horizons: List[Horizon], plan: Plan, workers: int, timeout: float = 1800.0,
                        sample=None) -> List[dict]:
    """Every candidate's ``child_rollouts`` (``workers`` children at a time). Checks that all children saw the same
    pi_e choice and, per horizon, the same affected set size."""
    out = []
    for i in range(0, len(cands), workers):
        kids = [_fork_child(lambda c=c: child_rollouts(h, c, horizons, plan, sample, timeout))
                for c in cands[i:i + workers]]
        out += _collect_all(kids, timeout)
    if len({r["policy_choice"] for r in out}) != 1 or any(len({r["h"][j]["n"] for r in out}) != 1
                                                          for j in range(len(horizons))):
        raise RuntimeError(f"rollouts disagree on pi_e's choice or the affected set: "
                           f"{[(r['choice'], r['policy_choice'], [t['n'] for t in r['h']]) for r in out]}")
    return out


def _by(res, j: int, pe: int) -> dict:
    return {(pe if r["choice"] is None else r["choice"]): r["h"][j] for r in res}


def _cpu_children() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def read_log(path: Path, header: dict) -> dict:
    """Decisions already logged in ``path`` (k -> row), for a resume. A new log gets ``header`` as its first line;
    an existing one must carry the same header. A partial last line (a crash mid-write) is dropped."""
    header = json.loads(json.dumps(header))
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(json.dumps({"header": header}) + "\n")
        return {}
    text = path.read_text()
    lines = text.split("\n")
    complete, partial = lines[:-1], lines[-1]
    first = json.loads(complete[0]) if complete else {}
    if first.get("header") != header:
        old = first.get("header") or {}
        diff = sorted(k for k in set(old) | set(header) if old.get(k) != header.get(k))
        raise SystemExit(f"{path}: logged with other parameters ({diff}); move it away or use another --out")
    if partial:
        path.write_text("\n".join(complete) + "\n")
    rows = [json.loads(x) for x in complete[1:] if x]
    return {r["k"]: r for r in rows}


# ------------------------------------------------------------------ closed loop

def closed_loop(sc, hz: Horizon, plan: Plan, cut, out_dir, log_path, workers=4, future="true",
                max_decisions=None, candidates="all", child_timeout=1800.0, chooser=None, plan_every=1,
                header=None):
    """Run the window with lookahead decisions after ``cut``; write outputs to ``out_dir`` and one JSON line per
    planned decision to ``log_path`` (appended after a header; decisions already logged are replayed, see
    ``read_log``). ``chooser(by, pe, plan) -> (best, info)`` replaces ``select_candidate`` (tests);
    ``candidates="pe"`` evaluates only pi_e's choice (neutrality check). ``plan_every=K`` plans only every K-th
    decision after the cut (pi_e elsewhere): a different question, see the module docstring."""
    if hz.rel:
        raise ValueError("resolve a relative time horizon first")
    chooser = chooser or select_candidate
    logged = read_log(log_path, header if header is not None else {"horizon": hz.label})
    h = build_simulator(argv_for(sc, out_dir))
    sample_pool = output_pool(sc) if future == "sampled-lengths" else None
    cap = ev.max_request_tokens(sc)
    n_planned = n_dev = n_eligible = n_replayed = 0
    decided, cpu = {}, 0.0
    t0 = time.time()
    with open(log_path, "a") as log:
        while h.next_decision():
            req = h.sim._scheduler._request_queue[0]
            n = h.decisions
            skip = req.arrived_at < cut or (max_decisions is not None and n_planned >= max_decisions)
            if not skip:
                n_eligible += 1
                skip = (n_eligible - 1) % plan_every != 0
            if skip:
                h.run(lambda x: x.decisions > n)
                continue
            k = n + 1
            if k in logged:  # resume: replay the logged decision
                row = logged[k]
                rec = apply_decision(h, row["chosen"])
                if rec["policy_choice"] != row["pe"]:
                    raise SystemExit(f"resume diverged at decision {k}: pi_e chose {rec['policy_choice']}, the log "
                                     f"says {row['pe']}")
                best, pe, info = row["chosen"], row["pe"], row
                n_replayed += 1
                cpu += row.get("cpu_s", 0.0)
            else:
                c0 = _cpu_children()
                cands = list(range(sc.replicas)) if candidates == "all" else [None]
                sample = ((1000003 * n + 7), sample_pool, cap) if future == "sampled-lengths" else None
                res = evaluate_candidates(h, cands, [hz], plan, workers, child_timeout, sample)
                pe = res[0]["policy_choice"]
                by = _by(res, 0, pe)
                best, info = chooser(by, pe, plan)
                rec = apply_decision(h, best)
                if rec["policy_choice"] != pe or rec["executed"] != best:
                    raise RuntimeError(f"parent decision {k}: pi_e chose {rec['policy_choice']} (children: {pe}), "
                                       f"executed {rec['executed']} (wanted {best})")
                dc = _cpu_children() - c0
                cpu += dc
                keys = ("sum", "hinge", "hinge_ttft", "post_hinge", "routed_e2e", "routed_censored",
                        "n_unfinished", "cpu_s")
                cand = {str(c): {x: by[c][x] for x in keys} | {"score": info["scores"][c]} for c in sorted(by)}
                row = {"k": k, "t": req.arrived_at, "pe": pe, "chosen": best, "decided_by": info["decided_by"],
                       "gain": info.get("gain"), "term_gain": info.get("term_gain"), "cands": cand,
                       "n_aff": by[pe]["n"], "cpu_s": dc, "wall": time.time() - t0}
                log.write(json.dumps(row) + "\n")
                log.flush()
            n_planned += 1
            n_dev += best != pe
            decided[info["decided_by"]] = decided.get(info["decided_by"], 0) + 1
    if n_replayed != len(logged):
        raise SystemExit(f"{log_path}: {len(logged) - n_replayed} logged decisions were not reached on replay")
    h.write_output()
    return {"planned": n_planned, "deviations": n_dev, "deviation_frac": n_dev / max(n_planned, 1),
            "decided_by": decided, "replayed": n_replayed, "wall_s_session": time.time() - t0,
            "cpu_s_children": cpu, "cpu_s_per_decision": cpu / max(n_planned, 1)}


# ------------------------------------------------------------------ open-loop horizon selection

def open_loop(sc, horizons: List[Horizon], ref: Horizon, plan: Plan, cut, log_path, workers=4, future="true",
              every=10, max_points=None, child_timeout=1800.0, header=None) -> dict:
    """pi_e alone; at every ``every``-th decision after ``cut`` (at most ``max_points``) all candidates are rolled
    out under ``horizons`` + ``ref`` and logged. Returns ``openloop_stats`` of all logged points (resumable)."""
    if plan.score == "lex":
        raise ValueError("open-loop selection needs a scalar score")
    all_h = list(horizons) + [ref]
    if any(x.rel for x in all_h):
        raise ValueError("resolve relative time horizons first")
    logged = read_log(log_path, header if header is not None else {"horizons": [x.label for x in all_h]})
    rows = dict(logged)
    sample_pool = output_pool(sc) if future == "sampled-lengths" else None
    cap = ev.max_request_tokens(sc)
    n_eligible = n_points = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, open(log_path, "a") as log:
        h = build_simulator(argv_for(sc, tmp))
        while h.next_decision():
            req = h.sim._scheduler._request_queue[0]
            n = h.decisions
            if req.arrived_at >= cut:
                n_eligible += 1
                if (n_eligible - 1) % every == 0:
                    if max_points is not None and n_points >= max_points:
                        break
                    n_points += 1
                    if n + 1 not in rows:
                        c0 = _cpu_children()
                        sample = ((1000003 * n + 7), sample_pool, cap) if future == "sampled-lengths" else None
                        res = evaluate_candidates(h, list(range(sc.replicas)), all_h, plan, workers, child_timeout,
                                                  sample)
                        pe = res[0]["policy_choice"]
                        row = {"k": n + 1, "t": req.arrived_at, "pe": pe,
                               "h": {x.label: {str(c): t for c, t in sorted(_by(res, j, pe).items())}
                                     for j, x in enumerate(all_h)},
                               "cpu_s": _cpu_children() - c0, "wall": time.time() - t0}
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        rows[n + 1] = row
            h.run(lambda x: x.decisions > n)
    stats = openloop_stats(list(rows.values()), [x.label for x in horizons], ref.label, plan)
    cpu = [r["cpu_s"] for r in rows.values()]
    stats["_run"] = {"points": len(rows), "cpu_s_children": float(sum(cpu)),
                     "cpu_s_per_point": float(np.mean(cpu)) if cpu else None}
    return stats


def _ranks(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(method="average").to_numpy()


def openloop_stats(rows, labels, ref_label, plan: Plan) -> dict:
    """Per horizon, agreement of its candidate scores with the reference horizon's, over the logged points."""
    out = {}
    for lab in list(labels) + [ref_label]:
        acc = {k: [] for k in ("spearman", "top1", "deviate_agree", "pair_sign_agree", "ref_gain", "harmful",
                               "deviates", "cpu_s")}
        for row in rows:
            pe = row["pe"]
            by_h = {int(c): t for c, t in row["h"][lab].items()}
            by_r = {int(c): t for c, t in row["h"][ref_label].items()}
            s_h = {c: sum(score_parts(t, plan).values()) for c, t in by_h.items()}
            s_r = {c: sum(score_parts(t, plan).values()) for c, t in by_r.items()}
            best_h, _ = select_candidate(by_h, pe, plan)
            best_r, _ = select_candidate(by_r, pe, plan)
            cs = sorted(s_h)
            a, b = np.array([s_h[c] for c in cs]), np.array([s_r[c] for c in cs])
            if a.std() > 0 and b.std() > 0:
                acc["spearman"].append(float(np.corrcoef(_ranks(a), _ranks(b))[0, 1]))
            acc["top1"].append(best_h == best_r)
            acc["deviate_agree"].append((best_h != pe) == (best_r != pe))
            acc["pair_sign_agree"] += [np.sign(s_h[c] - s_h[pe]) == np.sign(s_r[c] - s_r[pe]) for c in cs if c != pe]
            acc["ref_gain"].append(s_r[pe] - s_r[best_h])
            acc["harmful"].append(s_r[best_h] > s_r[pe])
            acc["deviates"].append(best_h != pe)
            acc["cpu_s"].append(sum(t["cpu_s"] for t in by_h.values()))
        m = {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}
        out[lab] = {"points": len(rows), "spearman_mean": m["spearman"], "top1_agree": m["top1"],
                    "deviate_agree": m["deviate_agree"], "pair_sign_agree": m["pair_sign_agree"],
                    "ref_gain_mean": m["ref_gain"], "ref_gain_sum": float(np.sum(acc["ref_gain"])),
                    "harmful_frac": m["harmful"], "deviate_frac": m["deviates"],
                    "standalone_cpu_s_per_decision": m["cpu_s"]}
    return out


# ------------------------------------------------------------------ summary with provenance

def _sha(p) -> str:
    return ev.file_sha(Path(p).resolve())[:12]


def _row_key(params: dict) -> str:
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    hz = params.get("horizon") or "+".join(params.get("horizons", []))
    return f"{params.get('mode', 'closed')}_{params['score']}_{hz}_{params['future']}_{h}".replace("/", "_")


def load_summary(path: Path, meta: dict) -> dict:
    if not path.exists():
        return {"format": FORMAT, "meta": meta, "rows": {}}
    summ = json.loads(path.read_text())
    if summ.get("format") != FORMAT:
        raise SystemExit(f"{path} was written by an older headroom.py (no provenance on its rows); use a new --out")
    diff = sorted(k for k in set(meta) | set(summ.get("meta", {})) if meta.get(k) != summ["meta"].get(k))
    if diff:
        old = summ.get("meta", {})
        what = ", ".join(f"{k}: {old.get(k)!r} -> {meta.get(k)!r}" for k in diff)
        raise SystemExit(f"{path} holds results for a different setup ({what}); use a new --out")
    return summ


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def vs_pi_e(r: dict, pe: dict) -> dict:
    """Log ratios against pi_e (negative = better): the four P99s, mean E2E and their tail4 combination."""
    out = {f"log_p99_{m}": math.log(r[f"p99_{m}"] / pe[f"p99_{m}"]) for m in ("e2e", "ttft", "tpot", "tbt")}
    out["log_mean_e2e"] = math.log(r["mean_e2e"] / pe["mean_e2e"])
    out["tail4"] = 0.25 * sum(out[f"log_p99_{m}"] for m in ("e2e", "ttft", "tpot", "tbt"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="closed", choices=("closed", "openloop"))
    ap.add_argument("--trace", default="Tool")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=2000)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--model", default="l3_8b")
    ap.add_argument("--horizon", help="comma-separated: H=<n>, T=<s>[+drain], T=<f>x[+drain], full. closed: one "
                    "closed-loop run per horizon (default T=1x+drain); openloop: the horizons compared (default "
                    "H=64,T=0.5x+drain,T=1x+drain,T=2x+drain,T=1x)")
    ap.add_argument("--reference", default="full", help="openloop: the reference horizon")
    ap.add_argument("--every", type=int, default=10, help="openloop: every k-th decision after the warm-up")
    ap.add_argument("--max-points", type=int, help="openloop: at most this many decisions")
    ap.add_argument("--score", default="tail", choices=("tail", "lex", "hinge", "sum"))
    ap.add_argument("--tail-weight", type=float, help="w of the tail score (default 25 * pi_e's mean E2E)")
    ap.add_argument("--censor", default="elapsed", choices=("elapsed", "remaining"), help="pure T=<t> only")
    ap.add_argument("--min-gain", type=float, default=0.0, help="score gain a deviation needs (scalar scores)")
    ap.add_argument("--future", default="true", choices=("true", "sampled-lengths"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--child-timeout", type=float, default=1800.0, help="seconds per batch of rollout children")
    ap.add_argument("--max-decisions", type=int)
    ap.add_argument("--plan-every", type=int, default=1, help="closed: plan every K-th decision (another question)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    default_h = "T=1x+drain" if a.mode == "closed" else "H=64,T=0.5x+drain,T=1x+drain,T=2x+drain,T=1x"
    horizons = [Horizon.parse(s) for s in (a.horizon or default_h).split(",") if s.strip()]
    sc = ev.Scenario(a.trace, a.replicas, a.scale, a.start, a.len, a.p, model=a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ_path = out / "summary.json"
    with tempfile.TemporaryDirectory() as tmp:
        bundle = bundle_of(build_simulator(argv_for(sc, tmp)))
    meta = {"scenario": sc.tag, "policy": Path(POLICY).name, "policy_sha": _sha(POLICY), "mooncake_sha":
            _sha(MOONCAKE), "code_version": ev.code_version(), "bundle": bundle}
    summ = load_summary(summ_path, meta)
    rows = summ["rows"]

    def save():
        _atomic_json(summ_path, summ)

    def check_bundle(name):
        b = rows[name].get("bundle")
        if b is not None and b != bundle:
            raise SystemExit(f"{name} ran with prediction bundle {b}, expected {bundle}")

    if "pi_e" not in rows:  # pi_e alone, in process (the harness's subprocess run is pi_e_harness)
        with tempfile.TemporaryDirectory() as tmp:
            h = build_simulator(argv_for(sc, tmp))
            h.run()
            h.write_output()
            r = score_output(tmp, sc)
        if not r["ok"]:
            raise SystemExit(f"pi_e failed: {r['reason']}")
        rows["pi_e"] = r
        save()
    check_bundle("pi_e")
    if a.skip_baselines:
        print("WARNING: --skip-baselines: the in-process pi_e is not checked against evaluate.run_one")
    else:
        for name, sched, pol in (("pi_e_harness", "evolved", POLICY), ("lor", "lor", None),
                                 ("mooncake", "evolved", MOONCAKE)):
            if name not in rows:
                r = ev.run_one(sched, sc, policy=pol)
                if not r.get("ok"):  # not stored: a later run retries it
                    print(f"WARNING: baseline {name} failed: {r.get('reason')} {r.get('stderr', '')}")
                    continue
                rows[name] = {k: r.get(k) for k in ("ok", "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e",
                                                     "mean_ttft", "scored", "bundle")}
                save()
            check_bundle(name)
        hx = rows.get("pi_e_harness")
        if hx and any(hx[k] != rows["pi_e"][k] for k in ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e")):
            raise SystemExit("the in-process pi_e run differs from evaluate.run_one's; the fork path is not the "
                             "harness's simulation")
    pe = rows["pi_e"]
    weight = a.tail_weight if a.tail_weight is not None else 25.0 * pe["mean_e2e"]
    plan = Plan(a.score, pe["p99_e2e"], pe["p99_ttft"], weight, a.censor, pe["mean_tpot"], a.min_gain)
    common = {"score": a.score, "future": a.future, "weight": weight if a.score == "tail" else None,
              "min_gain": a.min_gain, "tau_e": plan.tau_e, "tau_f": plan.tau_f, "tpot_ref": plan.tpot_ref,
              "censor": a.censor, "cut": pe["cut"], "policy_sha": meta["policy_sha"], "code_sha": CODE_SHA}
    failed = False
    if a.mode == "openloop":
        ref = Horizon.parse(a.reference)
        hs = [x.resolve(pe["p99_e2e"]) for x in horizons]
        params = dict(common, mode="openloop", horizons=[x.label for x in horizons], reference=ref.label,
                      horizon_values=[x.value for x in hs], every=a.every, max_points=a.max_points)
        key = _row_key(params)
        stats = open_loop(sc, hs, ref.resolve(pe["p99_e2e"]), plan, pe["cut"], out / f"{key}.jsonl", a.workers,
                          a.future, a.every, a.max_points, a.child_timeout, header=params)
        rows[key] = {"params": params, "stats": stats}
        save()
        print(json.dumps({key: rows[key]}, indent=1))
    else:
        for hz in horizons:
            hz_abs = hz.resolve(pe["p99_e2e"])
            params = dict(common, mode="closed", horizon=hz.label, horizon_mode=hz_abs.mode,
                          horizon_value=hz_abs.value, max_decisions=a.max_decisions, plan_every=a.plan_every)
            key = _row_key(params)
            if key in rows:
                print(f"{key}: already in {summ_path}, reused")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                info = closed_loop(sc, hz_abs, plan, pe["cut"], tmp, out / f"{key}.jsonl", a.workers, a.future,
                                   a.max_decisions, child_timeout=a.child_timeout, plan_every=a.plan_every,
                                   header=params)
                res = score_output(tmp, sc)
            info.update(res)
            if not res["ok"]:  # not stored as final; the log stays for inspection
                print(f"ERROR: {key} failed the completion gate: {res['reason']}")
                failed = True
                continue
            info["vs_pi_e"] = vs_pi_e(res, pe)
            rows[key] = {"params": params, **info}
            save()
            print(json.dumps({key: rows[key]}, indent=1))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
