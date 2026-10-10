"""Closed-loop headroom pilot (RL design discussion, 2026-10-10; reviews G r1-r4).

Question: with the true future (hindsight), does picking at every routing decision the candidate replica whose
rollout scores best, with pi_e (the evolved rule l3_48a067a7df63) as the rollout policy, improve the window's P99
tails over pi_e alone? With the ``full`` horizon this is exact hindsight one-step policy improvement over pi_e under
the proxy score below (no truncation): a null bounds the gain of one-step lookahead ON THAT SCORE, not "the
headroom" of every planner.

Modes (``--mode``):
- ``closed`` (headline: ``--horizon full``, the default): at each decision after the warm-up every candidate replica
  is tried in an ``os.fork`` child (forced with ``sim_fork.override_next_decision``, then pi_e routes the rest of the
  window, arrivals continuing, to completion) and the affected set is scored; the parent executes the best candidate
  (ties and gains <= ``min_gain`` keep pi_e's choice) through the same forcing path and continues. Only the first
  request of a burst is planned (bursts are single requests with jittered arrivals). Early decisions' affected sets
  include requests from before the warm-up cut, which the window metrics do not score.
- ``openloop`` (run FIRST in each window): pi_e runs alone; at every ``--every``-th decision after the cut all
  candidates are rolled out under each ``--horizon`` and under ``--reference`` (``full``); judges the cheaper proxy
  horizons against ``full``, gives ``full``'s own one-step gain distribution, and pre-specifies the closed loop's tail
  weight (``weight_sweep``, below).
- ``placebo``: ``--seeds`` closed loops that deviate from pi_e uniformly at random at the lookahead run's deviation
  rate (``--rate``, default: the closed ``full`` row's), no rollouts.
- ``noise``: ``--seeds`` pi_e runs with every arrival after the warm-up cut jittered by U(-d, d) seconds
  (``--noise-delta``, pre-registered d = 5 ms, far below the trace's timing resolution): the run-to-run spread of the
  window P99s under a perturbation that should not matter (the P99 lottery of the trajectory). It acts throughout the
  scored window. (Review G r4: the earlier ``tiebreak`` kind reseeded pi_e's tie-break RNG, which on Tool w0+2000 is
  drawn only at t <= 2 s, before the cut, so it relabelled idle replicas and measured nothing; ``token`` moved no
  P99. Both are removed.)
- ``pilot``: ``--out`` = the 4 windows' directories, comma-separated; prints ``pilot_verdict``.

Horizons (``--horizon``; within a decision the affected set is identical for all candidates):
- ``full``: the rest of the window with arrivals continuing, run to completion. Exact, no truncation bias. Its
  reference drain at the window end is the same one the scored window has.
- ``T=<t>+drain`` (t in seconds or ``<f>x`` = f times pi_e's window P99 E2E): arrivals for t seconds, then a
  contention-free drain of the affected set. A PROXY: the drain under-states the tails of requests still in flight at
  the cutoff (review G r3, Tool K=400: it removed 70-88 % of the set's E2E hinge, differently per candidate, and
  flattered the candidate it would have picked). Valid only as far as ``openloop`` shows it ranks like ``full``.
- ``T=<t>`` (pure cutoff, censored; ``--censor elapsed|remaining``): BLIND SPOT (review G r2): every request arriving
  at or after the decision has E2E <= t, so at ``T=1x`` its E2E hinge is identically 0; censoring credits deferral.
- ``H=<n>``: n decisions, then a drain (the r1 proxy; ~8 s of arrivals on Tool at H=64).

Open-loop statistics (``openloop_stats``). Per point and horizon: Spearman correlation of the candidate scores with
``full``'s, top-1 and deviate-or-not agreement, pairwise sign agreement against pi_e's choice, ``ref_gain`` = the
``full`` score gain of the horizon's choice over pi_e's choice, regret = ``full`` score of the horizon's choice minus
``full``'s best. Every statistic is stratified: ``informative`` (the horizon ended before the window end, n_h <
n_full; PRIMARY) and ``reached_end`` (identical to ``full`` by construction). Means come with paired bootstrap 95 %
CIs over points; paired horizon differences (points informative for both) of ``ref_gain`` and regret too. Diagnostics
from the ``full`` child: each horizon's affected set is also scored under continued arrivals (``cont``), separating
drain bias (horizon vs cont) from post-horizon externality (cont vs full). CIs are circular moving-block bootstraps
over points in decision order (``BLOCK`` = 20 points; neighbouring points share almost all of their ``full`` future,
review G r4), and the lag-1 autocorrelation of the per-point gains is reported; with gaps (stratum, NaN) the blocks
span them, and the CIs remain conditional on the one pi_e trajectory. Noise floor per point: pi_e's own choice
rolled out to ``full`` with every later arrival jittered by U(-d, d) (``jitter_future``, the same d as the noise
runs; a routing perturbation right after the decision, so it cannot be 0 by construction); |delta full score| is
reported beside the gains (median, p90, count of exact zeros, median gain/floor ratio).
HEADLINE BOUND FOR A NULL: ``full``'s own one-step gain distribution (median, p90, mean with CI) at pi_e's states (not
the lookahead trajectory's states; standard for a policy-improvement check), DESCRIPTIVE beside the floor: the
argmin over R candidates in hindsight also harvests chaotic differences (G r3: the post-horizon part of ``full``'s
candidate differences is barely rank-correlated with anything shorter), so its gain exceeds a single-draw floor even
without causal headroom, and "gain within the floor" is not used as a criterion.

Tail weight (``weight_sweep``; PRE-SPECIFIED 2026-10-10, before any rerun). Review G r4 smoke (Tool w0+2000, 2
points): under w = 25 m ``full``'s argmin cut the summed E2E by ~2,000 s and made BOTH tail hinges worse (13 of 14
alternatives: lower sum, higher E2E hinge), i.e. the score's mean term (~9x the tails) bought mean with tail. For w = f
m, f in 25, 100, 400 and ``tails`` (hinges only), the open loop reports the per-term change of ``full``'s argmin vs
pi_e's choice. Rule: the closed loop uses the SMALLEST w on that grid whose argmin does not worsen either tail term on
average over the window's open-loop points (mean change of the E2E hinge <= 0 and of the TTFT hinge <= 0); if none,
``tails`` (flagged ``rule_met: False``). ``--tail-weight auto`` (the closed loop's default) reads that choice from the
window's complete open-loop row; the verdict uses the closed row at that weight only.

PRE-SPECIFIED SUCCESS RULE (fixed 2026-10-10, revised after review G r4, before any rerun). Windows: Tool and Conv,
8R s0.75 p1, w0+2000 and w2000+2000 (4 windows). Per window, with L = the closed ``full`` lookahead run's log P99
ratios vs pi_e (tail score at the pre-specified weight), P = >= 20 placebo runs at the same deviation rate, N = >= 20
noise runs (arrival jitter, d = 5 ms; only rows of the current per-mode ``decision_sha``, one noise kind, never
pooled):
- ``positive``: L_e2e < P5(P_e2e) and L_e2e < P2.5(N_e2e), and likewise for TTFT; and (guard against token pace) L
  for TPOT and TBT <= P97.5(N) of the same metric;
- ``negative``: L_e2e or L_ttft > P97.5(N) of that metric;
- ``null``: L_e2e and L_ttft within [P2.5(N), P97.5(N)];
- ``inconclusive``: anything else.
Lookahead headroom is claimed only if all 4 windows are ``positive``; "no headroom" (for one-step lookahead on the
pre-specified score) only if all 4 are ``null`` or ``negative`` (the former open-loop leg, "gain within the noise floor
at >= 90 % of points", is dropped: see the headline bound above); anything else is inconclusive. ``window_verdict``
writes the per-window label into the summary (row ``verdict``); ``pilot_verdict`` (``--mode pilot``) the claim.

Cost (CPU-s per planned decision = R rollouts; Mac at load ~6-7, upper bounds; Alpine is faster per core). Tool 8R
s0.75 w0+2000, a full pi_e run ~25 s: ``full`` from the first decision after the cut 81 (10 per rollout), falling
about linearly to 0 at the window end, so ~40 on average: ~1,658 x 40 = 66,000 CPU-s = ~18 CPU-h per window.
Proxies at the first point: ``H=64`` 9.6 (r1 closed loop: 16.4 on average), ``T=0.5x+drain`` 25.7,
``T=1x+drain`` 44.7, ``T=2x+drain`` 77, ``T=1x`` 32.6. Open loop (one child per candidate serves every horizon:
it runs to ``full`` and drains the shorter horizons in forked grandchildren; plus one noise rollout): 121 CPU-s at
the first point, ~70 on average, so ``--every 10`` (165 points) ~3.2 CPU-h. Placebo and noise: one pi_e-speed run
each (~25 CPU-s). ``--plan-every K`` divides the closed-loop cost by K but answers a different question (lookahead at
1/K of the decisions; a null at K > 1 does not bound K = 1, a positive is not a per-decision estimate).

Scores (``--score``), from per-request E2E e_i and TTFT f_i over the affected set; tau_e, tau_f = pi_e's window P99
E2E and P99 TTFT, m = pi_e's window mean E2E:
- ``tail`` (default): ``sum_i e_i + w * (sum_i (e_i/tau_e - 1)+ + sum_i (f_i/tau_f - 1)+)``, w = 25 m unless
  ``--tail-weight``. The hinges are the tail terms (Rockafellar-Uryasev: ``tau + sum_j (x_j - tau)+ / (0.01 N)`` with
  tau = the window P99 bounds CVaR_0.99, tight at tau). The weight is a DESIGN CHOICE, a mean regulariser:
  evaluate.score on one scenario puts 0.375 on each log P99 and has a one-sided mean term (only above 1.05 x the best
  baseline's mean; likely inactive at pi_e, then its weight on the summed E2E is 0; when active the linearisation
  gives w = 37.5 m). TTFT has its own hinge (placements trade TTFT against E2E). TPOT and TBT are not scored; every
  closed row reports ``vs_pi_e`` (log ratios of the four P99s and mean E2E, and tail4) so token-pace damage shows.
  A P99 regression under a mean-dominated weight is not evidence against headroom (it may be the weight's trade of
  tail for mean); hence the open-loop weight rule above.
  Logged per decision: each candidate's score and terms, the per-term gain of the choice and ``decided_by``.
- ``lex`` (r1 rule, closed only), ``sum``, ``hinge``.

Resampled output lengths (``--future sampled-lengths``; a PARTIAL information gap): only output lengths are
resampled in the children (future arrivals unconditioned, in-flight requests conditioned on tokens generated, capped
at ``max_request_tokens``, from the trace outside the window); arrival times, prompts, prefixes and sessions stay true.
Known gaps: a later session turn's prompt carries the original output block ids, so when an earlier turn's output is
resampled longer (fresh ids) or shorter (fewer cached blocks) the later turn loses prefix hits it has in the true
future: sampled rollouts see systematically fewer session prefix hits, about equally across candidates. Only the
replicas' ``_kv_cache_manager`` lists are refitted, not a disk tier's (unused by the pilot scenarios).

Reuse and resume. ``<out>/summary.json`` carries ``meta`` (scenario, policy and Mooncake sha, code_version,
prediction bundle); another meta or format is refused. Log files and row keys hash all parameters plus
``decision_sha(mode)``: per mode, the source of the functions that decide what its log or run contains (the rollout,
forcing, scoring core plus that mode's loop or run code) and sim_fork.py, so docstring, statistics and control-code
edits do not orphan a closed loop's log. Still: land every change before the Alpine chains start. Summary rows
also store ``post_sha()`` (the post-processing source); a row whose post_sha differs is recomputed from its log (a
closed run replays its log, ~one pi_e run; open-loop statistics are pure post-processing, refreshed by rerunning
``--mode openloop``, which replays its log without new rollouts). Each log is appended one JSON line per decision
after a header; a run resumes from it (header must match; replay checks pi_e's choice and the arrival time at each
logged decision). Rows that fail the completion gate are not stored.

Alpine (acpu, <= 22 CPUs; jobs resume, so chain 4 h jobs; ship cache/prediction_bundle_<key>.pkl so the bundle in
meta matches): one window per job, ``--workers 8`` (8 rollouts at once, one batch per decision)::

  cd $REPO; jid=
  for i in 1 2 3 4 5 6 7 8; do
    jid=$(sbatch --parsable ${jid:+--dependency=afterany:$jid} --cpus-per-task=10 --mem=48G --time=04:00:00 \\
      --job-name=hr_tool_w0 --export=ALL,REPO_DIR=$REPO,CMD="python -m evolve.rl_plan.headroom --trace Tool \\
      --start 0 --workers 8 --out evolve/rl_plan/out/v4_tool_w0" scripts/submit_python_job.sbatch)
  done

A job that finds the run complete reuses the row and exits in ~1 min. Two such chains at once stay within 22 CPUs.

Usage (each window: open loop, closed loop at the open loop's weight, then placebo and noise; then the pilot):
  .venv/bin/python -m evolve.rl_plan.headroom --mode openloop --trace Tool --out evolve/rl_plan/out/v4_tool_w0
  .venv/bin/python -m evolve.rl_plan.headroom --trace Tool --out evolve/rl_plan/out/v4_tool_w0
  .venv/bin/python -m evolve.rl_plan.headroom --mode placebo --trace Tool --out evolve/rl_plan/out/v4_tool_w0
  .venv/bin/python -m evolve.rl_plan.headroom --mode noise --trace Tool --out evolve/rl_plan/out/v4_tool_w0
  .venv/bin/python -m evolve.rl_plan.headroom --mode pilot --out <tool_w0>,<tool_w2000>,<conv_w0>,<conv_w2000>
"""

import argparse
import gc
import hashlib
import inspect
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
NOISE_DELTA = 0.005  # s, pre-registered arrival jitter of the noise controls
SIM_FORK_SHA = hashlib.sha256(Path(sim_fork.__file__).read_bytes()).hexdigest()[:16]  # as loaded


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


def fit_hashes(hashes, n_tokens: int, block_size: int, tag: int) -> list:
    """A block-hash list for a request of ``n_tokens`` total tokens (the trace's lists cover prompt AND output, one
    id per full block, and ``Request`` asserts that): truncated, or extended with fresh ids that occur in no trace
    (trace ids are >= 0; fresh ones are -(tag * 10^6 + k + 1), unique per ``tag``)."""
    nb = n_tokens // block_size
    old = list(hashes)
    return old[:nb] if nb <= len(old) else old + [-(tag * 1_000_000 + k + 1) for k in range(len(old), nb)]


def resample_outputs(h, rng, pool, cap):
    """Replace every output length the router cannot know (in-flight requests: conditioned on what they have
    generated; future arrivals: unconditioned), capped at ``cap`` total tokens, and fit their block-hash lists to the
    new lengths (output blocks of a longer output get fresh ids, also in the replica KV managers' per-request hash
    lists, which must cover every full block). Child-only state."""
    from vidur.kv_cache.utils import BlockHashType
    gs = h.sim._scheduler
    kvs = [getattr(rs, "_kv_cache_manager", None) for rs in gs._replica_schedulers.values()]
    inflight = _inflight(h)
    if inflight:
        gen = np.array([generated_tokens(r) for r in inflight])
        hi = np.array([cap - r.original_num_prefill_tokens for r in inflight])
        new = _draw(pool, gen, np.maximum(hi, gen + 1), rng)
        for r, L in zip(inflight, new):
            L = int(L)
            r._num_decode_tokens = r.original_num_prefill_tokens + L - r.num_prefill_tokens
            r._original_num_decode_tokens = L
            if r._block_hash_ids is not None:
                r._block_hash_ids = fit_hashes(r._block_hash_ids, r.total_tokens, r._block_size, r.id)
                for kv in kvs:
                    lst = None if kv is None else kv.req_to_block_hashes.get(r.id)
                    if lst and len(lst) < len(r._block_hash_ids):
                        lst += [BlockHashType(x, tuple()) for x in r._block_hash_ids[len(lst):]]
    g = h.sim._request_generator
    i = g.num_requests_generated
    arr = np.array(g._num_decode_tokens, copy=True)
    if len(arr) > i:
        pre = np.asarray(g._num_prefill_tokens[i:])
        arr[i:] = _draw(pool, np.zeros(len(pre), int), np.maximum(cap - pre, 1), rng)
        hs = list(g._block_hash_ids)
        for j in range(i, len(arr)):
            if hs[j] is not None:
                hs[j] = fit_hashes(hs[j], int(g._num_prefill_tokens[j] + arr[j]), int(g._block_size[j]), 10**9 + j)
        g._block_hash_ids = hs
    g._num_decode_tokens = arr


def jitter_arrivals(g, i0: int, t_min: float, delta: float, rng) -> None:
    """Shift the arrival times of trace requests ``i0..num_requests`` (none generated yet) by U(-delta, delta)
    seconds, never before ``t_min``, and restore arrival order with a stable sort. Every per-request trace array of
    the generator (each list or array attribute as long as ``_arrived_at``: prompt and output lengths, block hashes,
    block size, session, and any column a later simulator version adds) is permuted with the times, so a request keeps
    all its fields. The arrays are replaced, not edited (a fork shares them with its parent)."""
    t_all = np.asarray(g._arrived_at)
    n_all, n = len(t_all), g.num_requests
    if n <= i0 or delta <= 0:
        return
    t = np.array(t_all, dtype=float, copy=True)
    t[i0:n] = np.maximum(t[i0:n] + rng.uniform(-delta, delta, n - i0), t_min)
    order = np.arange(n_all)
    order[i0:n] = i0 + np.argsort(t[i0:n], kind="stable")
    for name, v in list(vars(g).items()):
        if name == "_arrived_at" or not name.startswith("_") or isinstance(v, (str, bytes)):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_all:
            setattr(g, name, v[order])
        elif isinstance(v, list) and len(v) == n_all:
            setattr(g, name, [v[j] for j in order])
    g._arrived_at = t[order]


def jitter_future(h, delta: float, seed: int) -> str:
    """``jitter_arrivals`` on every arrival the simulation has not generated yet (after a decision: everything that
    arrives after it), never before the current time or the last generated arrival. Returns a description."""
    g = h.sim._request_generator
    i0 = g.num_requests_generated
    t_min = max(h.now, float(g._arrived_at[i0 - 1]) if i0 > 0 else h.now)
    jitter_arrivals(g, i0, t_min, delta, np.random.default_rng(seed))
    return f"arrivals {i0}.. jittered by U(-{delta:g}, {delta:g}) s, seed {seed}"


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


def child_rollouts(h, choice, horizons: List[Horizon], plan: Plan, sample=None, timeout: float = 1800.0,
                   after_decision: Optional[Callable] = None) -> dict:
    """One candidate's rollout under every horizon (run inside an ``os.fork`` child; mutates ``h``).

    The child runs once to the longest horizon. At each horizon's stop point it scores that horizon: a pure time
    cutoff is scored in place (censored); a horizon that drains is drained in a forked grandchild (or in place when
    it is the last one), so the shared prefix is simulated once. ``cpu_s`` per horizon estimates its standalone cost
    (the prefix up to its stop plus its own drain). When ``full`` is among the horizons, every other horizon's
    affected set is also scored at the end of the full run (``cont``: the same requests under continued arrivals).
    ``after_decision(h) -> str`` (noise-floor rollouts, ``full`` only) perturbs the child right after the decision;
    its description is returned as ``perturbed``."""
    if after_decision is not None and any(hz.mode != "full" for hz in horizons):
        raise ValueError("a perturbed (noise-floor) rollout scores only the full horizon")
    after_os_fork(h)
    gc.disable()  # the child's GC would touch (and copy) every page of the parent
    c0 = time.process_time()
    if sample is not None:
        seed, pool, cap = sample
        resample_outputs(h, np.random.default_rng(seed), pool, cap)
    perturbed = None
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
    if after_decision is not None:
        perturbed = after_decision(h)

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
    n_at_stop = [None] * len(horizons)
    pending = list(range(len(horizons)))
    while pending:
        ended = not h.run(lambda x: any(_due(horizons[i], x, n0, cut[i]) for i in pending))
        due = list(pending) if ended else [i for i in pending if _due(horizons[i], h, n0, cut[i])]
        for i in due:
            pending.remove(i)
            n_at_stop[i] = n_aff + len(made)
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
    if any(hz.mode == "full" for hz in horizons):  # the full run has completed every request
        for i, hz in enumerate(horizons):
            if hz.mode != "full":
                out[i]["cont"] = horizon_terms((affected + made)[:n_at_stop[i]], n_aff, None, plan)
    return {"choice": choice, "policy_choice": rec["policy_choice"], "executed": rec["executed"], "h": out,
            "perturbed": perturbed}


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


def _run_children(fns: List[Callable[[], dict]], workers: int, timeout: float) -> List[dict]:
    out = []
    for i in range(0, len(fns), workers):
        out += _collect_all([_fork_child(f) for f in fns[i:i + workers]], timeout)
    return out


def _check_consistent(res: List[dict], nh: int) -> None:
    if len({r["policy_choice"] for r in res}) != 1 or any(len({r["h"][j]["n"] for r in res}) != 1
                                                          for j in range(nh)):
        raise RuntimeError(f"rollouts disagree on pi_e's choice or the affected set: "
                           f"{[(r['choice'], r['policy_choice'], [t['n'] for t in r['h']]) for r in res]}")


def evaluate_candidates(h, cands, horizons: List[Horizon], plan: Plan, workers: int, timeout: float = 1800.0,
                        sample=None) -> List[dict]:
    """Every candidate's ``child_rollouts`` (``workers`` children at a time). Checks that all children saw the same
    pi_e choice and, per horizon, the same affected set size."""
    res = _run_children([lambda c=c: child_rollouts(h, c, horizons, plan, sample, timeout) for c in cands],
                        workers, timeout)
    _check_consistent(res, len(horizons))
    return res


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
    lines = path.read_text().split("\n")
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
    planned decision to ``log_path`` (appended after a header; decisions already logged are replayed, checking pi_e's
    choice and the arrival time, see ``read_log``). ``chooser(by, pe, plan) -> (best, info)`` replaces
    ``select_candidate`` (tests); ``candidates="pe"`` evaluates only pi_e's choice (neutrality check).
    ``plan_every=K`` plans only every K-th decision after the cut (pi_e elsewhere): a different question."""
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
                if rec["policy_choice"] != row["pe"] or row["t"] != req.arrived_at:
                    raise SystemExit(f"resume diverged at decision {k}: pi_e chose {rec['policy_choice']} for the "
                                     f"request arriving at {req.arrived_at}; the log says {row['pe']} at {row['t']}")
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


# ------------------------------------------------------------------ controls: placebo and noise

def placebo_run(sc, rate: float, seed: int, cut, out_dir, plan_every=1) -> dict:
    """pi_e with random deviations: at each eligible decision (after ``cut``, every ``plan_every``-th) a uniformly
    random replica is forced with probability q = rate * R / (R - 1), so pi_e's choice is replaced by a uniformly
    random other replica with probability ``rate`` (pi_e's choice need not be known in advance)."""
    R = sc.replicas
    q = min(1.0, rate * R / (R - 1))
    rng = np.random.default_rng(seed)
    h = build_simulator(argv_for(sc, out_dir))
    n_elig = n_dev = n_plan = 0
    while h.next_decision():
        req = h.sim._scheduler._request_queue[0]
        n = h.decisions
        if req.arrived_at >= cut:
            n_elig += 1
            if (n_elig - 1) % plan_every == 0:
                n_plan += 1
                choice = int(rng.integers(R)) if rng.random() < q else None
                rec = apply_decision(h, choice)
                n_dev += rec["executed"] != rec["policy_choice"]
                continue
        h.run(lambda x: x.decisions > n)
    h.write_output()
    return {"planned": n_plan, "deviations": n_dev, "deviation_frac": n_dev / max(n_plan, 1)}


def noise_run(sc, seed: int, cut, out_dir, delta: float) -> dict:
    """pi_e with every arrival at or after the warm-up ``cut`` jittered by U(-delta, delta) seconds (seeded,
    ``jitter_arrivals``; never moved before the cut or an earlier arrival): a perturbation of the scored window far
    below the trace's own timing resolution that should not matter, so the spread of the window P99s over seeds is the
    noise floor of a closed-loop P99 change (the P99 lottery of the trajectory). It acts at every scored decision,
    unlike pi_e's tie-break RNG, which on Tool w0+2000 is only drawn before t = 2 s (review G r4). The warm-up cut
    of ``score_output`` (from the first and last arrival) can move by <= warmup_frac * delta."""
    h = build_simulator(argv_for(sc, out_dir))
    g = h.sim._request_generator
    arr = np.asarray(g._arrived_at[:g.num_requests])
    i0 = int(np.searchsorted(arr, cut))
    jitter_arrivals(g, i0, max(cut, float(arr[i0 - 1])) if i0 > 0 else cut, delta,
                    np.random.default_rng(2_000_003 + seed))
    h.run()
    h.write_output()
    return {"jittered_from": i0, "delta": delta}


# ------------------------------------------------------------------ open-loop horizon selection

def open_loop(sc, horizons: List[Horizon], ref: Horizon, plan: Plan, cut, log_path, workers=4, future="true",
              every=10, max_points=None, child_timeout=1800.0, header=None, noise_delta=NOISE_DELTA,
              mean_e2e=None) -> dict:
    """pi_e alone; at every ``every``-th decision after ``cut`` (at most ``max_points``) all candidates are rolled
    out under ``horizons`` + ``ref`` (``full``), plus one noise rollout of pi_e's own choice to ``full`` with every
    later arrival jittered by U(-noise_delta, noise_delta) s (``jitter_future``); logged per point.
    Returns ``openloop_stats`` (``mean_e2e``: pi_e's window mean E2E, for the weight sweep)."""
    if not noise_delta > 0:
        raise ValueError("noise_delta must be > 0")
    if plan.score == "lex":
        raise ValueError("open-loop selection needs a scalar score")
    if ref.mode != "full":
        raise ValueError("the reference horizon must be full")
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
                        fns = [lambda c=c: child_rollouts(h, c, all_h, plan, sample, child_timeout)
                               for c in range(sc.replicas)]
                        jit = lambda x, s=1_000_003 + n: jitter_future(x, noise_delta, s)  # noqa: E731
                        fns.append(lambda: child_rollouts(h, None, [ref], plan, sample, child_timeout,
                                                          after_decision=jit))
                        res = _run_children(fns, workers, child_timeout)
                        res, noise = res[:-1], res[-1]
                        _check_consistent(res, len(all_h))
                        pe = res[0]["policy_choice"]
                        full_pe = _by(res, len(all_h) - 1, pe)[pe]
                        if noise["policy_choice"] != pe or noise["h"][0]["n"] != full_pe["n"]:
                            raise RuntimeError(f"noise rollout at decision {n + 1} disagrees with the candidates")
                        nz = {"perturbed": noise["perturbed"],
                              "delta": {k: noise["h"][0][k] - full_pe[k] for k in ("sum", "hinge", "hinge_ttft")}}
                        row = {"k": n + 1, "t": req.arrived_at, "pe": pe,
                               "h": {x.label: {str(c): t for c, t in sorted(_by(res, j, pe).items())}
                                     for j, x in enumerate(all_h)},
                               "noise": nz, "cpu_s": _cpu_children() - c0, "wall": time.time() - t0}
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        rows[n + 1] = row
            h.run(lambda x: x.decisions > n)
    stats = openloop_stats(list(rows.values()), [x.label for x in horizons], ref.label, plan, mean_e2e)
    cpu = [r["cpu_s"] for r in rows.values()]
    stats["_run"] = {"points": len(rows), "cpu_s_children": float(sum(cpu)),
                     "cpu_s_per_point": float(np.mean(cpu)) if cpu else None}
    return stats


def _ranks(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(method="average").to_numpy()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


BLOCK = 20  # moving-block bootstrap block length, in open-loop points (review G r4: neighbouring points share futures)


def boot_ci(x, B: int = 2000, seed: int = 0, block: int = 1) -> dict:
    """Mean of ``x`` (NaN dropped) with a bootstrap 95 % CI. ``block=1``: i.i.d. over entries (paired use: pass
    per-point differences); ``block=L``: circular moving-block bootstrap with blocks of L consecutive entries, for
    serially correlated points (entries in decision order; with NaN or stratum gaps dropped the blocks span them)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    L = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    if L == 1:
        idx = rng.integers(n, size=(B, n))
    else:
        nb = -(-n // L)
        idx = ((rng.integers(n, size=(B, nb))[:, :, None] + np.arange(L)) % n).reshape(B, nb * L)[:, :n]
    m = x[idx].mean(axis=1)
    out = {"n": int(n), "mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)),
           "hi": float(np.percentile(m, 97.5))}
    if L > 1:
        out["block"] = L
    return out


def lag1(x) -> Optional[float]:
    """Lag-1 autocorrelation of ``x`` in order (NaN dropped); None when undefined."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x[:-1].std() == 0 or x[1:].std() == 0:
        return None
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _point_metrics(row, lab, ref_label, plan, key=None) -> dict:
    """One point's agreement metrics of horizon ``lab`` (or its ``key`` sub-record, e.g. ``cont``) with the
    reference."""
    pe = row["pe"]
    by_h = {int(c): (t[key] if key else t) for c, t in row["h"][lab].items()}
    by_r = {int(c): t for c, t in row["h"][ref_label].items()}
    s_h = {c: sum(score_parts(t, plan).values()) for c, t in by_h.items()}
    s_r = {c: sum(score_parts(t, plan).values()) for c, t in by_r.items()}
    best_h, _ = select_candidate(by_h, pe, plan)
    best_r, _ = select_candidate(by_r, pe, plan)
    cs = sorted(s_h)
    others = [c for c in cs if c != pe]
    return {"spearman": _spearman(np.array([s_h[c] for c in cs]), np.array([s_r[c] for c in cs])),
            "top1": float(best_h == best_r), "deviate_agree": float((best_h != pe) == (best_r != pe)),
            "pair_sign_agree": float(np.mean([np.sign(s_h[c] - s_h[pe]) == np.sign(s_r[c] - s_r[pe])
                                              for c in others])) if others else float("nan"),
            "ref_gain": s_r[pe] - s_r[best_h], "regret": s_r[best_h] - s_r[best_r],
            "harmful": float(s_r[best_h] > s_r[pe]), "deviates": float(best_h != pe),
            "cpu_s": sum(t.get("cpu_s", 0.0) for t in row["h"][lab].values()),
            "informative": next(iter(row["h"][lab].values()))["n"] < next(iter(row["h"][ref_label].values()))["n"]}


_METRICS = ("spearman", "top1", "deviate_agree", "pair_sign_agree", "ref_gain", "regret", "harmful", "deviates",
            "cpu_s")
WEIGHT_GRID = ((25.0, "25m"), (100.0, "100m"), (400.0, "400m"), (1e9, "tails"))  # multiples of pi_e's mean E2E


def weight_sweep(rows, ref_label: str, mean_e2e: float, block: int = BLOCK) -> dict:
    """Per tail weight w = f * m (``WEIGHT_GRID``; ``tails`` ~ hinges only, the summed E2E a tie-break): the
    ``ref_label`` (full) argmin of the tail score at each point and its per-term change against pi_e's choice
    (argmin minus pi_e; negative = better): summed E2E, E2E hinge, TTFT hinge (unweighted), with block-bootstrap CIs
    and the fractions of points where each hinge improves or worsens. ``choice``: the PRE-SPECIFIED weight rule (fixed
    2026-10-10, before any rerun): the smallest w on the grid whose argmin does not worsen either tail term on
    average (mean change of the E2E hinge <= 0 and of the TTFT hinge <= 0); if none does, ``tails``
    (``rule_met: False``)."""
    rows = sorted(rows, key=lambda r: r["k"])
    out, choice = {}, None
    for f, name in WEIGHT_GRID:
        p = Plan("tail", weight=f * mean_e2e)
        d = {"sum": [], "hinge": [], "hinge_ttft": []}
        dev = []
        for r in rows:
            by_r = {int(c): t for c, t in r["h"][ref_label].items()}
            best, _ = select_candidate(by_r, r["pe"], p)
            dev.append(float(best != r["pe"]))
            for k in d:
                d[k].append(by_r[best][k] - by_r[r["pe"]][k])
        a = {k: np.array(v, float) for k, v in d.items()}
        out[name] = {"weight_m": f, "weight": f * mean_e2e, "deviates": boot_ci(dev, block=block),
                     **{f"d_{k}": boot_ci(v, block=block) for k, v in a.items()},
                     **{f"frac_{k}_{w}": (float(np.mean(op(a[k], 0.0))) if len(rows) else None)
                        for k in ("hinge", "hinge_ttft") for w, op in (("better", np.less), ("worse", np.greater))}}
        ok = len(rows) and a["hinge"].mean() <= 0 and a["hinge_ttft"].mean() <= 0
        if choice is None and ok:
            choice = {"label": name, "weight_m": f, "weight": f * mean_e2e, "rule_met": True}
    if choice is None:
        f, name = WEIGHT_GRID[-1]
        choice = {"label": name, "weight_m": f, "weight": f * mean_e2e, "rule_met": False}
    return {"grid": out, "choice": choice}


def openloop_stats(rows, labels, ref_label, plan: Plan, mean_e2e: Optional[float] = None, block: int = BLOCK) -> dict:
    """Per horizon and stratum (``informative`` = the horizon ended before the window end: PRIMARY;
    ``reached_end``; ``all``) the mean and block-bootstrap 95 % CI of each agreement metric, and the lag-1
    autocorrelation of ``ref_gain``; drain-bias and externality diagnostics from ``cont``; paired horizon differences
    of ref_gain and regret; the reference's own one-step gain against the per-point noise floor; with ``mean_e2e``,
    the tail-weight sweep and the pre-specified weight choice (``weight_sweep``). See the module docstring."""
    rows = sorted(rows, key=lambda r: r["k"])
    pm = {lab: [_point_metrics(r, lab, ref_label, plan) for r in rows] for lab in labels}
    out = {}
    for lab in labels:
        info_mask = np.array([p["informative"] for p in pm[lab]], bool)
        st = {}
        for name, mask in (("informative", info_mask), ("reached_end", ~info_mask),
                           ("all", np.ones(len(rows), bool))):
            st[name] = {"points": int(mask.sum())} | {
                m: boot_ci([p[m] for p, k in zip(pm[lab], mask) if k], block=block) for m in _METRICS}
        st["informative"]["ref_gain_lag1"] = lag1([p["ref_gain"] for p, k in zip(pm[lab], info_mask) if k])
        has_cont = rows and all("cont" in next(iter(r["h"][lab].values())) for r in rows)
        if has_cont:  # drain bias (horizon vs the same set under continued arrivals) and externality (cont vs full)
            sp_hc, sp_cf = [], []
            for r, k in zip(rows, info_mask):
                if not k:
                    continue
                cs = sorted(int(c) for c in r["h"][lab])
                s_h = np.array([sum(score_parts(r["h"][lab][str(c)], plan).values()) for c in cs])
                s_c = np.array([sum(score_parts(r["h"][lab][str(c)]["cont"], plan).values()) for c in cs])
                s_f = np.array([sum(score_parts(r["h"][ref_label][str(c)], plan).values()) for c in cs])
                sp_hc.append(_spearman(s_h, s_c))
                sp_cf.append(_spearman(s_c, s_f))
            st["informative"]["spearman_vs_cont"] = boot_ci(sp_hc, block=block)
            st["informative"]["spearman_cont_vs_full"] = boot_ci(sp_cf, block=block)
        out[lab] = st
    pairs = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            both = [k for k in range(len(rows)) if pm[a][k]["informative"] and pm[b][k]["informative"]]
            pairs[f"{a} - {b}"] = {m: boot_ci([pm[a][k][m] - pm[b][k][m] for k in both], block=block)
                                   for m in ("ref_gain", "regret")}
    out["paired_informative"] = pairs
    gains, floor = [], []
    for r in rows:
        by_r = {int(c): t for c, t in r["h"][ref_label].items()}
        s_r = {c: sum(score_parts(t, plan).values()) for c, t in by_r.items()}
        best_r, _ = select_candidate(by_r, r["pe"], plan)
        gains.append(s_r[r["pe"]] - s_r[best_r])
        nz = r.get("noise")
        floor.append(abs(sum(score_parts(nz["delta"], plan).values())) if nz else float("nan"))
    g, f = np.array(gains, float), np.array(floor, float)
    ok = ~np.isnan(f)
    out[ref_label] = {
        "points": len(rows), "one_step_gain": boot_ci(g, block=block), "gain_lag1": lag1(g),
        "gain_median": float(np.median(g)) if len(g) else None,
        "gain_p90": float(np.percentile(g, 90)) if len(g) else None,
        "noise_floor_median": float(np.median(f[ok])) if ok.any() else None,
        "noise_floor_p90": float(np.percentile(f[ok], 90)) if ok.any() else None,
        "noise_floor_zero": int((f[ok] == 0).sum()),
        "frac_gain_above_noise": float(np.mean(g[ok] > f[ok])) if ok.any() else None,
        "gain_over_floor_median": (float(np.median(g[ok] / np.maximum(f[ok], 1e-12))) if ok.any() else None),
        "noise_points": int(ok.sum())}
    if mean_e2e is not None and plan.score == "tail":
        sw = weight_sweep(rows, ref_label, mean_e2e, block)
        out[ref_label]["weights"] = sw["grid"]
        out[ref_label]["weight_choice"] = sw["choice"]
    return out


# ------------------------------------------------------------------ verdict (pre-specified rule)

def _params(v) -> dict:
    return v.get("params", {}) if isinstance(v, dict) else {}


def head_rows(rows: dict, sha: Optional[str] = None, weight: Optional[float] = None, plan_every: int = 1) -> list:
    """The closed full-horizon lookahead rows the rule is about: tail score, true future, no ``--max-decisions``,
    ``plan_every``; with ``sha`` only rows of that closed-loop ``decision_sha``, with ``weight`` only that tail
    weight. Shared by ``window_verdict`` and the placebo's default rate."""
    out = []
    for v in rows.values():
        p = _params(v)
        if (p.get("mode") == "closed" and p.get("horizon") == "full" and p.get("plan_every") == plan_every
                and p.get("max_decisions") is None and p.get("future") == "true" and p.get("score") == "tail"
                and (sha is None or p.get("decision_sha") == sha)
                and (weight is None or abs(p.get("weight", float("nan")) - weight) <= 1e-9 * max(1.0, weight))):
            out.append(v)
    return out


def window_verdict(rows: dict, shas: Optional[dict] = None, noise_delta: Optional[float] = None,
                   weight: Optional[float] = None) -> dict:
    """The module docstring's per-window rule from the summary rows: the closed ``full`` lookahead (``head_rows``),
    placebo runs at its deviation rate and noise runs. ``shas`` (mode -> ``decision_sha(mode)``) keeps only rows of
    the current code; noise rows must be one kind (``jitter``; ``noise_delta`` when given), never pooled."""
    shas = shas or {}
    heads = head_rows(rows, shas.get("closed"), weight)
    if len(heads) != 1:
        return {"label": "missing", "reason": f"{len(heads)} closed full-horizon lookahead rows"
                + (f" at weight {weight:.6g}" if weight is not None else "")}
    L = heads[0]["vs_pi_e"]
    rate = heads[0]["deviation_frac"]
    ctl = [v for v in rows.values() if _params(v).get("mode") in ("placebo", "noise")]
    cur = [v for v in ctl if shas.get(_params(v)["mode"]) in (None, _params(v).get("decision_sha"))]
    P = [v["vs_pi_e"] for v in cur if _params(v)["mode"] == "placebo" and abs(_params(v)["rate"] - rate) < 1e-9]
    kinds = {}
    for v in cur:
        p = _params(v)
        if p["mode"] == "noise":
            kinds.setdefault(f"{p.get('noise_kind')}:{p.get('noise_delta')}", []).append(v["vs_pi_e"])
    want = f"jitter:{noise_delta}" if noise_delta is not None else (next(iter(kinds)) if len(kinds) == 1 else None)
    counts = {"noise_by_kind": {k: len(v) for k, v in kinds.items()}, "stale_controls": len(ctl) - len(cur)}
    if want is None:
        return {"label": "missing", "reason": f"several noise kinds {sorted(kinds)}; name the pre-registered one",
                **counts}
    N = kinds.get(want, [])
    if len(P) < 20 or len(N) < 20:
        return {"label": "missing", "reason": f"{len(P)} placebo (rate {rate:.4f}) and {len(N)} noise ({want}) runs; "
                f"need 20", **counts}
    q = lambda xs, m, p: float(np.percentile([x[f"log_p99_{m}"] for x in xs], p))  # noqa: E731
    lab = {m: L[f"log_p99_{m}"] for m in ("e2e", "ttft", "tpot", "tbt")}
    pos = all(lab[m] < q(P, m, 5) and lab[m] < q(N, m, 2.5) for m in ("e2e", "ttft")) and \
        all(lab[m] <= q(N, m, 97.5) for m in ("tpot", "tbt"))
    neg = any(lab[m] > q(N, m, 97.5) for m in ("e2e", "ttft"))
    null = all(q(N, m, 2.5) <= lab[m] <= q(N, m, 97.5) for m in ("e2e", "ttft"))
    label = "positive" if pos else "negative" if neg else "null" if null else "inconclusive"
    return {"label": label, "lookahead": lab, "rate": rate, "weight": _params(heads[0]).get("weight"),
            "n_placebo": len(P), "n_noise": len(N), "noise_kind": want, **counts,
            "placebo_p5": {m: q(P, m, 5) for m in lab}, "noise_p2.5": {m: q(N, m, 2.5) for m in lab},
            "noise_p97.5": {m: q(N, m, 97.5) for m in lab}}


def pilot_verdict(windows: dict) -> dict:
    """The pre-specified cross-window claim from each window's summary rows (name -> rows, with ``verdict``):
    ``headroom`` only if all 4 windows are ``positive``; ``no headroom`` (for one-step lookahead on the pre-registered
    score) only if all 4 are ``null`` or ``negative``; else ``inconclusive``. Also collects each window's open-loop
    one-step gain against its noise floor (descriptive, not part of the rule)."""
    labels, ol = {}, {}
    for name, rows in windows.items():
        labels[name] = (rows.get("verdict") or {}).get("label", "missing")
        for v in rows.values():
            p = _params(v)
            if p.get("mode") == "openloop" and p.get("max_points") is None and "stats" in v:
                f = v["stats"].get(p.get("reference", "full"), {})
                ol[name] = {k: f.get(k) for k in ("one_step_gain", "gain_median", "noise_floor_median",
                                                  "frac_gain_above_noise", "gain_over_floor_median",
                                                  "weight_choice")}
    vals = list(labels.values())
    if len(vals) != 4:
        claim = "incomplete"
    elif all(x == "positive" for x in vals):
        claim = "headroom"
    elif all(x in ("null", "negative") for x in vals):
        claim = "no headroom"
    else:
        claim = "inconclusive"
    return {"claim": claim, "windows": labels, "openloop": ol}


# ------------------------------------------------------------------ summary with provenance

def _src_sha(objs) -> str:
    h = hashlib.sha256()
    for o in objs:
        h.update(inspect.getsource(o).encode())
    return h.hexdigest()[:16]


def decision_sha(mode: str = "closed") -> str:
    """What a ``mode``'s log or run depends on, per mode (review G r4: control code must not orphan closed logs):
    the rollout, forcing, sampling and scoring core plus that mode's loop (closed, openloop) or run code (placebo,
    noise), plus sim_fork.py."""
    core = [argv_for, Horizon, Plan, request_costs, rollout_terms, score_parts, select_candidate, _inflight,
            _next_time, apply_decision, output_pool, _draw, generated_tokens, fit_hashes, resample_outputs, _due,
            horizon_terms, child_rollouts, evaluate_candidates, _by]
    parts = {"closed": core + [closed_loop],
             "openloop": core + [jitter_arrivals, jitter_future, open_loop],
             "placebo": [argv_for, apply_decision, placebo_run],
             "noise": [argv_for, jitter_arrivals, noise_run]}
    if mode not in parts:
        raise ValueError(f"unknown mode {mode!r}")
    return _src_sha(parts[mode]) + ":" + SIM_FORK_SHA


def post_sha() -> str:
    """What a summary row computed from a log depends on (post-processing)."""
    return _src_sha([score_output, vs_pi_e, openloop_stats, _point_metrics, boot_ci, lag1, _spearman, weight_sweep,
                     head_rows, window_verdict])


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


def chosen_weight(rows: dict, future: str, psha: str) -> Optional[dict]:
    """The pre-specified tail weight (``weight_sweep``'s choice) from this window's complete open-loop row of the
    current open-loop ``decision_sha`` and post-processing; None when there is no such row. Refuses ambiguity and
    stale statistics."""
    ol = [v for v in rows.values() if _params(v).get("mode") == "openloop" and _params(v).get("max_points") is None
          and _params(v).get("future") == future and _params(v).get("decision_sha") == decision_sha("openloop")]
    if not ol:
        return None
    if any(v.get("post_sha") != psha for v in ol):
        raise SystemExit("the open-loop row's statistics are from older post-processing: rerun --mode openloop "
                         "(replays its log, no new rollouts)")
    ch = {json.dumps(v["stats"]["full"].get("weight_choice"), sort_keys=True) for v in ol}
    if len(ch) != 1 or "null" in ch:
        raise SystemExit(f"open-loop rows disagree on (or lack) the weight choice: {sorted(ch)}")
    return json.loads(ch.pop())


def parse_weight(spec: str, mean_e2e: float) -> float:
    s = spec.strip()
    try:
        w = float(s[:-1]) * mean_e2e if s.endswith("m") else float(s)
    except ValueError:
        raise SystemExit(f"bad --tail-weight {spec!r}: use a number, <f>m (f x pi_e's mean E2E) or auto")
    if not (w > 0 and math.isfinite(w)):
        raise SystemExit(f"bad --tail-weight {spec!r}: must be > 0")
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="closed", choices=("closed", "openloop", "placebo", "noise", "pilot"))
    ap.add_argument("--trace", default="Tool")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=2000)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--model", default="l3_8b")
    ap.add_argument("--horizon", help="comma-separated: full, H=<n>, T=<s>[+drain], T=<f>x[+drain]. closed: one "
                    "run per horizon (default full); openloop: the proxies compared with --reference (default "
                    "H=64,T=0.5x+drain,T=1x+drain,T=2x+drain,T=1x)")
    ap.add_argument("--reference", default="full", help="openloop: the reference horizon (must be full)")
    ap.add_argument("--every", type=int, default=10, help="openloop: every k-th decision after the warm-up")
    ap.add_argument("--max-points", type=int, help="openloop: at most this many decisions")
    ap.add_argument("--seeds", type=int, default=20, help="placebo/noise: number of runs")
    ap.add_argument("--rate", type=float, help="placebo: deviation rate (default: the closed full row's)")
    ap.add_argument("--noise-delta", type=float, default=NOISE_DELTA,
                    help="noise runs and the open-loop per-point noise rollout: arrival jitter U(-d, d) seconds "
                    "(pre-registered default)")
    ap.add_argument("--score", default="tail", choices=("tail", "lex", "hinge", "sum"))
    ap.add_argument("--tail-weight", help="w of the tail score: a number, <f>m (f x pi_e's mean E2E) or auto (the "
                    "open-loop row's pre-specified choice). Default: auto for closed, 25m otherwise")
    ap.add_argument("--censor", default="elapsed", choices=("elapsed", "remaining"), help="pure T=<t> only")
    ap.add_argument("--min-gain", type=float, default=0.0, help="score gain a deviation needs (scalar scores)")
    ap.add_argument("--future", default="true", choices=("true", "sampled-lengths"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--child-timeout", type=float, default=3600.0, help="seconds per batch of rollout children")
    ap.add_argument("--max-decisions", type=int)
    ap.add_argument("--plan-every", type=int, default=1, help="closed/placebo: plan every K-th decision "
                    "(another question)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", required=True, help="output directory (pilot: the 4 windows' directories, "
                    "comma-separated)")
    a = ap.parse_args(argv)
    if a.mode == "pilot":
        wins = {d: json.loads((Path(d) / "summary.json").read_text())["rows"] for d in a.out.split(",") if d}
        print(json.dumps({"pilot": pilot_verdict(wins)}, indent=1))
        return
    if not a.noise_delta > 0:
        raise SystemExit("--noise-delta must be > 0")
    default_h = "full" if a.mode == "closed" else "H=64,T=0.5x+drain,T=1x+drain,T=2x+drain,T=1x"
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
    psha = post_sha()

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
    # the pre-specified weight, once the open loop has run (the open-loop mode itself refreshes it below)
    w_pre = None if a.mode == "openloop" else chosen_weight(rows, a.future, psha)
    wspec = a.tail_weight or ("auto" if a.mode == "closed" else "25m")
    if wspec == "auto" and a.mode == "openloop":
        raise SystemExit("--tail-weight auto is the closed loop's; the open loop chooses it")
    if wspec == "auto":
        if a.score == "tail" and w_pre is None:
            raise SystemExit("--tail-weight auto: no complete open-loop row in this --out; run --mode openloop first "
                             "(it pre-specifies the weight) or pass --tail-weight")
        weight = w_pre["weight"] if w_pre else 25.0 * pe["mean_e2e"]
    else:
        weight = parse_weight(wspec, pe["mean_e2e"])
    plan = Plan(a.score, pe["p99_e2e"], pe["p99_ttft"], weight, a.censor, pe["mean_tpot"], a.min_gain)
    common = {"score": a.score, "future": a.future, "weight": weight if a.score == "tail" else None,
              "min_gain": a.min_gain, "tau_e": plan.tau_e, "tau_f": plan.tau_f, "tpot_ref": plan.tpot_ref,
              "censor": a.censor, "cut": pe["cut"], "policy_sha": meta["policy_sha"],
              "decision_sha": decision_sha(a.mode)}
    shas = {m: decision_sha(m) for m in ("closed", "placebo", "noise")}
    failed = False

    def store(key, params, info, res):
        nonlocal failed
        info.update(res)
        if not res["ok"]:  # not stored as final
            print(f"ERROR: {key} failed the completion gate: {res['reason']}")
            failed = True
            return
        info["vs_pi_e"] = vs_pi_e(res, pe)
        rows[key] = {"params": params, "post_sha": psha, **info}
        save()

    if a.mode == "openloop":
        ref = Horizon.parse(a.reference)
        hs = [x.resolve(pe["p99_e2e"]) for x in horizons]
        params = dict(common, mode="openloop", horizons=[x.label for x in horizons], reference=ref.label,
                      horizon_values=[x.value for x in hs], every=a.every, max_points=a.max_points,
                      noise_kind="jitter", noise_delta=a.noise_delta)
        key = _row_key(params)
        stats = open_loop(sc, hs, ref.resolve(pe["p99_e2e"]), plan, pe["cut"], out / f"{key}.jsonl", a.workers,
                          a.future, a.every, a.max_points, a.child_timeout, header=params,
                          noise_delta=a.noise_delta, mean_e2e=pe["mean_e2e"])
        rows[key] = {"params": params, "post_sha": psha, "stats": stats}
        save()
        print(json.dumps({key: rows[key]}, indent=1))
        try:
            w_pre = chosen_weight(rows, a.future, psha)
        except SystemExit as e:
            print(f"WARNING: no pre-specified weight: {e}")
    elif a.mode == "closed":
        for hz in horizons:
            hz_abs = hz.resolve(pe["p99_e2e"])
            params = dict(common, mode="closed", horizon=hz.label, horizon_mode=hz_abs.mode,
                          horizon_value=hz_abs.value, max_decisions=a.max_decisions, plan_every=a.plan_every)
            key = _row_key(params)
            if key in rows and rows[key].get("post_sha") == psha:
                print(f"{key}: already in {summ_path}, reused")
                continue
            with tempfile.TemporaryDirectory() as tmp:  # a complete log replays without new rollouts
                info = closed_loop(sc, hz_abs, plan, pe["cut"], tmp, out / f"{key}.jsonl", a.workers, a.future,
                                   a.max_decisions, child_timeout=a.child_timeout, plan_every=a.plan_every,
                                   header=params)
                store(key, params, info, score_output(tmp, sc))
            print(json.dumps({key: rows.get(key)}, indent=1))
    else:
        rate = a.rate
        if a.mode == "placebo" and rate is None:  # the same head the verdict uses
            heads = head_rows(rows, shas["closed"], w_pre["weight"] if w_pre else None, a.plan_every)
            if len(heads) != 1:
                raise SystemExit(f"--rate needed: {len(heads)} closed full-horizon rows (current code, true future, "
                                 f"no --max-decisions, plan_every {a.plan_every}"
                                 + (f", weight {w_pre['weight']:.6g}" if w_pre else "") + ") to take it from")
            rate = heads[0]["deviation_frac"]
        ctl = {"future": a.future, "score": "-", "cut": pe["cut"], "policy_sha": meta["policy_sha"],
               "decision_sha": decision_sha(a.mode)}  # the score weight does not affect a control run
        for seed in range(a.seeds):
            params = dict(ctl, mode=a.mode, seed=seed)
            if a.mode == "placebo":
                params.update(rate=rate, plan_every=a.plan_every)
            else:
                params.update(noise_kind="jitter", noise_delta=a.noise_delta)
            key = _row_key(params)
            if key in rows and rows[key].get("post_sha") == psha:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                info = (placebo_run(sc, rate, seed, pe["cut"], tmp, a.plan_every) if a.mode == "placebo"
                        else noise_run(sc, seed, pe["cut"], tmp, a.noise_delta))
                store(key, params, info, score_output(tmp, sc))
    rows["verdict"] = window_verdict(rows, shas, a.noise_delta, w_pre["weight"] if w_pre else None)
    save()
    print(json.dumps({"verdict": rows["verdict"]}, indent=1))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
