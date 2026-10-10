"""Closed-loop headroom pilot (RL design discussion, 2026-10-10; review G r1 fixes).

Question: with the true future (hindsight), does picking at every routing decision the candidate replica whose
rollout scores best, with pi_e (the evolved rule l3_48a067a7df63) as the rollout policy, improve the window's P99
tails over pi_e alone? The method must be able to answer either way, so the rollout objective is built to see what
forms window tails (sustained load after the decision) and to trade mean against tail as the window objective does.

Loop. At each decision after the warm-up, every candidate replica is tried in an ``os.fork`` child: the child forces
the arriving request onto that replica (``sim_fork.override_next_decision``), routes everything after it with pi_e
and scores the affected set. The parent takes the best-scoring candidate (ties and gains <= ``min_gain`` keep pi_e's
choice), executes it through the same forcing mechanism and continues. Only the first request of a burst is planned
(with jittered arrivals, p >= 1, bursts are single requests; ``route()`` policies with bursts are refused).

Horizons (``--horizon``, comma-separated for a sweep):
- ``T=<sec>`` or ``T=<f>x`` (default ``T=1x``): TIME horizon. The rollout runs for T seconds of simulated time after
  the decision WITH arrivals continuing (``<f>x`` = f times pi_e's window P99 E2E, so ``T=1x`` covers a tail request's
  whole lifetime). Affected set = requests in flight at the decision (including the one being routed) + every arrival
  up to the cutoff, identical for all candidates. Censoring at the cutoff (``--censor``): ``elapsed`` (default)
  scores an unfinished request by its age at the cutoff, a lower bound on its E2E (TTFT likewise when prefill has
  not finished); ``remaining`` adds its remaining decode tokens times pi_e's window mean TPOT (TTFT stays the lower
  bound). ``elapsed`` slightly favours candidates that push work past the cutoff; at T >= P99 E2E only requests
  arriving in the last part of the horizon are censored.
- ``H=<n>``: DECISION horizon (the r1 method): the next n routing decisions under pi_e, then arrivals stop and the
  system drains; every affected request finishes (none censored). Blind to load arriving after the n decisions
  (review G r1: H=64 is ~8 s of arrivals on Tool, against a 95 s P99 E2E).

Cost per planned decision = R rollouts (R = replicas, ``--workers`` at a time) of:
- ``H=n``: n decisions plus a drain. Tool 8R s0.75 w0+2000, H=64 (r1 run): 16.4 CPU-s per decision (2.05 s per
  rollout; 27,205 CPU-s for 1,658 decisions).
- ``T=t``: t seconds of loaded simulation per rollout. Measured 2026-10-10 (busy Mac, load ~6): Tool 8R s0.75
  w0+2000, T=1x = 94.8 s: 30.4 CPU-s per decision (3.8 s per rollout; a full pi_e run of the window is ~25 s), so a
  full window (1,658 planned decisions) is ~14 CPU-h, ~3.7 h wall at 4 workers. Conv 4R s1.5 w6000+200: T=1x
  (32.9 s) 6.0 CPU-s per decision, H=16 1.7. Cost grows about linearly in t. Every run records
  ``cpu_s_children`` and ``cpu_s_per_decision`` in the summary; ``--plan-every K`` divides the cost by K.

Scores (``--score``), from per-request E2E e_i and TTFT f_i over the affected set; tau_e, tau_f = pi_e's window P99
E2E and P99 TTFT, m = pi_e's window mean E2E:
- ``tail`` (default): ``sum_i e_i + w * (sum_i (e_i/tau_e - 1)+ + sum_i (f_i/tau_f - 1)+)``, w = 25 m unless
  ``--tail-weight``. Why: the window objective (evaluate's ``tail4``) is 0.25 log P99 per metric plus a mean-E2E
  term. By Rockafellar-Uryasev, ``tau + sum_j (x_j - tau)+ / (0.01 N)`` with tau = the window P99 bounds the window
  CVaR_0.99 (which bounds P99) and is tight at tau, so a placement that changes the window's hinge sum by d changes
  log CVaR_0.99 by about d / (0.01 N tau); a change D in the summed E2E changes log mean E2E by D / (N m).
  Equating the two marginal rates gives the per-term weights 0.25 * m / 0.01 = 25 m on the normalised hinges and 1
  on the summed E2E (N, the window's scored count, cancels). TTFT gets its own hinge because placements trade TTFT
  against E2E (the r1 lex run moved P99 TTFT by +73 %). TPOT and TBT are left out: they are per-token decode
  quantities that a truncated rollout measures poorly, and they moved by < 5 % under any placement rule seen so far.
  The scalarisation also fixes r1's lex flaw: a hinge gain of 1e-8 can no longer outrank a worse sum.
  Logged per decision: each candidate's terms (``mean`` = sum E2E, ``e2e_tail``, ``ttft_tail``, both weighted), the
  per-term gain of the chosen candidate over pi_e's choice and ``decided_by`` = the term with the largest gain
  (``keep`` when pi_e's choice is kept).
- ``lex``: (sum (e_i/tau_e - 1)+, sum e_i) lexicographically, ties within 1e-9 (the r1 rule).
- ``sum``, ``hinge``: one term each.

Sampled futures (``--future sampled``): in each decision's children (common random numbers: one seed per
decision), the output length of every request the router could not know is resampled: not-yet-arrived requests
from the empirical output-length distribution, in-flight requests (including the one being routed) from the same
distribution conditioned on exceeding the tokens they have already generated. All totals are capped at
``evaluate.max_request_tokens`` (the trace filter's cap, 128K context on Llama-3). The distribution is the
capacity-filtered trace OUTSIDE the window (the window's own realisations are never used; a full-trace scenario
falls back to the whole trace and the summary says so). Conditioning is on generated tokens only, not on the prompt.
The parent always continues on the true trace.

Reuse. ``<out>/summary.json`` carries ``meta`` (scenario tag, policy and Mooncake file sha, code_version,
prediction-bundle fingerprint). A file with a different meta, or from the r1 format, is refused: use a new --out.
Rows are reused only under that meta; each lookahead row is keyed by all its parameters (horizon, score, weight,
censor, future, min_gain, max_decisions) and stores them, so a row is never relabelled. tau, the warm-up cut and the
weights come from the pi_e row of the same file.

Usage:
  .venv/bin/python -m evolve.rl_plan.headroom --trace Tool --out evolve/rl_plan/out/tool_s075_w2000_v2
  .venv/bin/python -m evolve.rl_plan.headroom --trace Tool --len 600 --horizon H=64,T=0.5x,T=1x,T=2x \\
      --out evolve/rl_plan/out/tool_sweep
"""

import argparse
import gc
import hashlib
import json
import os
import resource
import select
import signal
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from evolve import evaluate as ev
from evolve.sim_fork import build_simulator, override_next_decision
from tail_metrics import summarize_rows  # evolve.evaluate puts scripts/ on sys.path

POLICY = str(ev.REPO / "evolve/finalists/l3_48a067a7df63.py")
MOONCAKE = str(ev.REPO / "evolve/policies/systems/mooncake_min_ttft.py")
FORMAT = 2


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
    mode: str  # "time" (seconds of simulated time, arrivals continue) | "decisions" (then drain)
    value: float
    rel: bool = False  # time given as a multiple of pi_e's window P99 E2E

    @staticmethod
    def parse(spec: str) -> "Horizon":
        k, _, v = spec.strip().partition("=")
        if k.upper() == "H" and v.isdigit() and int(v) > 0:
            return Horizon("decisions", int(v))
        if k.upper() == "T" and v:
            rel = v.endswith("x")
            x = float(v[:-1] if rel else v)
            if x > 0:
                return Horizon("time", x, rel)
        raise ValueError(f"bad horizon {spec!r}: use H=<decisions>, T=<seconds> or T=<multiple>x")

    def resolve(self, p99_e2e: float) -> "Horizon":
        return replace(self, value=self.value * p99_e2e, rel=False) if self.rel else self

    @property
    def label(self) -> str:
        if self.mode == "decisions":
            return f"H{int(self.value)}"
        return f"T{self.value:g}x" if self.rel else f"T{self.value:.4g}s"


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


def child_rollout(h, choice, hz: Horizon, plan: Plan, sample=None) -> dict:
    """One candidate's rollout (run inside an ``os.fork`` child; mutates ``h``)."""
    gc.disable()  # the child's GC would touch (and copy) every page of the parent
    t_wall = time.time()
    if sample is not None:
        seed, pool, cap = sample
        resample_outputs(h, np.random.default_rng(seed), pool, cap)
    affected = _inflight(h)
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
    if hz.mode == "time":
        cutoff = t_dec + hz.value
        h.run(lambda x: _next_time(x) > cutoff)
    else:
        cutoff = None
        n0 = h.decisions - 1
        h.run(lambda x: x.at_decision() and x.decisions >= n0 + hz.value)
        g.num_requests = g.num_requests_generated  # stop arrivals, then drain
        h.run()
    e2e, ttft, fin = request_costs(affected + made, cutoff, plan.censor, plan.tpot_ref)
    out = {"choice": choice, "policy_choice": rec["policy_choice"], "executed": rec["executed"],
           "n": int(len(e2e)), "n_unfinished": int((~fin).sum()), "child_wall_s": time.time() - t_wall}
    out.update(rollout_terms(e2e, ttft, plan.tau_e, plan.tau_f))
    return out


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


def evaluate_candidates(h, cands, hz: Horizon, plan: Plan, workers: int, timeout: float = 1800.0, sample=None):
    out = []
    for i in range(0, len(cands), workers):
        kids = [_fork_child(lambda c=c: child_rollout(h, c, hz, plan, sample)) for c in cands[i:i + workers]]
        out += _collect_all(kids, timeout)
    if len({r["policy_choice"] for r in out}) != 1 or len({r["n"] for r in out}) != 1:
        raise RuntimeError(f"rollouts disagree on pi_e's choice or the affected set: "
                           f"{[(r['choice'], r['policy_choice'], r['n']) for r in out]}")
    return out


# ------------------------------------------------------------------ closed loop

def closed_loop(sc, hz: Horizon, plan: Plan, cut, out_dir, log_path, workers=4, future="true",
                max_decisions=None, candidates="all", child_timeout=1800.0, chooser=None, plan_every=1):
    """Run the window with lookahead decisions after ``cut``; write outputs to ``out_dir`` and one JSON line per
    planned decision to ``log_path``. ``chooser(by, pe, plan) -> (best, info)`` replaces ``select_candidate``
    (tests); ``candidates="pe"`` evaluates only pi_e's choice (neutrality check). ``plan_every=K`` plans only every
    K-th decision after the cut (the others keep pi_e's choice): a cost knob that measures lookahead applied at 1/K
    of the decisions, not the per-decision headroom."""
    if hz.rel:
        raise ValueError("resolve a relative time horizon first")
    chooser = chooser or select_candidate
    h = build_simulator(argv_for(sc, out_dir))
    sample_pool = output_pool(sc) if future == "sampled" else None
    cap = ev.max_request_tokens(sc)
    n_planned = n_dev = n_eligible = 0
    decided = {}
    t0 = time.time()
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(log_path, "w") as log:
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
            cands = list(range(sc.replicas)) if candidates == "all" else [None]
            sample = ((1000003 * n + 7), sample_pool, cap) if future == "sampled" else None
            res = evaluate_candidates(h, cands, hz, plan, workers, child_timeout, sample)
            pe = res[0]["policy_choice"]
            by = {(pe if r["choice"] is None else r["choice"]): r for r in res}
            best, info = chooser(by, pe, plan)
            rec = apply_decision(h, best)
            if rec["policy_choice"] != pe or rec["executed"] != best:
                raise RuntimeError(f"parent decision {n + 1}: pi_e chose {rec['policy_choice']} (children: {pe}), "
                                   f"executed {rec['executed']} (wanted {best})")
            n_planned += 1
            n_dev += best != pe
            decided[info["decided_by"]] = decided.get(info["decided_by"], 0) + 1
            cand = {str(c): {k: by[c][k] for k in ("sum", "hinge", "hinge_ttft", "n_unfinished")}
                    | {"score": info["scores"][c]} for c in sorted(by)}
            row = {"k": n + 1, "t": req.arrived_at, "pe": pe, "chosen": best, "decided_by": info["decided_by"],
                   "gain": info.get("gain"), "term_gain": info.get("term_gain"), "cands": cand,
                   "n_aff": res[0]["n"], "child_wall_s": max(r["child_wall_s"] for r in res),
                   "wall": time.time() - t0}
            log.write(json.dumps(row) + "\n")
            log.flush()
    h.write_output()
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_c = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
    return {"planned": n_planned, "deviations": n_dev, "deviation_frac": n_dev / max(n_planned, 1),
            "decided_by": decided, "wall_s": time.time() - t0, "cpu_s_children": cpu_c,
            "cpu_s_per_decision": cpu_c / max(n_planned, 1)}


# ------------------------------------------------------------------ summary with provenance

def _sha(p) -> str:
    return ev.file_sha(Path(p).resolve())[:12]


def _row_key(params: dict) -> str:
    h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return f"lookahead_{params['score']}_{params['horizon']}_{params['future']}_{h}"


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", default="Tool")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=2000)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--model", default="l3_8b")
    ap.add_argument("--horizon", default="T=1x", help="comma-separated: H=<decisions>, T=<s>, T=<f>x (sweep)")
    ap.add_argument("--score", default="tail", choices=("tail", "lex", "hinge", "sum"))
    ap.add_argument("--tail-weight", type=float, help="w of the tail score (default 25 * pi_e's mean E2E)")
    ap.add_argument("--censor", default="elapsed", choices=("elapsed", "remaining"))
    ap.add_argument("--min-gain", type=float, default=0.0, help="score gain a deviation needs (scalar scores)")
    ap.add_argument("--future", default="true", choices=("true", "sampled"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--child-timeout", type=float, default=1800.0, help="seconds per batch of rollout children")
    ap.add_argument("--max-decisions", type=int)
    ap.add_argument("--plan-every", type=int, default=1, help="plan every K-th decision after the warm-up (cost)")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    horizons = [Horizon.parse(s) for s in a.horizon.split(",") if s.strip()]
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
    if not a.skip_baselines:
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
    for hz in horizons:
        hz_abs = hz.resolve(pe["p99_e2e"])
        params = {"horizon": hz.label, "horizon_mode": hz_abs.mode, "horizon_value": hz_abs.value,
                  "score": a.score, "censor": a.censor if hz_abs.mode == "time" else None,
                  "future": a.future, "min_gain": a.min_gain, "max_decisions": a.max_decisions,
                  "plan_every": a.plan_every,
                  "weight": weight if a.score == "tail" else None, "tau_e": plan.tau_e, "tau_f": plan.tau_f,
                  "tpot_ref": plan.tpot_ref if a.censor == "remaining" else None, "cut": pe["cut"]}
        key = _row_key(params)
        if key in rows:
            print(f"{key}: already in {summ_path}, reused")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            info = closed_loop(sc, hz_abs, plan, pe["cut"], tmp, out / f"{key}.jsonl", a.workers, a.future,
                               a.max_decisions, child_timeout=a.child_timeout, plan_every=a.plan_every)
            info.update(score_output(tmp, sc))
        if a.future == "sampled" and sc.window_start < 0:
            info["pool_note"] = "full-trace scenario: output lengths sampled from the whole trace"
        rows[key] = {"params": params, **info}
        save()
        print(json.dumps({key: rows[key]}, indent=1))


if __name__ == "__main__":
    main()
