"""Step C0: is part of the one-step gain over the base router learnable from the router's view?
(SELECTION amendment 2026-10-13, choices C1-C12; design deck https://claude.ai/artifact/6uQBgTFkhFqsg5JyPbNbe9)

The base router is the frozen finalist 4873 (``BASE``). C0 has five stages, each a CLI mode:

1. ``taus``: the 14 v4t bar routers run on every training window (cached by ``evaluate.ensure_baselines``); a
   window's threshold for metric m is the bar's best window P99 of m there (C7).
2. ``log`` (one window per job): 4873 routes the window. At every ``--every``-th decision after the warm-up cut
   (the first request of a burst, as in ``headroom``) the router's view is recorded (the ``RequestView`` and
   ``ReplicaView``s 4873 is given, captured from its own call in a probe fork) together with the router's arrival
   history, and every replica is scored on the recorded future: 8 forked rollouts, each forcing one replica and
   running the window to completion with arrivals continuing. The parent always follows 4873, so the logged states
   are 4873's own path. Rollout terms per candidate: summed E2E and the four hinges
   sum (x/tau - 1)+ of E2E, TTFT, TPOT over the affected requests and of TBT over their token gaps.
   One log holds every scored decision of a window with its eligible index ``j``; C0 uses j % 6 == 0, C0b
   j % 3 == 0, so C0b resumes C0's log and scores only the new points.
3. ``fit``: items 1 and 2 of the pass mark by leave-one-range-out (a request range's two arrival draws are held out
   together), with the override margin chosen by an inner leave-one-range-out split of the training ranges.
4. ``train``: the final gain model and margin on all ranges; writes the deployable policy file (``c0_policy``).
5. ``item3``: the learned router and 4873 on the 6 heldout3 Tool/Conv 8R 0.75 scenarios, plus 20 noise runs of 4873
   per scenario (arrivals jittered by U(-5, 5) ms), all in process. ``verdict`` combines the three items.

Score of a candidate (C7): S = sum E2E + w (H_e2e + H_ttft + H_tpot + s_tbt H_tbt), w = step B's transferred weight
per trace, s_tbt = the base run's scored requests per scored token gap (so the TBT hinge, summed over tokens, weighs
like one per-request hinge). Gain of replica c at a decision: S(base pick) - S(c), in seconds; positive = better.

Gain model (fixed before any decision is scored): scikit-learn HistGradientBoostingRegressor, squared error,
max_iter 300, learning_rate 0.05, max_leaf_nodes 15, min_samples_leaf 100, l2_regularization 1.0, random_state 0,
no early stopping; trained on every (decision, replica other than the base pick) row with the target clipped to
the training rows' 0.5th-99.5th percentiles. Margin grid: 0, the 50th, 75th, 90th, 95th and 99th percentiles of
the inner-held-out decisions' best predicted gains (those above 0), and +inf (never override); the margin with the
largest total realised gain on the inner held-out decisions wins, ties to the larger margin.

Runbook (Alpine tree with the bundle shipped; ``--stage c0`` or ``c0b``)::

  python -m evolve.rl_plan.c0 --mode taus --stage c0 --workers 10            # one job
  python -m evolve.rl_plan.c0 --mode log --window Tool:0:20 --stage c0 --workers 8 --out evolve/rl_plan/out/c0
  python -m evolve.rl_plan.c0 --mode fit --stage c0 --out evolve/rl_plan/out/c0          # items 1-2
  python -m evolve.rl_plan.c0 --mode train --stage c0 --out evolve/rl_plan/out/c0
  python -m evolve.rl_plan.c0 --mode item3 --stage c0 --workers 10 --out evolve/rl_plan/out/c0
  python -m evolve.rl_plan.c0 --mode verdict --stage c0 --out evolve/rl_plan/out/c0
"""
import argparse
import concurrent.futures as cf_
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sps

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import boot as cluster_boot, clusters
from evolve.rl_plan import c0_features as cf
from evolve.rl_plan import headroom as hr
from evolve.rl_plan.c0_policy import view_dicts
from evolve.sim_fork import after_os_fork, build_simulator
from vidur.scheduler.global_scheduler.evolved_global_scheduler import EvolvedGlobalScheduler

BASE = str((ev.REPO / "evolve/finalists/v4/v4_487363424749.py").resolve())
BAR = "v4t"
WEIGHTS = {"Tool": 1774.199515681546, "Conv": 2336.2813084642503}  # step B's transferred w (s), per trace
DRAWS = (20, 21)
LEN = 2000
STARTS = {"c0": {"Tool": tuple(range(0, 20001, 4000)), "Conv": tuple(range(0, 10001, 2000))},
          "c0b": {"Tool": tuple(range(0, 20001, 2000)), "Conv": tuple(range(0, 10001, 2000))}}
EVERY = {"c0": 6, "c0b": 3}
METRICS4 = ("e2e", "ttft", "tpot", "tbt")
MIN_OVERRIDES = 30
ALPHA_ITEM3 = 0.05  # one-sided, the geometric mean of the four P99 ratios against noise
ALPHA_WORSE = 0.025  # one-sided, a significant worsening of any guarded metric
GUARDED = ("e2e", "ttft", "tpot", "tbt", "mean_e2e", "p50_e2e")
MARGIN_Q = (50, 75, 90, 95, 99)
MODEL_PARAMS = dict(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                    l2_regularization=1.0, early_stopping=False, random_state=0)
CLIP_Q = (0.5, 99.5)
BOOT = 10000


def windows(stage: str) -> List[ev.Scenario]:
    return [ev.Scenario(t, 8, 0.75, s, LEN, p, model="l3_8b") for t in ("Tool", "Conv") for s in STARTS[stage][t]
            for p in DRAWS]


def range_id(tag: str) -> str:
    """'Tool:4000' from a window tag (the request range; its arrival draws share it)."""
    parts = tag.split("-")
    return f"{parts[2]}:{parts[4][1:].split('+')[0]}"


def heldout_c0() -> List[ev.Scenario]:
    return [sc for sc in ev.STAGES["heldout3"] if sc.trace in ("Tool", "Conv") and sc.replicas == 8
            and sc.scale == 0.75]


# ------------------------------------------------------------------ thresholds (the bar on each window)

BAR_SIZE = 14  # BAR_V4's 10 plus tuned Dynamo and tuned Mooncake seeds 0-2 (SELECTION amendment 2026-10-11)


def bar_rows(sc, bundle: Optional[str] = None) -> dict:
    """The bar's runs on ``sc`` from the baseline cache (``--mode taus`` fills it; a log job never runs baselines
    itself, so concurrent jobs do not race on the cache). Refuses a bar of other than 14 routers, a missing run, or
    (``bundle``) a run with another prediction bundle."""
    bar = ev.BARS[BAR]
    if len(bar) != BAR_SIZE:
        raise SystemExit(f"bar {BAR} has {len(bar)} routers here, expected {BAR_SIZE}: install the tuned controls")
    cache = ev.load_baseline_cache()
    rows = {}
    for b in bar:
        r = cache.get(ev.baseline_key(b, sc))
        if not r or not r.get("ok", True):
            raise SystemExit(f"no bar run of {b} on {sc.tag}: run --mode taus first")
        if bundle is not None and r.get("bundle") != bundle:
            raise SystemExit(f"bar run of {b} on {sc.tag} used bundle {r.get('bundle')}, expected {bundle}")
        rows[b] = r
    return rows


def taus_from_cache(sc, bundle: Optional[str] = None) -> dict:
    """tau per metric = the bar's best (lowest) window P99 on ``sc``."""
    rows = bar_rows(sc, bundle)
    return {m: min(float(r[f"p99_{m}"]) for r in rows.values()) for m in METRICS4}


def bar_record(sc) -> dict:
    """Which routers set tau: name, file sha and P99s per router."""
    return {(Path(b[len("policy:"):]).stem if b.startswith("policy:") else b):
            {"key": ev.baseline_key(b, sc).split("|")[-1], **{m: float(r[f"p99_{m}"]) for m in METRICS4}}
            for b, r in bar_rows(sc).items()}


# ------------------------------------------------------------------ the router's view at a decision

_CAP = {"armed": False, "args": None}


def install_capture() -> None:
    """Record the arguments of the next ``choose`` call (the first request of the burst) when armed. Only records:
    the call and its result are unchanged (``test_c0``: the logger's parent run equals a plain run byte for byte)."""
    cls = EvolvedGlobalScheduler
    if getattr(cls._call, "_c0_capture", False):
        return
    orig = cls._call

    def _call(self, fn, *args, **kw):
        if _CAP["armed"] and not kw.get("observe") and fn is getattr(self._policy, "choose", None):
            _CAP["armed"] = False
            _CAP["args"] = args
        return orig(self, fn, *args, **kw)

    _call._c0_capture = True
    cls._call = _call


def capture_view(h) -> tuple:
    """(base pick, request view dict, replica view dicts) at the pending decision, from a probe fork."""
    probe = h.fork()
    _CAP["armed"], _CAP["args"] = True, None
    try:
        rec = hr.apply_decision(probe, None)
    finally:
        _CAP["armed"] = False
    args = _CAP["args"]
    del probe
    if args is None:
        raise RuntimeError("the base router's choose() was not called at this decision")
    req, reps = view_dicts(args[0], args[1])
    return rec["policy_choice"], req, reps


# ------------------------------------------------------------------ rollouts and their terms

def request_arrays(h, reqs) -> dict:
    """Per-request E2E, TTFT, TPOT (requests with > 1 output token) and the token gaps of ``reqs`` after a drained
    run (``reqs`` all completed). TPOT as ``tail_metrics.tpot``; gaps as the replica stores record them for
    ``token_gaps.npz`` (exact TBT)."""
    e2e, ttft, _ = hr.request_costs(reqs, None)
    dec = np.array([r.num_decode_tokens for r in reqs], float)
    m = dec > 1
    tpot = (e2e[m] - ttft[m]) / (dec[m] - 1)
    ids = np.array([r.id for r in reqs], np.int64)
    stores = list(h.sim._cluster_metric_store._replica_metric_stores.values())
    gaps = np.concatenate([np.frombuffer(s._token_gaps, dtype=np.float64) for s in stores]) if stores else np.zeros(0)
    greq = np.concatenate([np.frombuffer(s._token_gap_req, dtype=np.int64) for s in stores]) if stores \
        else np.zeros(0, np.int64)
    keep = np.isin(greq, ids)
    return {"ids": ids, "e2e": e2e, "ttft": ttft, "tpot": tpot, "gaps": gaps[keep], "gap_ids": greq[keep]}


def hinge(x: np.ndarray, tau: float) -> float:
    return float(np.maximum(x / tau - 1.0, 0.0).sum())


def terms4(arr: dict, taus: dict) -> dict:
    """The C7 score's terms, plus sums and counts above tau of every metric (for the deferred mean-only and
    threshold-count reward ablations; they do not enter the C7 score)."""
    x = {"e2e": arr["e2e"], "ttft": arr["ttft"], "tpot": arr["tpot"], "tbt": arr["gaps"]}
    out = {"sum": float(arr["e2e"].sum()), "n": int(len(arr["e2e"])), "n_tpot": int(len(arr["tpot"])),
           "n_gaps": int(len(arr["gaps"]))}
    for m, v in x.items():
        out[f"h_{m}"] = hinge(v, taus[m])
        out[f"sum_{m}"] = float(v.sum())
        out[f"above_{m}"] = int((v > taus[m]).sum())
    return out


def score(t: dict, w: float, s_tbt: float) -> float:
    return t["sum"] + w * (t["h_e2e"] + t["h_ttft"] + t["h_tpot"] + s_tbt * t["h_tbt"])


def c0_child(h, choice: int, taus: dict) -> dict:
    """One candidate (inside an ``os.fork`` child; mutates ``h``): force ``choice``, run the window to completion
    with arrivals continuing, return the terms over the affected set (requests in flight at the decision, the routed
    burst included, plus every later arrival)."""
    after_os_fork(h)
    gc.disable()
    c0 = time.process_time()
    affected = hr._inflight(h)
    g = h.sim._request_generator
    made = []
    orig_next = g.get_next_request

    def wrapped():
        r = orig_next()
        if r is not None:
            made.append(r)
        return r
    g.get_next_request = wrapped
    rec = hr.apply_decision(h, choice)
    h.run()
    if not h.sim._scheduler.is_empty():
        raise RuntimeError("rollout ended with requests left in the scheduler")
    t = terms4(request_arrays(h, affected + made), taus)
    t["cpu_s"] = time.process_time() - c0
    return {"choice": choice, "policy_choice": rec["policy_choice"], "executed": rec["executed"], "t": t}


def c0_sha() -> str:
    """What a log row depends on (log headers carry it; a change orphans the logs). The feature code that turns a
    row's raw view into model inputs runs at fit time and is hashed separately (``feature_sha``)."""
    return hr._src_sha([request_arrays, hinge, terms4, c0_child, capture_view, install_capture, log_window,
                        view_dicts, cf.hist_features, cf.prune_hist, hr.apply_decision, hr._inflight,
                        hr.request_costs, hr.argv_for]) \
        + hashlib.sha256(repr((cf.HIST_WINDOWS, cf.DT_CAP)).encode()).hexdigest()[:4] + ":" + hr.SIM_FORK_SHA


def feature_sha() -> str:
    return hr._src_sha([cf.features, cf._per_replica]) + hashlib.sha256(repr(cf.FEATURES).encode()).hexdigest()[:4]


def log_window(sc, log_path: Path, taus: dict, cut: float, every: int, workers: int, header: dict,
               timeout: float = 3600.0, max_points: Optional[int] = None, write_dir: Optional[Path] = None) -> dict:
    """4873 routes ``sc``; at every ``every``-th eligible decision (arrival >= ``cut``) its view is logged and every
    replica scored (``c0_child``). Rows already in the log are kept (a resume, or C0b after C0: rows are keyed by the
    decision number ``k`` and carry their eligible index ``j``); the arrival time of a logged row is checked.
    ``write_dir``: also write the parent's outputs there (neutrality test)."""
    install_capture()
    logged = hr.read_log(log_path, header)
    R = sc.replicas
    hist: list = []
    j = -1
    n_new = n_points = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, open(log_path, "a") as log:
        h = build_simulator(hr.argv_for(sc, write_dir or tmp, policy=BASE))
        gs = h.sim._scheduler
        while h.next_decision():
            q = gs._request_queue
            t, burst, n = q[0].arrived_at, len(q), h.decisions
            hf = cf.hist_features(hist, t)
            raw_hist = [list(x) for x in hist]  # the router's history before this burst (pruned to 60 s)
            hist.append((t, burst))
            cf.prune_hist(hist, t)
            if t >= cut:
                j += 1
                if j % every == 0 and (max_points is None or n_points < max_points):
                    n_points += 1
                    k = n + 1
                    if k in logged:
                        if logged[k]["t"] != t or logged[k]["j"] != j:
                            raise SystemExit(f"{log_path}: decision {k} was logged at t={logged[k]['t']}, "
                                             f"j={logged[k]['j']}; this run reaches it at t={t}, j={j}")
                    else:
                        c0 = hr._cpu_children()
                        pe, req, reps = capture_view(h)
                        if req["arrived_at"] != t or req["position_in_burst"] != 0 or req["burst_size"] != burst:
                            raise RuntimeError(f"captured view does not match decision {k}")
                        res = hr._run_children([lambda c=c: c0_child(h, c, taus) for c in range(R)], workers,
                                               timeout)
                        if any(r["policy_choice"] != pe or r["executed"] != r["choice"] for r in res) \
                                or len({r["t"]["n"] for r in res}) != 1:
                            raise RuntimeError(f"rollouts at decision {k} disagree: "
                                               f"{[(r['choice'], r['policy_choice'], r['t']['n']) for r in res]}")
                        row = {"k": k, "j": j, "t": t, "pe": pe, "burst": burst, "req": req, "reps": reps,
                               "hist": hf, "hist_raw": raw_hist, "cands": {str(r["choice"]): r["t"] for r in res},
                               "cpu_s": hr._cpu_children() - c0, "wall": time.time() - t0}
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        n_new += 1
            h.run(lambda x: x.decisions > n)
        if write_dir is not None:
            h.write_output()
    return {"points": n_points, "new": n_new, "wall_s": time.time() - t0}


def log_main(sc, out: Path, every: int, workers: int, child_timeout: float = 3600.0,
             max_points: Optional[int] = None) -> dict:
    """One window: the base run (P99s, warm-up cut, bundle), tau from the bar, the summary, then ``log_window``;
    records completion at this spacing when the whole window was walked. One job per window at a time (lock)."""
    import fcntl
    bar_rows(sc)  # refuse early, before the base run, when the bar's runs are missing
    d = window_dir(out, sc)
    d.mkdir(parents=True, exist_ok=True)
    lock = open(d / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{d}: another job is logging this window")
    with tempfile.TemporaryDirectory() as tmp:
        h = build_simulator(hr.argv_for(sc, tmp, policy=BASE))
        bundle = hr.bundle_of(h)
        h.run()
        h.write_output()
        del h
        base = hr.score_output(tmp, sc)
    gc.collect()
    if not base["ok"]:
        raise SystemExit(f"base run failed: {base['reason']}")
    taus = taus_from_cache(sc, bundle)
    meta = _meta(sc, bundle)
    summ_path = d / "summary.json"
    summ = {"meta": meta, "base": base, "taus": taus, "bar": bar_record(sc), "w": WEIGHTS[sc.trace],
            "s_tbt": base["scored"] / base["n_tbt"], "complete": {}}
    header = {**meta, "taus": taus, "cut": base["cut"]}
    hr.read_log(d / "log.jsonl", header)  # refuses a log written with another setup before anything is written
    if summ_path.exists():
        old = json.loads(summ_path.read_text())
        if any(old.get(k) != summ[k] for k in ("meta", "base", "taus", "bar", "w", "s_tbt")):
            raise SystemExit(f"{summ_path}: written with another setup; move it away")
        summ["complete"] = old.get("complete", {})
    hr._atomic_json(summ_path, summ)
    r = log_window(sc, d / "log.jsonl", taus, base["cut"], every, workers, header, child_timeout, max_points)
    if max_points is None:  # the whole window was walked: every point at this spacing is in the log
        summ["complete"][str(every)] = {"points": r["points"], "header": header}
        hr._atomic_json(summ_path, summ)
    return r


# ------------------------------------------------------------------ data set for fitting

def window_dir(out: Path, sc) -> Path:
    return Path(out) / sc.tag


def load_window(d: Path, every: int) -> dict:
    """A window's summary and its rows with j % every == 0."""
    summ = json.loads((d / "summary.json").read_text())
    rows = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()[1:] if x.strip()]
    rows = [r for r in rows if r["j"] % every == 0]
    return {"summary": summ, "rows": sorted(rows, key=lambda r: r["k"])}


def check_windows(dirs: Dict[str, Path], every: int) -> Dict[str, dict]:
    """Every window logged completely at this spacing, with the current code and one setup: the log header equals
    the completion record's and the current ``code_version`` and ``c0_sha``; the logged points at j % every == 0 are
    exactly the recorded count. Refuses otherwise."""
    wins, bad = {}, []
    cv, sha = ev.code_version(), c0_sha()
    setups = set()
    for tag, d in sorted(dirs.items()):
        if not (d / "summary.json").exists():
            bad.append(f"{tag}: not logged")
            continue
        try:
            w = load_window(d, every)
            head = json.loads((d / "log.jsonl").read_text().split("\n", 1)[0])["header"]
        except (OSError, ValueError, KeyError) as e:
            bad.append(f"{tag}: unreadable log or summary ({type(e).__name__}: {e})")
            continue
        done = w["summary"].get("complete", {}).get(str(every))
        if done is None:
            bad.append(f"{tag}: log not complete at every={every}")
        elif done["header"] != head or head["code_version"] != cv or head["decision_sha"] != sha:
            bad.append(f"{tag}: header / code_version / decision_sha differ from the current code")
        elif len(w["rows"]) != done["points"]:
            bad.append(f"{tag}: {len(w['rows'])} rows at every={every}, completion record says {done['points']}")
        setups.add(json.dumps({k: v for k, v in head.items() if k not in ("scenario", "taus", "cut")},
                              sort_keys=True))
        wins[tag] = w
    if len(setups) > 1:
        bad.append(f"{len(setups)} different setups across windows")
    if bad:
        raise SystemExit(f"{len(bad)} window problem(s): " + "; ".join(bad[:5]))
    return wins


def dataset(wins: Dict[str, dict]) -> dict:
    """Decision-level arrays: per decision its range, base pick, feature rows of all R replicas (R x F) and gains
    S(b) - S(c) for all c (0 at b)."""
    X, G, rng, trace, base = [], [], [], [], []
    for tag, w in sorted(wins.items()):
        s = w["summary"]
        wgt, s_tbt = s["w"], s["s_tbt"]
        for r in w["rows"]:
            b = r["pe"]
            R = len(r["reps"])
            sc_ = [score(r["cands"][str(c)], wgt, s_tbt) for c in range(R)]
            X.append(cf.features(r["req"], r["reps"], b, r["hist"]))
            G.append([sc_[b] - sc_[c] for c in range(R)])
            rng.append(range_id(tag))
            trace.append(tag.split("-")[2])
            base.append(b)
    return {"X": np.asarray(X, float), "G": np.asarray(G, float), "range": np.asarray(rng), "trace": np.asarray(trace),
            "base": np.asarray(base, int)}


def _alt_rows(ds: dict, idx: np.ndarray):
    """Training rows (decision, replica != base pick) of decisions ``idx``."""
    R = ds["X"].shape[1]
    mask = np.ones((len(idx), R), bool)
    mask[np.arange(len(idx)), ds["base"][idx]] = False
    return ds["X"][idx][mask], ds["G"][idx][mask]


def fit_model(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingRegressor
    lo, hi = np.percentile(y, CLIP_Q)
    return HistGradientBoostingRegressor(**MODEL_PARAMS).fit(X, np.clip(y, lo, hi))


def predict_best(model, ds: dict, idx: np.ndarray) -> dict:
    """At each decision of ``idx``: the replica other than the base pick with the largest predicted gain, its
    prediction and its recorded gain."""
    R = ds["X"].shape[1]
    P = model.predict(ds["X"][idx].reshape(-1, ds["X"].shape[2])).reshape(len(idx), R)
    P[np.arange(len(idx)), ds["base"][idx]] = -np.inf
    best = P.argmax(axis=1)
    return {"best": best, "pred": P[np.arange(len(idx)), best], "gain": ds["G"][idx][np.arange(len(idx)), best],
            "pred_all": P}


def choose_margin(ds: dict, idx: np.ndarray) -> dict:
    """The margin by inner leave-one-range-out over the ranges of ``idx``: per held-out range a model trained on the
    others predicts; grid = 0, the MARGIN_Q percentiles of the positive best predictions, +inf; the margin with the
    largest total recorded gain wins (ties: the larger)."""
    preds, gains = [], []
    for g in sorted(set(ds["range"][idx])):
        tr, te = idx[ds["range"][idx] != g], idx[ds["range"][idx] == g]
        m = fit_model(*_alt_rows(ds, tr))
        pb = predict_best(m, ds, te)
        preds.append(pb["pred"])
        gains.append(pb["gain"])
    p, y = np.concatenate(preds), np.concatenate(gains)
    pos = p[p > 0]
    grid = sorted({0.0, math.inf} | ({float(np.percentile(pos, q)) for q in MARGIN_Q} if len(pos) else set()))
    total = {m: float(y[p > m].sum()) for m in grid}
    best = max(grid, key=lambda m: (total[m], m))
    return {"margin": best, "grid": grid, "total_gain": [total[m] for m in grid],
            "overrides": [int((p > m).sum()) for m in grid]}


def loro(ds: dict) -> dict:
    """Outer leave-one-range-out: per held-out range, margin by the inner split of the other ranges, model trained
    on all of them, applied to the held-out range's decisions."""
    allidx = np.arange(len(ds["range"]))
    per = []
    for g in sorted(set(ds["range"])):
        tr, te = allidx[ds["range"] != g], allidx[ds["range"] == g]
        mg = choose_margin(ds, tr)
        m = fit_model(*_alt_rows(ds, tr))
        pb = predict_best(m, ds, te)
        alts = _alt_rows(ds, te)
        per.append({"range": g, "idx": te, "margin": mg["margin"], "pred": pb["pred"], "gain": pb["gain"],
                    "override": pb["pred"] > mg["margin"], "best": pb["best"],
                    "alt_pred": m.predict(alts[0]), "alt_gain": alts[1]})
    return {"folds": per}


NO_GAIN = -1e18  # a bootstrap resample without overrides: below every real gain (finite, so percentiles stay finite)


def _spearman0(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(sps.spearmanr(a, b).statistic)


def _ratio(num, den):
    return num / den if den > 0 else float("nan")


def items12(ds: dict, lo: dict, n_boot: int = BOOT, seed: int = 0) -> dict:
    """Items 1 and 2 from the pooled left-out decisions; 95 % CIs from a bootstrap over request ranges."""
    folds = lo["folds"]
    gain_ov = [f["gain"][f["override"]] for f in folds]
    n_ov = int(sum(len(x) for x in gain_ov))
    mean_gain = _ratio(sum(float(x.sum()) for x in gain_ov), n_ov)
    # item 2: within each held-out range (both draws), the Spearman correlation of the best alternative's predicted
    # and recorded gain; its mean over ranges (a range with constant predictions or gains counts as 0). Range-level
    # offsets shared by predictions and gains cannot pass it (review r1 M4); the pooled correlation is reported.
    rho_g = np.array([_spearman0(f["pred"], f["gain"]) for f in folds])
    rho_pooled = _spearman0(np.concatenate([f["pred"] for f in folds]), np.concatenate([f["gain"] for f in folds]))
    rng = np.random.default_rng(seed)
    bm, br = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(folds), len(folds))
        n = sum(len(gain_ov[p]) for p in pick)
        # a resample without overrides shows no gain: it counts as below every finite value (review r1 B1)
        bm[i] = _ratio(sum(float(gain_ov[p].sum()) for p in pick), n) if n > 0 else NO_GAIN
        br[i] = rho_g[pick].mean()
    ci_m = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    ci_r = [float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))]
    item1 = {"overrides": n_ov, "min_overrides": MIN_OVERRIDES, "mean_gain_per_override": mean_gain, "ci95": ci_m,
             "resamples_without_overrides": int((bm == NO_GAIN).sum()),
             "ok": bool(n_ov >= MIN_OVERRIDES and ci_m[0] > 0)}
    item2 = {"spearman_within_range_mean": float(rho_g.mean()), "ci95": ci_r,
             "per_range": {f["range"]: float(r) for f, r in zip(folds, rho_g)}, "spearman_pooled": rho_pooled,
             "ok": bool(ci_r[0] > 0)}
    # diagnostics
    ap = np.concatenate([f["alt_pred"] for f in folds])
    ag = np.concatenate([f["alt_gain"] for f in folds])
    ev_ = 1.0 - float(np.var(ag - ap) / np.var(ag)) if np.var(ag) > 0 else float("nan")
    n_dec = int(sum(len(f["pred"]) for f in folds))
    split = {}
    for tr in ("Tool", "Conv"):
        fs = [f for f in folds if f["range"].startswith(tr)]
        g = np.concatenate([f["gain"][f["override"]] for f in fs]) if fs else np.zeros(0)
        nd = sum(len(f["pred"]) for f in fs)
        split[tr] = {"decisions": nd, "overrides": int(len(g)), "mean_gain_per_override": _ratio(float(g.sum()), len(g)),
                     "total_gain": float(g.sum())}
    diag = {"decisions": n_dec, "override_rate": _ratio(n_ov, n_dec), "agreement_with_base": 1 - _ratio(n_ov, n_dec),
            "explained_variance_left_out": ev_, "margins": {f["range"]: f["margin"] for f in folds},
            "per_range": {f["range"]: {"decisions": int(len(f["pred"])), "overrides": int(f["override"].sum()),
                                       "gain": float(f["gain"][f["override"]].sum())} for f in folds},
            "trace_split": split,
            "oracle_best_gain_mean": float(np.mean(ds["G"].max(axis=1)))}
    return {"item1": item1, "item2": item2, "diagnostics": diag}


# ------------------------------------------------------------------ the final router

class LeastPendingAltModel:
    """A fixed 'gain model' for tests: after a quiet spell (``dt_prev`` >= 1 s since the previous burst) it predicts
    gain 1 for the alternative with the least pending prefill (the first such), else 0 everywhere; with a margin in
    (0, 1) the router then overrides to that replica after every quiet spell. Exercises the features, the arrival
    history and the override booking end to end."""

    def predict(self, X):
        X = np.asarray(X)
        out = np.zeros(len(X))
        if X[0, cf.FEATURES.index("dt_prev")] >= 1.0:
            out[int(np.argmin(X[:, cf.FEATURES.index("log_pending")]))] = 1.0
        return out


def _sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


class FeatureRecorder:
    """A 'gain model' for tests that predicts 0 everywhere and records every feature matrix it is given."""
    seen: list = []

    def predict(self, X):
        FeatureRecorder.seen.append(np.array(X, float))
        return np.zeros(len(X))


def write_policy(out: Path, model, margin: float, provenance: dict, name: str = "c0") -> dict:
    """Pickle the model bundle and write the generated policy file next to it."""
    mp_ = Path(out) / f"{name}_model.pkl"
    with open(mp_, "wb") as f:
        pickle.dump({"model": model, "margin": margin, "features": cf.FEATURES, "provenance": provenance,
                     "base_sha": ev.file_sha(Path(BASE)), "sklearn": _sklearn_version()}, f)
    pol = Path(out) / f"{name}_router.py"
    sha = ev.file_sha(mp_)
    pol.write_text(POLICY_TEMPLATE.format(base=BASE, model=str(mp_.resolve()), model_sha=sha, margin=margin))
    return {"model": str(mp_), "model_sha": sha, "policy": str(pol), "policy_sha": ev.file_sha(pol)}


POLICY_TEMPLATE = '''"""Step C0's learned router (generated by evolve.rl_plan.c0 --mode train; do not edit).
Base router: {base}
Model: {model} (sha256 {model_sha})
Margin: {margin!r}"""
from evolve.rl_plan.c0_policy import build as _build

choose = _build({base!r}, {model!r})
'''


def train_final(ds: dict, out: Path, provenance: dict) -> dict:
    allidx = np.arange(len(ds["range"]))
    mg = choose_margin(ds, allidx)
    model = fit_model(*_alt_rows(ds, allidx))
    return write_policy(out, model, mg["margin"], provenance) | mg


# ------------------------------------------------------------------ item 3: the learned router routing

def _run_in_process(job) -> dict:
    """(kind, scenario, policy, seed): 'run' = one run of ``policy``; 'noise' = 4873 with every arrival jittered by
    U(-delta, delta) (``headroom.noise_run``)."""
    kind, sc, policy, seed = job
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "noise":
            hr.set_base_policy(BASE)
            hr.noise_run(sc, seed, 0.0, tmp, hr.NOISE_DELTA)
            stats = None
        else:
            h = build_simulator(hr.argv_for(sc, tmp, policy=policy))
            h.run()
            h.write_output()
            stats = dict(h.sim._scheduler._ctx.state.get("c0_stats", {}))
        r = hr.score_output(tmp, sc)
    r["c0_stats"] = stats
    return r


def logs_vs(r: dict, b: dict) -> dict:
    out = {f"log_p99_{m}": math.log(r[f"p99_{m}"] / b[f"p99_{m}"]) for m in METRICS4}
    out["log_p99_mean_e2e"] = math.log(r["mean_e2e"] / b["mean_e2e"])
    out["log_p99_p50_e2e"] = math.log(r["p50_e2e"] / b["p50_e2e"])
    out["log_p99_geo4"] = 0.25 * sum(out[f"log_p99_{m}"] for m in METRICS4)
    return out


def item3_verdict(res: dict) -> dict:
    """``res``: tag -> {"new", "base", "noise": [20 runs]}. L = log ratios new / base; reference = the 380 ordered
    pairwise differences of the 20 noise runs (``headroom.NOISE_CENTRE``); ``headroom.pooled_test``."""
    import itertools
    if hr.NOISE_CENTRE != "pairwise":
        raise RuntimeError("item 3 is pre-registered with the pairwise noise reference")
    want = sorted(sc.tag for sc in heldout_c0())
    if sorted(res) != want:
        return {"ok": None, "status": "invalid_scenarios", "got": sorted(res), "expected": want}
    wins, per = [], {}
    keys = ["geo4"] + list(GUARDED)
    for tag, x in sorted(res.items()):
        seeds = [r.get("seed") for r in x.get("noise", [])]
        if sorted(seeds) != list(range(20)) or "new" not in x or "base" not in x:
            return {"ok": None, "status": "incomplete", "missing": tag}
        L = logs_vs(x["new"], x["base"])
        N = [logs_vs(a, b) for a, b in itertools.permutations(x["noise"], 2)]
        wins.append((L, N))
        per[tag] = {k: L[f"log_p99_{k}"] for k in keys} | {"c0_stats": x["new"].get("c0_stats")}
    pt = hr.pooled_test(wins, metrics=tuple(keys))
    if any("error" in v for v in pt.values()):
        return {"ok": None, "status": "not_computable", "pooled": pt, "per_scenario": per}
    gm = float(np.mean([L["log_p99_geo4"] for L, _ in wins]))
    tags = sorted(res)
    groups = clusters(tags)
    ci = {}
    for k in keys:
        d = np.array([per[t][k] for t in tags])
        g, lo_, hi_ = cluster_boot(d, groups)
        ci[k] = {"geo": float(g), "lo": float(lo_), "hi": float(hi_)}
    worse = {k: {"p_worse": pt[k]["p_high"], "ok": bool(pt[k]["p_high"] >= ALPHA_WORSE)} for k in GUARDED}
    better = {"geo_ratio": math.exp(gm), "p": pt["geo4"]["p_low"],
              "ok": bool(gm < 0 and pt["geo4"]["p_low"] < ALPHA_ITEM3)}
    return {"ok": bool(better["ok"] and all(v["ok"] for v in worse.values())), "better": better,
            "not_worse": worse, "per_scenario": per, "pooled": pt,
            "cluster_ci_reported": {"note": f"{len(groups)} configurations; reported, not a clause", **ci}}


# ------------------------------------------------------------------ CLI

def _meta(sc, bundle: str) -> dict:
    return {"scenario": sc.tag, "base": Path(BASE).name, "base_sha": ev.file_sha(Path(BASE))[:12],
            "code_version": ev.code_version(), "bundle": bundle, "decision_sha": c0_sha()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=("taus", "log", "fit", "train", "item3", "verdict", "report"))
    ap.add_argument("--stage", default="c0", choices=sorted(STARTS))
    ap.add_argument("--window", help="log: Trace:start:draw, e.g. Tool:4000:20")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--child-timeout", type=float, default=3600.0)
    ap.add_argument("--max-points", type=int, help="log: at most this many scored decisions (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hr.set_base_policy(BASE)
    wins = windows(a.stage)
    if a.mode == "taus":
        ev.ensure_baselines(wins, ev.BARS[BAR], workers=a.workers)
        taus = {sc.tag: taus_from_cache(sc) for sc in wins}
        hr._atomic_json(out / f"taus_{a.stage}.json", {"bar": BAR, "taus": taus, "code_version": ev.code_version()})
        print(json.dumps({"windows": len(taus)}))
        return
    if a.mode == "log":
        if not a.window:
            raise SystemExit("--mode log needs --window Trace:start:draw")
        t, s_, p = a.window.split(":")
        sc = ev.Scenario(t, 8, 0.75, int(s_), LEN, int(p), model="l3_8b")
        if sc.tag not in {w.tag for w in wins}:
            raise SystemExit(f"{sc.tag} is not a {a.stage} window")
        print(json.dumps(log_main(sc, out, EVERY[a.stage], a.workers, a.child_timeout, a.max_points)))
        return
    dirs = {sc.tag: window_dir(out, sc) for sc in wins}
    if a.mode in ("fit", "train"):
        ds = dataset(check_windows(dirs, EVERY[a.stage]))
        prov = {"stage": a.stage, "every": EVERY[a.stage], "decisions": int(len(ds["range"])),
                "ranges": sorted(set(ds["range"])), "model": MODEL_PARAMS, "clip_q": CLIP_Q, "margin_q": MARGIN_Q,
                "c0_sha": c0_sha(), "feature_sha": feature_sha(), "code_version": ev.code_version()}
        if a.mode == "fit":
            res = items12(ds, loro(ds)) | {"provenance": prov}
            hr._atomic_json(out / f"items12_{a.stage}.json", res)
        else:
            res = train_final(ds, out, prov) | {"provenance": prov}
            hr._atomic_json(out / f"train_{a.stage}.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "diagnostics"}, indent=1, default=str))
        return
    if a.mode == "item3":
        tr = json.loads((out / f"train_{a.stage}.json").read_text())
        if ev.file_sha(Path(tr["policy"])) != tr["policy_sha"] or ev.file_sha(Path(tr["model"])) != tr["model_sha"]:
            raise SystemExit("the trained router's files changed since --mode train")
        path = out / f"item3_{a.stage}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        if res and res.get("_policy_sha") != tr["policy_sha"]:
            raise SystemExit(f"{path}: from another trained router; move it away")
        meta3 = {"code_version": ev.code_version(), "base_sha": ev.file_sha(Path(BASE))[:12],
                 "noise_sha": hr.decision_sha("noise"), "noise_delta": hr.NOISE_DELTA}
        if res and res.get("_meta") != meta3:
            raise SystemExit(f"{path}: written with another code version, base or noise rule; move it away")
        res["_policy_sha"], res["_meta"] = tr["policy_sha"], meta3
        jobs = []
        for sc in heldout_c0():
            x = res.setdefault(sc.tag, {"noise": []})
            if "new" not in x:
                jobs.append(("run", sc, tr["policy"], None, "new"))
            if "base" not in x:
                jobs.append(("run", sc, BASE, None, "base"))
            have = {r["seed"] for r in x["noise"]}
            jobs += [("noise", sc, None, s, "noise") for s in range(20) if s not in have]
        with cf_.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) as ex:
            futs = {ex.submit(_run_in_process, j[:4]): j for j in jobs}
            for fu in cf_.as_completed(futs):
                kind, sc, _, seed, slot = futs[fu]
                r = fu.result()
                if not r.get("ok"):
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(f"{slot} run on {sc.tag} failed: {r.get('reason')}")
                x = res[sc.tag]
                if slot == "noise":
                    x["noise"].append(r | {"seed": seed})
                else:
                    x[slot] = r
                hr._atomic_json(path, res)
        v = item3_verdict({k: v for k, v in res.items() if not k.startswith("_")}) | {"policy_sha": tr["policy_sha"]}
        hr._atomic_json(out / f"item3_verdict_{a.stage}.json", v)
        print(json.dumps({k: v[k] for k in ("ok", "better", "not_worse") if k in v}, indent=1))
        return
    if a.mode == "report":  # reported, never a clause: the learned router on all 27 heldout3 scenarios (policy_table)
        tr = json.loads((out / f"train_{a.stage}.json").read_text())
        scen = list(ev.STAGES["heldout3"])
        ev.ensure_baselines(scen, tuple(ev.BARS[BAR]) + (f"policy:{BASE}", f"policy:{Path(tr['policy']).resolve()}"),
                            workers=a.workers)
        from evolve.analysis import policy_table
        policy_table.main(["--stage", "heldout3", "--bar", BAR, "--policy", tr["policy"], "--policy", BASE,
                           "--out", str(out / f"policy_table_heldout3_{a.stage}.json")])
        return
    if a.mode == "verdict":
        i12 = json.loads((out / f"items12_{a.stage}.json").read_text())
        p3 = out / f"item3_verdict_{a.stage}.json"
        i3 = json.loads(p3.read_text()) if p3.exists() else {"ok": None, "status": "not run"}
        tr = json.loads((out / f"train_{a.stage}.json").read_text()) if (out / f"train_{a.stage}.json").exists() \
            else {}
        if p3.exists() and i3.get("policy_sha") != tr.get("policy_sha"):
            raise SystemExit(f"item 3 was run with another trained router than train_{a.stage}.json's")
        if p3.exists() and tr.get("provenance") != i12.get("provenance"):
            raise SystemExit("items 1-2 and the trained router come from different data or code")
        ok12 = i12["item1"]["ok"] and i12["item2"]["ok"]
        if not ok12:
            nxt = "C0b (SELECTION 2026-10-13)" if a.stage == "c0" else "stop C; Kamil decides"
        elif i3.get("ok") is None:
            nxt = f"item 3: {i3.get('status', 'not run')}"
        else:
            nxt = "pass: full C needs Kamil's budget approval" if i3["ok"] else \
                "items 1-2 pass, item 3 fails: report; Kamil decides (more data or a shorter horizon)"
        v = {"stage": a.stage, "item1": i12["item1"], "item2": i12["item2"], "item3": {k: i3.get(k) for k in
             ("ok", "better", "not_worse", "status")}, "pass": bool(ok12 and i3.get("ok")), "next": nxt}
        if a.stage == "c0b":
            v["two_looks"] = ("C0b is the second look after C0 failed item 1 or 2: the false-pass rate of each of items "
                              "1-2 across both looks is at most 5 % one-sided (2.5 % for one look)")
        hr._atomic_json(out / f"verdict_{a.stage}.json", v)
        print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
