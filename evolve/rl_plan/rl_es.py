"""RL by evolution strategies (ES): a neural correction to finalist 4873, trained on episode returns (2026-10-14,
Kamil: "do not stop until there is a working RL policy").

Why ES. The PPO residual (``rl_residual.py``) could not learn a planted +1,000 per-override signal in 960 episodes:
a single routing decision's effect on the C7 score is buried under the chaotic response of the rest of the episode
(per-trace advantage SD up to 1e7). A policy-level comparison on identical arrivals (common random numbers) sums the
effect of every changed decision, so its signal grows with the number of decisions while the chaos does not. ES
(Salimans et al., 2017) estimates the policy gradient from such paired episode returns: antithetic parameter
perturbations theta +- sigma eps, each evaluated on the same windows. No per-decision credit assignment is needed and
the objective is the C7 score itself (tails included), not a surrogate.

Policy. 4873 prices every replica (``costs``); the policy adds S * g_theta(x_c) to replica c's price and then lets
4873's own selection (stickiness, ties) choose. g_theta is a small MLP shared across replicas; its output layer
starts at zero, so the initial policy is exactly 4873 (tested). S is the spread of 4873's prices at that call.
x_c holds what a router sees (the views) plus two things 4873 does not use:
- an online output-length predictor: the mean log output length of earlier completed requests with the same first
  prompt block (the system prompt; Tool's output length is largely set by it) and prompt-size bucket, learned from
  ``observe`` events during the episode (never from the trace file);
- per replica, the predicted decode work of its outstanding requests (the sum of their predicted output lengths).

``--probe L`` replaces the MLP by a fixed rule g = L * (relative predicted decode work), a one-parameter check that
the new information is worth anything.

Fitness. Per window, with 4873 run on the same window and P99s over the requests after a warm-up cut:
``tails4`` (default) F = -mean over P99 E2E, TTFT, TPOT and TBT of log(policy / 4873); ``score`` F = -log(S_theta /
S_4873) with the C7 score S (``rl_residual.score_total``). A few random overrides (0.2 % of decisions) move log S by
SD 0.095 per 6,000-request window but the four-tail mean by only 0.020 (``out/noise_probe.json``), so tails4 is the
default. The ES gradient uses antithetic differences, so 4873's score cancels;
it is logged for the learning curve. Training: Tool and Conv windows at random starts, draws p22-p39, loads from
``--configs`` (trace:replicas:load). Learning curves: the centre policy on fixed held-out windows (p40-p43) vs 4873, as in rl_residual.

Usage:
    python -m evolve.rl_plan.rl_es --out evolve/rl_plan/out/es/s0 --workers 8 --pairs 8 --windows 4
    python -m evolve.rl_plan.rl_es --mode probe --probe 3 --out evolve/rl_plan/out/es_probe
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan import headroom as hr
from evolve.rl_plan import rl_residual as rl
from evolve.rl_plan.c0_policy import load_base
from evolve.sim_fork import build_simulator

BASE = c0.BASE
TRACES = rl.TRACES
N_REQ = rl.N_REQ
TRAIN_DRAWS = rl.TRAIN_DRAWS
EVAL_DRAWS = rl.EVAL_DRAWS
LONG_TOKENS = 128  # a "long" output for the count feature
PRIOR_LOG = math.log1p(64.0)  # predictor's value before any completion
PRIOR_N = 2.0  # pseudo-count shrinking a group's mean to its parent's
FEATURES = (
    # request
    "log_pred_out", "pred_conf", "log_prompt", "load", "log_mean_pending",
    # replica c (relative to the cluster where it is a level)
    "cost_rel", "is_min", "d_out", "d_running", "d_log_pending", "d_kv", "cached_frac", "is_session_last",
    "d_log_decode_work", "d_long",
)
F_IN = len(FEATURES)
HIDDEN = 32


# ------------------------------------------------------------------ output-length predictor (online)

def _bucket(n: int) -> int:
    return int(math.log2(max(n, 1)))


class LengthPredictor:
    """Hierarchical running means of log(1 + output tokens): (first block, prompt bucket) -> first block -> all."""

    def __init__(self):
        self.s = {}

    def _get(self, k):
        return self.s.get(k, (0.0, 0))

    def predict(self, h0, n: int):
        sg, ng = self._get(("all",))
        m = (sg + PRIOR_N * PRIOR_LOG) / (ng + PRIOR_N)
        sh, nh = self._get(("h", h0))
        m = (sh + PRIOR_N * m) / (nh + PRIOR_N)
        sb, nb = self._get(("hb", h0, _bucket(n)))
        m = (sb + PRIOR_N * m) / (nb + PRIOR_N)
        return m, nb + 0.5 * nh

    def update(self, h0, n: int, out_tokens: int) -> None:
        y = math.log1p(out_tokens)
        for k in (("all",), ("h", h0), ("hb", h0, _bucket(n))):
            s, c = self._get(k)
            self.s[k] = (s + y, c + 1)


# ------------------------------------------------------------------ policy

# Oracle diagnostic (``--oracle``): true output lengths, keyed by (arrival time, prompt tokens), set by run_episode in
# the simulator's process before the policy routes. Not a deployable policy; it bounds what the predictor could buy.
_ORACLE = None

REPLICA_F = tuple(range(5, F_IN))  # per-replica features (request-level ones shift every replica equally)
N_LIN = 3 * len(REPLICA_F)  # linear policy: each replica feature, and its products with predicted length and load


def init_params(rng: np.random.Generator, arch: str = "mlp") -> dict:
    if arch == "linear":
        return {"lin": np.zeros(N_LIN)}

    def glorot(a, b):
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), (a, b))
    return {"W1": glorot(F_IN, HIDDEN), "b1": np.zeros(HIDDEN), "W2": glorot(HIDDEN, HIDDEN), "b2": np.zeros(HIDDEN),
            "w3": np.zeros(HIDDEN), "b3": np.zeros(1)}


def _keys(P: dict):
    return ("lin",) if "lin" in P else ("W1", "b1", "W2", "b2", "w3", "b3")


def flatten(P: dict) -> np.ndarray:
    return np.concatenate([np.asarray(P[k], float).ravel() for k in _keys(P)])


def unflatten(v: np.ndarray, like: dict) -> dict:
    out, i = {}, 0
    for k in _keys(like):
        n = np.asarray(like[k]).size
        out[k] = v[i:i + n].reshape(np.asarray(like[k]).shape)
        i += n
    return out


def mlp(P: dict, X: np.ndarray) -> np.ndarray:
    if "lin" in P:
        Z = X[:, REPLICA_F]
        return np.hstack([Z, Z * X[:, [0]], Z * X[:, [3]]]) @ P["lin"]
    h = np.tanh(X @ P["W1"] + P["b1"])
    h = np.tanh(h @ P["W2"] + P["b2"])
    return h @ P["w3"] + P["b3"][0]


def features(request, replicas, costs, eff_cached, pred, conf, work, longs) -> np.ndarray:
    R = len(replicas)
    n = max(request.num_prefill_tokens, 1)
    out = np.array([r.outstanding_requests for r in replicas], float)
    run = np.array([r.running for r in replicas], float)
    pend = np.log1p(np.array([r.pending_prefill_tokens for r in replicas], float))
    kv = np.array([r.kv_usage for r in replicas], float)
    cst = np.asarray(costs, float)
    spread = cst.std() + 1.0
    lw = np.log1p(np.asarray(work, float))
    lg = np.asarray(longs, float)
    X = np.empty((R, F_IN))
    X[:, 0] = pred / 5.0
    X[:, 1] = math.log1p(conf) / 5.0
    X[:, 2] = math.log(n) / 10.0
    X[:, 3] = out.mean() / 10.0
    X[:, 4] = math.log1p(np.mean([r.pending_prefill_tokens for r in replicas])) / 10.0
    X[:, 5] = np.clip((cst - cst.min()) / spread, 0.0, 5.0)
    X[:, 6] = cst <= cst.min() + 1e-9
    X[:, 7] = (out - out.mean()) / (out.mean() + 1.0)
    X[:, 8] = (run - run.mean()) / (run.mean() + 1.0)
    X[:, 9] = (pend - pend.mean()) / 5.0
    X[:, 10] = kv - kv.mean()
    X[:, 11] = np.asarray(eff_cached, float) / n
    X[:, 12] = [i == request.session_last_replica for i in range(R)]
    X[:, 13] = (lw - lw.mean()) / 3.0
    X[:, 14] = (lg - lg.mean()) / (lg.mean() + 1.0)
    return X


def build_policy(params_path, probe=None):
    """Returns (choose, observe). ``probe`` = L: g = L * relative predicted decode work (no MLP)."""
    base = load_base(BASE)
    P = None
    if params_path:
        with np.load(params_path) as f:
            P = {k: f[k] for k in f.files}
    bound = float(P["bound"]) if P is not None and "bound" in P else 0.0
    # load gate (2026-10-16): below `gate` mean outstanding requests per replica (router-owned, exact) the policy is
    # plain 4873, which is near-optimal at light load where the learned correction hurt (plan 11, regime map)
    gate = float(P["gate"]) if P is not None and "gate" in P else 0.0
    orig_select = base._select

    def _state(ctx, R):
        st = ctx.state.get("es")
        if st is None:
            st = {"pred": LengthPredictor(), "outs": [dict() for _ in range(R)], "work": [0.0] * R,
                  "long": [0] * R, "calls": 0, "moved": 0}  # moved: pick is not 4873's cheapest price
            ctx.state["es"] = st
        return st

    def _select(request, replicas, ctx, costs, num, n, eff_cached, ramp):
        R = len(replicas)
        st = _state(ctx, R)
        h0 = request.prefix_hashes[0] if request.prefix_hashes else None
        pred, conf = st["pred"].predict(h0, n)
        if _ORACLE is not None:
            pred, conf = math.log1p(_ORACLE[(request.arrived_at, request.num_prefill_tokens)][0]), 1e3
        new = list(costs)
        busy = gate <= 0 or float(np.mean([r.outstanding_requests for r in replicas])) >= gate
        st["gated"] = st.get("gated", 0) + (not busy)
        if busy and (P is not None or probe is not None):
            work = st["work"]
            if probe is not None:
                lw = np.log1p(np.asarray(work, float))
                g = probe * (lw - lw.mean()) / 3.0
            else:
                g = mlp(P, features(request, replicas, costs, eff_cached, pred, conf, work, st["long"]))
                if bound > 0:  # bounded nudge: at most `bound` spreads, whatever the features (2026-10-15)
                    g = bound * np.tanh(np.asarray(g) / bound)
            spread = float(np.std(costs)) + 1.0
            new = [c + spread * float(x) for c, x in zip(costs, g)]
        pick = orig_select(request, replicas, ctx, new, num, n, eff_cached, ramp)
        st["calls"] += 1
        st["moved"] += pick != int(np.argmin(costs))
        if _ORACLE is not None:
            _ORACLE[(request.arrived_at, request.num_prefill_tokens)].pop(0)
        p_tokens = math.expm1(pred)
        st["outs"][pick].setdefault((request.arrived_at, request.num_prefill_tokens), []).append((p_tokens, h0))
        st["work"][pick] += p_tokens
        st["long"][pick] += p_tokens >= LONG_TOKENS
        return pick

    base._select = _select

    def choose(request, replicas, ctx):
        return base.choose(request, replicas, ctx)

    def observe(outcome, ctx):
        if outcome.event != "end":
            return
        st = ctx.state.get("es")
        if st is None:
            return
        q = st["outs"][outcome.replica].get((outcome.arrived_at, outcome.num_prefill_tokens))
        if not q:
            return  # routed before a preemption changed the prompt length: should not happen (prompt is original)
        p_tokens, h0 = q.pop(0)
        st["work"][outcome.replica] -= p_tokens
        st["long"][outcome.replica] -= p_tokens >= LONG_TOKENS
        st["pred"].update(h0, outcome.num_prefill_tokens, outcome.num_decode_tokens)

    return choose, observe


POLICY_TEMPLATE = '''"""ES policy (generated by evolve.rl_plan.rl_es; do not edit)."""
from evolve.rl_plan.rl_es import build_policy as _build

choose, observe = _build({params!r}, {probe!r})
'''


def write_policy(path: Path, params_path, probe=None) -> Path:
    path.write_text(POLICY_TEMPLATE.format(params=str(Path(params_path).resolve()) if params_path else None,
                                           probe=probe))
    return path


# ------------------------------------------------------------------ episodes

def run_episode(job: dict) -> dict:
    """job: trace, start, length, draw, replicas, scale, params (npz path or None = 4873), probe, consts; optional
    write_dir. Returns tail metrics and the C7 score over the requests after the warm-up cut."""
    hr.set_base_policy(BASE)
    sc = ev.Scenario(job["trace"], job["replicas"], job["scale"], job["start"], job["length"], job["draw"],
                     model="l3_8b", jitter_s=job.get("jitter_s", -1.0))
    k = job["consts"].get(job["trace"])
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        if job.get("params") is None and job.get("probe") is None:
            pol = BASE
        else:
            pol = str(write_policy(Path(tmp) / "es_policy.py", job.get("params"), job.get("probe")))
        h = build_simulator(hr.argv_for(sc, job.get("write_dir") or tmp, policy=pol))
        g = h.sim._request_generator
        made = sorted((e._request for e in h.sim._event_queue if getattr(e, "_request", None) is not None
                       and type(e).__name__ == "RequestArrivalEvent"), key=lambda r: r.arrived_at)
        global _ORACLE
        _ORACLE = None
        if job.get("oracle"):
            _ORACLE = {}
            for r in made:
                _ORACLE.setdefault((r.arrived_at, r.num_prefill_tokens), []).append(r.num_decode_tokens)
        orig = g.get_next_request

        def wrapped():
            r = orig()
            if r is not None:
                made.append(r)
                if _ORACLE is not None:
                    _ORACLE.setdefault((r.arrived_at, r.num_prefill_tokens), []).append(r.num_decode_tokens)
            return r
        g.get_next_request = wrapped
        h.run()
        if not h.sim._scheduler.is_empty():
            raise RuntimeError("episode ended with requests left in the scheduler")
        arr = c0.request_arrays(h, made)
        st = h.sim._scheduler._ctx.state.get("es", {})
        _ORACLE = None
        if job.get("write_dir"):
            h.write_output()
    a_ = arr["arrived"]
    cut = float(a_.min() + rl.WARM_FRAC * (a_.max() - a_.min()))
    sub = c0.subset(arr, a_ >= cut)
    return {"trace": job["trace"], "replicas": job["replicas"], "scale": job["scale"], "start": job["start"],
            "draw": job["draw"], "tag": job.get("tag"), "n": len(made), "wall_s": time.time() - t0,
            **rl.tail_metrics(arr, cut),
            "score": rl.score_total(sub, k["taus"], k["w"], k["s_tbt"]) if k else float("nan"),
            "calls": st.get("calls", 0), "moved": st.get("moved", 0)}


def _key(e):
    return (e["trace"], e["replicas"], e["scale"], e["start"], e["draw"])


def _group(e):
    """A request range of one configuration: the bootstrap unit (its draws are resampled together)."""
    return (e["trace"], e["replicas"], e["scale"], e["start"])


SYNTH_LEN = 0  # Synth (3,993 requests) is always run in full, with its arrival jitter
CONFIGS_DEFAULT = "Tool:8:0.75,Conv:8:0.75,Tool:8:1.0,Conv:8:1.0"


def parse_configs(s: str) -> list:
    out = []
    for x in s.split(","):
        t, r, sc = x.split(":")
        out.append((t, int(r), float(sc)))
    return out


def window_job(tr: str, replicas: int, scale: float, start: int, length: int, draw: int, consts: dict) -> dict:
    j = {"trace": tr, "replicas": replicas, "scale": scale, "start": start, "length": length, "draw": draw,
         "consts": consts}
    if tr == "Synth":
        j.update(start=-1, length=SYNTH_LEN, jitter_s=ev.SYNTH_JITTER_S)
    return j


METRICS = ("score", "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e", "mean_ttft", "mean_tpot", "mean_tbt",
           "p50_e2e", "p50_ttft")
TAILS4 = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt")
MEANS4 = ("mean_e2e", "mean_ttft", "mean_tpot", "mean_tbt")
# general latency (Kamil, 2026-10-15): the four means and the four P99s, equal weight
GEN8 = MEANS4 + TAILS4
FIT_METRICS = {"tails4": TAILS4, "gen8": GEN8}


def fitness(e: dict, r: dict, kind: str) -> float:
    if kind == "score":
        return -math.log(e["score"] / r["score"])
    return -float(np.mean([math.log(e[m] / r[m]) for m in FIT_METRICS[kind]]))


def paired_summary(res: list, ref: dict, seed: int = 0) -> dict:
    """Geometric-mean ratio to 4873 per metric, 95 % bootstrap CI over request ranges (trace, start, scale)."""
    groups = sorted({_group(e) for e in res})
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), (2000, len(groups)))
    out = {}
    for m in METRICS:
        if any(not np.isfinite(e[m]) for e in res):
            continue
        L = np.array([np.mean([math.log(e[m] / ref[_key(e)][m]) for e in res if _group(e) == g]) for g in groups])
        bs = L[picks].mean(1)
        out[m] = {"ratio": float(np.exp(L.mean())), "lo": float(np.exp(np.percentile(bs, 2.5))),
                  "hi": float(np.exp(np.percentile(bs, 97.5)))}
    for kind in ("tails4", "gen8"):
        if not all(np.isfinite(e[m]) for e in res for m in FIT_METRICS[kind]):
            continue
        L = np.array([np.mean([-fitness(e, ref[_key(e)], kind) for e in res if _group(e) == g]) for g in groups])
        bs = L[picks].mean(1)
        out[kind] = {"ratio": float(np.exp(L.mean())), "lo": float(np.exp(np.percentile(bs, 2.5))),
                     "hi": float(np.exp(np.percentile(bs, 97.5)))}
    out["moved_rate"] = float(sum(e["moved"] for e in res) / max(1, sum(e["calls"] for e in res)))
    return out


def eval_jobs(params, consts: dict, configs, length: int, probe=None, draws=None, oracle=False) -> list:
    """Fixed windows per configuration: rl_residual's disjoint request ranges (Synth: the full trace) x draws."""
    return [{**window_job(tr, R, sc, s, length, d, consts), "params": params, "probe": probe, "oracle": oracle}
            for (tr, R, sc) in configs for s in rl.EVAL_STARTS.get(tr, (-1,)) for d in (draws or EVAL_DRAWS)]


# ------------------------------------------------------------------ ES driver

def train(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    consts = rl.reward_constants(Path(a.c0_out))
    configs = parse_configs(a.configs)
    rng = np.random.default_rng(a.seed)
    ckpt = out / "ckpt.npz"
    P0 = init_params(np.random.default_rng(a.seed), a.arch)
    pmeta = {"bound": np.array(a.bound)} if a.bound > 0 else {}  # saved with every parameter file
    meta = {"args": vars(a), "features": FEATURES, "hidden": HIDDEN, "dim": int(flatten(P0).size),
            "code_version": ev.code_version()}
    if ckpt.exists():
        with np.load(ckpt, allow_pickle=True) as f:
            theta, m_, v_, it0 = f["theta"], f["m"], f["v"], int(f["iter"]) + 1
            rng = f["rng"].item()
        log_rows = [x for x in (out / "log.jsonl").read_text().splitlines() if json.loads(x)["iter"] < it0] \
            if (out / "log.jsonl").exists() else []
        (out / "log.jsonl").write_text("".join(x + "\n" for x in log_rows))
    else:
        theta = flatten(P0)
        m_, v_, it0 = np.zeros_like(theta), np.zeros_like(theta), 0
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    t_end = time.time() + a.hours * 3600
    ref_eval = {}
    with mp.get_context("forkserver").Pool(a.workers) as pool, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def save(vec, name):
            p = tmp / f"{name}.npz"
            np.savez(p, **unflatten(vec, P0), **pmeta)
            return str(p)

        if a.eval_every:
            ref_eval = {_key(e): e for e in pool.map(run_episode, eval_jobs(None, consts, configs, a.eval_len),
                                                    chunksize=1)}
        for it in range(it0, a.max_iters):
            if time.time() > t_end:
                break
            t_it = time.time()
            wins = []
            for i in range(a.windows):
                tr, R, sc = configs[(it * a.windows + i) % len(configs)]
                st_ = rl.draw_start(rng, tr, a.length, a.start_grid) if tr in N_REQ else -1
                wins.append((tr, R, sc, st_, int(rng.choice(TRAIN_DRAWS))))
            eps = rng.standard_normal((a.pairs, theta.size))
            paths = {"c": save(theta, "c")}
            for i in range(a.pairs):
                paths[f"p{i}"] = save(theta + a.sigma * eps[i], f"p{i}")
                paths[f"m{i}"] = save(theta - a.sigma * eps[i], f"m{i}")
            jobs = []
            for (tr, R, sc, s_, d) in wins:
                base = {**window_job(tr, R, sc, s_, a.length, d, consts), "oracle": a.oracle}
                jobs.append({**base, "params": None, "tag": "ref"})
                for name, p in paths.items():
                    jobs.append({**base, "params": p, "tag": name})
            res = pool.map(run_episode, jobs, chunksize=1)
            ref = {_key(e): e for e in res if e["tag"] == "ref"}
            F = {}
            for e in res:
                if e["tag"] != "ref":
                    F.setdefault(e["tag"], []).append(fitness(e, ref[_key(e)], a.fitness))
            Fm = {t: float(np.mean(v)) for t, v in F.items()}
            diff = np.array([Fm[f"p{i}"] - Fm[f"m{i}"] for i in range(a.pairs)])
            if a.rank:
                allv = np.array([Fm[f"p{i}"] for i in range(a.pairs)] + [Fm[f"m{i}"] for i in range(a.pairs)])
                r = allv.argsort().argsort() / (len(allv) - 1) - 0.5
                diff = r[:a.pairs] - r[a.pairs:]
            grad = (diff[:, None] * eps).sum(0) / (2 * a.pairs * a.sigma)
            if a.opt == "adam":
                m_ = 0.9 * m_ + 0.1 * grad
                v_ = 0.999 * v_ + 0.001 * grad ** 2
                step = a.lr * (m_ / (1 - 0.9 ** (it + 1))) / (np.sqrt(v_ / (1 - 0.999 ** (it + 1))) + 1e-8)
            else:  # SGD with momentum: steps scale with the estimated gradient, so pure noise moves theta less
                m_ = 0.9 * m_ + grad
                step = a.lr * m_
            theta = theta + step - a.lr * a.wd * theta
            pair_sd = float(np.std([Fm[f"p{i}"] - Fm[f"m{i}"] for i in range(a.pairs)]))
            rec = {"iter": it, "wall_s": time.time() - t_it, "windows": wins,
                   "center_F": Fm["c"], "center_ratio": math.exp(-Fm["c"]),
                   "pop_F": {t: round(v, 4) for t, v in Fm.items()},
                   "pop_moved": float(np.mean([e["moved"] / max(1, e["calls"]) for e in res
                                               if e["tag"] not in ("ref", "c")])),
                   "pop_F_mean": float(np.mean([Fm[t] for t in Fm if t != "c"])),
                   "pair_diff_sd": pair_sd, "pair_diff_mean_abs": float(np.mean(np.abs(diff))),
                   "grad_norm": float(np.linalg.norm(grad)), "step_norm": float(np.linalg.norm(step)),
                   "theta_norm": float(np.linalg.norm(theta)),
                   "center_moved": float(np.mean([e["moved"] / max(1, e["calls"]) for e in res if e["tag"] == "c"])),
                   "center_per_window": {":".join(map(str, _key(e))):
                                         round(fitness(e, ref[_key(e)], a.fitness), 4)
                                         for e in res if e["tag"] == "c"}}
            if a.eval_every and (it + 1) % a.eval_every == 0:
                pth = out / f"params_{it + 1}.npz"
                np.savez(pth, **unflatten(theta, P0), **pmeta)
                ge = pool.map(run_episode, eval_jobs(str(pth), consts, configs, a.eval_len, oracle=a.oracle),
                              chunksize=1)
                rec["eval"] = paired_summary(ge, ref_eval)
            np.savez(out / "params_cur.npz", **unflatten(theta, P0), **pmeta)
            np.savez(ckpt, theta=theta, m=m_, v=v_, iter=it, rng=np.array(rng, dtype=object))
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("iter", "wall_s", "center_ratio", "pair_diff_sd", "step_norm")}),
                  flush=True)


def probe(a) -> None:
    """Fixed-rule probe on training draws (``--probe-draws``; the eval draws stay unseen): g = L * relative predicted
    decode work."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    consts = rl.reward_constants(Path(a.c0_out))
    configs = parse_configs(a.configs)
    Ls = [float(x) for x in a.probe.split(",")]
    draws = [int(x) for x in a.probe_draws.split(",")]
    jobs = eval_jobs(None, consts, configs, a.eval_len, draws=draws)
    for L in Ls:
        jobs += eval_jobs(None, consts, configs, a.eval_len, probe=L, draws=draws, oracle=a.oracle)
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        res = pool.map(run_episode, jobs, chunksize=1)
    n = len(jobs) // (len(Ls) + 1)
    ref = {_key(e): e for e in res[:n]}
    summ = {str(L): paired_summary(res[(i + 1) * n:(i + 2) * n], ref) for i, L in enumerate(Ls)}
    (out / "probe.json").write_text(json.dumps({"summary": summ, "runs": res}, indent=1))
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="train", choices=("train", "probe"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-iters", type=int, default=10 ** 6)
    ap.add_argument("--c0-out", default="evolve/rl_plan/out/c0")
    ap.add_argument("--configs", default=CONFIGS_DEFAULT, help="trace:replicas:scale, comma-separated")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--length", type=int, default=4000)
    ap.add_argument("--start-grid", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--opt", default="sgd", choices=("sgd", "adam"))
    ap.add_argument("--arch", default="mlp", choices=("mlp", "linear"))
    ap.add_argument("--bound", type=float, default=0.0, help="g -> bound * tanh(g / bound); 0 = unbounded")
    ap.add_argument("--oracle", action="store_true", help="diagnostic: true output lengths instead of predictions")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-len", type=int, default=rl.EVAL_LEN)
    ap.add_argument("--probe", default="1,3,10")
    ap.add_argument("--probe-draws", default="22,23,24,25")
    ap.add_argument("--fitness", default="tails4", choices=("tails4", "gen8", "score"))
    a = ap.parse_args(argv)
    hr.set_base_policy(BASE)
    (train if a.mode == "train" else probe)(a)


if __name__ == "__main__":
    main()
