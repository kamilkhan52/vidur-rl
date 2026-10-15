"""Large-scale model-free RL, residual on finalist 4873 (Kamil, 2026-10-14: an agent routes every request and learns
from rewards with lots of data; start exactly at 4873; pilot of about 500 CPU-h with learning curves). Version 2,
after review r1 (``reviews/2026-10-14_rl_residual/r1.md``).

Policy. At every routing call the base router 4873 picks b. Each replica c gets the logit
    z_c = f_theta(x_c) + ALPHA * [c == b]
where x_c are C0's 29 features of c relative to b (``c0_features``) plus an is-base flag, normalised and clipped, and
f_theta is a small MLP shared across replicas whose last layer starts at zero. pi_theta = softmax(z); the deployed
policy is its argmax (ties go to b), which is exactly 4873 while f = 0 (tested). ALPHA is fixed. Training samples from
the behaviour policy mu = (1 - eps) pi_theta + eps U, with U uniform over the replicas other than b: a fixed
exploration floor, so learning cannot collapse onto 4873 (review M2). PPO optimises mu (its ratio is mu_new / mu_old).
An override is booked as C0's router books it (4873's attended-context FIFO entry moves to the chosen replica).

Reward. The C7 score of an episode, S = sum E2E + w (H_e2e + H_ttft + H_tpot + s_tbt H_tbt) with hinges
H = sum (x / tau - 1)+, is split into the cost incurred between consecutive routing decisions: E2E time and the E2E
and TTFT hinges accrue continuously over each request's life (a hinge from age tau on), the TPOT and TBT hinges at the
request's completion. The rewards of an episode sum to -S exactly (tested). tau and s_tbt are per-trace constants
(medians over C0's windows), w is C0's per-trace weight.

Returns and baseline. Discounted over simulated time, exp(-dt / T_GAMMA). T_GAMMA must cover the hinges, which start
at ages of 20-140 s (review B1); episodes are long windows (``length`` requests) and only decisions after the warm-up
(first WARM_FRAC of the arrival span) and at least TAIL_MULT * T_GAMMA before the last arrival enter the loss (review
M1). Every window is run ``rollouts`` times with the same arrivals and different action samples; the return from
decision k is compared with the mean return of the other rollouts at the same decision (the same request): an
input-driven baseline (Mao et al., ICLR 2019). Advantages are normalised per trace.

Signal checks (review B2). Every iteration logs, for several T_GAMMA, the mean advantage of override actions and its
standard error (the average harm of exploring should be visible). ``--plant K`` adds a planted reward of K score units
to every decision that routes to the non-base replica with the fewest outstanding requests; the planted share (how
much of pi_theta's non-base mass that replica gets, 1/(R-1) at the start) shows whether PPO can learn a signal of that
size from this data.

Data. Training: Tool and Conv, 8 replicas, load 0.75, windows at random starts, arrival draws p22-p39 (never used
elsewhere). Learning curves: the greedy policy against 4873 on 6 disjoint request ranges x draws p40-p43 (never trained
on), P99s after a 20 % warm-up cut, with a placebo (4873 with uniformly random overrides at the greedy policy's
override rate) and a bootstrap over request ranges. Validation (heldout3 p2, p3, p5) and test (test3b) are untouched.

Usage (Alpine):
    python -m evolve.rl_plan.rl_residual --out evolve/rl_plan/out/rl/s0 --seed 0 --workers 32 --hours 5
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan import c0_features as cf
from evolve.rl_plan import headroom as hr
from evolve.rl_plan.c0_policy import load_base, move_fifo, view_dicts
from evolve.sim_fork import build_simulator

BASE = c0.BASE
TRACES = ("Tool", "Conv")
N_REQ = {"Tool": 23608, "Conv": 12031}
TRAIN_DRAWS = tuple(range(22, 40))
EVAL_DRAWS = (40, 41, 42, 43)
EVAL_LEN = 4000
EVAL_STARTS = {"Tool": (0, 8000, 16000), "Conv": (0, 4000, 8000)}  # disjoint request ranges
SCEN = {"replicas": 8, "scale": 0.75}  # tests shrink it
F_IN = len(cf.FEATURES) + 1  # + is_base
HIDDEN = 64
ALPHA = 8.0  # pi_theta keeps b with probability 0.9977 at f = 0 (8 replicas); exploration comes from eps
Z_CLIP = 10.0
WARM_FRAC = 0.2
TAIL_MULT = 2.0
SNR_T = (20.0, 60.0, 120.0, 240.0, math.inf)
TRAJ_KEY = "rl_traj"
HIST_KEY = "rl_hist"
COUNT_KEY = "rl_counts"  # [routing calls, overrides]

DEFAULTS = {"windows_per_iter": 8, "rollouts": 4, "length": 6000, "t_gamma": 120.0, "eps": 0.02, "lr": 3e-4,
            "clip": 0.2, "epochs": 8, "minibatch": 8192, "target_kl": 0.01, "eval_every": 10, "max_grad_norm": 0.5,
            "plant": 0.0, "start_grid": 0}
I_OUT = cf.FEATURES.index("outstanding")


# ------------------------------------------------------------------ policy (numpy, used inside the simulator)

def init_params(rng: np.random.Generator, mu: np.ndarray, sd: np.ndarray) -> dict:
    def glorot(a, b):
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), (a, b)).astype(np.float64)
    return {"W1": glorot(F_IN, HIDDEN), "b1": np.zeros(HIDDEN), "W2": glorot(HIDDEN, HIDDEN), "b2": np.zeros(HIDDEN),
            "w3": np.zeros(HIDDEN), "b3": np.zeros(1), "mu": mu, "sd": sd}


def rows_in(rows, b: int) -> np.ndarray:
    X = np.asarray(rows, float)
    isb = np.zeros((len(X), 1))
    isb[b, 0] = 1.0
    return np.hstack([X, isb])


def logits(P: dict, X: np.ndarray, b: int) -> np.ndarray:
    """X: R x F_IN raw inputs (``rows_in``)."""
    Z = np.clip((X - P["mu"]) / P["sd"], -Z_CLIP, Z_CLIP)
    h = np.tanh(Z @ P["W1"] + P["b1"])
    h = np.tanh(h @ P["W2"] + P["b2"])
    z = h @ P["w3"] + P["b3"][0]
    z[b] += ALPHA
    return z


def behaviour(z: np.ndarray, b: int, eps: float) -> np.ndarray:
    p = np.exp(z - z.max())
    p /= p.sum()
    u = np.full(len(z), 1.0 / (len(z) - 1))
    u[b] = 0.0
    return (1.0 - eps) * p + eps * u


def build_policy(params_path: str, mode: str, seed: int, record: bool = True, eps: float = 0.0):
    """``mode``: 'sample' (training: from mu), 'greedy' (argmax pi_theta, ties to 4873's pick), or 'placebo'
    (4873, overriding to a uniformly random other replica with probability ``eps``; ``params_path`` unused)."""
    base = load_base(BASE)
    P = None
    if mode != "placebo":
        with np.load(params_path) as f:
            P = {k: f[k] for k in f.files}
    rng = np.random.default_rng(seed)

    def choose(request, replicas, ctx):
        b = base.choose(request, replicas, ctx)
        R = len(replicas)
        lp, X = 0.0, None
        if mode == "placebo":
            a = b
            if rng.random() < eps:
                a = int(rng.choice([c for c in range(R) if c != b]))
        else:
            hist = ctx.state.setdefault(HIST_KEY, [])
            req, reps = view_dicts(request, replicas)
            X = rows_in(cf.features(req, reps, b, cf.hist_features(hist, ctx.now)), b)
            if request.position_in_burst == 0:
                hist.append((ctx.now, request.burst_size))
                cf.prune_hist(hist, ctx.now)
            z = logits(P, X, b)
            if mode == "greedy":
                a = b if z[b] >= z.max() else int(np.argmax(z))
            else:
                m = behaviour(z, b, eps)
                a = int(rng.choice(R, p=m))
                lp = float(math.log(m[a]))
        cnt = ctx.state.setdefault(COUNT_KEY, [0, 0])
        cnt[0] += 1
        cnt[1] += a != b
        if record and X is not None:
            ctx.state.setdefault(TRAJ_KEY, []).append((X.astype(np.float32), b, a, lp, ctx.now,
                                                       request.num_prefill_tokens))
        if a != b:
            move_fifo(ctx.state, request.num_prefill_tokens, b, a)
        return a

    return choose


POLICY_TEMPLATE = '''"""RL residual policy (generated by evolve.rl_plan.rl_residual; do not edit)."""
from evolve.rl_plan.rl_residual import build_policy as _build

choose = _build({params!r}, {mode!r}, {seed!r}, {record!r}, {eps!r})
'''


def write_policy(path: Path, params_path, mode: str, seed: int, record: bool = True, eps: float = 0.0) -> Path:
    path.write_text(POLICY_TEMPLATE.format(params=str(Path(params_path).resolve()) if params_path else None,
                                           mode=mode, seed=seed, record=record, eps=float(eps)))
    return path


def save_params(P: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **P)
    os.replace(tmp, path)


# ------------------------------------------------------------------ rewards

def _overlap_cum(s: np.ndarray, e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F(t) = sum_i |[s_i, e_i) intersect [-inf, t)| for each t (vectorised; intervals with e <= s add 0)."""
    keep = e > s
    s, e = np.sort(s[keep]), np.sort(e[keep])
    cs, ce = np.concatenate([[0.0], np.cumsum(s)]), np.concatenate([[0.0], np.cumsum(e)])
    ns, ne = np.searchsorted(s, t, "left"), np.searchsorted(e, t, "left")
    return (ns * t - cs[ns]) - (ne * t - ce[ne])


def _point_cum(c: np.ndarray, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """sum of p_i over c_i < t."""
    o = np.argsort(c)
    cs = np.concatenate([[0.0], np.cumsum(p[o])])
    return cs[np.searchsorted(c[o], t, "left")]


def request_costs(arr: dict, taus: dict, w: float, s_tbt: float) -> dict:
    """Per request: arrival a, first token f, completion c, and the point cost at completion (TPOT and TBT hinges)."""
    a = arr["arrived"]
    c_ = a + arr["e2e"]
    f_ = a + arr["ttft"]
    pt = np.zeros(len(a))
    pt[arr["tpot_idx"]] += np.maximum(arr["tpot"] / taus["tpot"] - 1.0, 0.0)
    idx = {int(i): k for k, i in enumerate(arr["ids"])}
    gh = np.maximum(arr["gaps"] / taus["tbt"] - 1.0, 0.0)
    if len(gh):
        np.add.at(pt, np.array([idx[int(i)] for i in arr["gap_ids"]]), s_tbt * gh)
    return {"a": a, "f": f_, "c": c_, "point": w * pt}


def cum_cost(rc: dict, taus: dict, w: float, t: np.ndarray) -> np.ndarray:
    """C(t): cost incurred before time t (E2E time, E2E and TTFT hinges accrued; point costs completed)."""
    a, f_, c_ = rc["a"], rc["f"], rc["c"]
    out = _overlap_cum(a, c_, t)
    out += (w / taus["e2e"]) * _overlap_cum(a + taus["e2e"], c_, t)
    out += (w / taus["ttft"]) * _overlap_cum(a + taus["ttft"], f_, t)
    out += _point_cum(c_, rc["point"], t)
    return out


def score_total(arr: dict, taus: dict, w: float, s_tbt: float) -> float:
    """The C7 score over all requests (``c0.terms4`` / ``c0.score``)."""
    return c0.score(c0.terms4(arr, taus), w, s_tbt)


def decision_rewards(rc: dict, taus: dict, w: float, t_dec: np.ndarray) -> np.ndarray:
    """r_k = -(C(t_{k+1}) - C(t_k)); the last decision gets everything after it, so sum r = -S."""
    C = cum_cost(rc, taus, w, np.asarray(t_dec, float))
    end = float(np.max(rc["c"])) + 1.0
    C_end = cum_cost(rc, taus, w, np.array([end]))[0]
    nxt = np.append(C[1:], C_end)
    return -(nxt - C)


def discounted_returns(r: np.ndarray, t: np.ndarray, t_gamma: float) -> np.ndarray:
    G = np.empty(len(r))
    acc = 0.0
    for k in range(len(r) - 1, -1, -1):
        dt = (t[k + 1] - t[k]) if k + 1 < len(r) else 0.0
        acc = r[k] + (math.exp(-dt / t_gamma) if math.isfinite(t_gamma) else 1.0) * acc
        G[k] = acc
    return G


def loss_mask(t: np.ndarray, t_gamma: float) -> np.ndarray:
    """Decisions that enter the loss: after the warm-up and at least TAIL_MULT * T_GAMMA before the last arrival
    (at most half of the arrival span is cut at the end)."""
    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    tail = min(TAIL_MULT * t_gamma, 0.5 * span) if math.isfinite(t_gamma) else 0.5 * span
    return (t >= t0 + WARM_FRAC * span) & (t <= t1 - tail)


def planted_target(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per decision: the non-base replica with the fewest outstanding requests (the first on ties)."""
    out = X[:, :, I_OUT].astype(np.float64).copy()
    out[np.arange(len(b)), b] = np.inf
    return out.argmin(axis=1)


# ------------------------------------------------------------------ episodes (in pool workers)

def tail_metrics(arr: dict, cut: float) -> dict:
    """P99s etc. over the requests arriving at or after ``cut`` (the warm-up cut)."""
    m = arr["arrived"] >= cut
    sub = c0.subset(arr, m)
    e2e = sub["e2e"]
    return {"p99_e2e": float(np.percentile(e2e, 99)), "p99_ttft": float(np.percentile(sub["ttft"], 99)),
            "p99_tpot": float(np.percentile(sub["tpot"], 99)) if len(sub["tpot"]) else float("nan"),
            "p99_tbt": float(np.percentile(sub["gaps"], 99)) if len(sub["gaps"]) else float("nan"),
            "mean_e2e": float(e2e.mean()), "p50_e2e": float(np.median(e2e))}


def run_episode(job: dict) -> dict:
    """job: trace, start, length, draw, params, mode, seed, eps, record, consts, replicas, scale (``SCEN``);
    optionally write_dir (the simulator's outputs are written there)."""
    hr.set_base_policy(BASE)
    sc = ev.Scenario(job["trace"], job["replicas"], job["scale"], job["start"], job["length"], job["draw"],
                     model="l3_8b")
    k = job["consts"][job["trace"]]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        pol = write_policy(Path(tmp) / "rl_policy.py", job["params"], job["mode"], job["seed"], job["record"],
                           job.get("eps", 0.0))
        h = build_simulator(hr.argv_for(sc, job.get("write_dir") or tmp, policy=str(pol)))
        g = h.sim._request_generator
        # requests generated while the simulator was built sit in arrival events
        made = sorted((e._request for e in h.sim._event_queue if getattr(e, "_request", None) is not None
                       and type(e).__name__ == "RequestArrivalEvent"), key=lambda r: r.arrived_at)
        if len(made) != g.num_requests_generated:
            raise RuntimeError("could not find every request generated at build time")
        orig = g.get_next_request

        def wrapped():
            r = orig()
            if r is not None:
                made.append(r)
            return r
        g.get_next_request = wrapped
        h.run()
        if not h.sim._scheduler.is_empty():
            raise RuntimeError("episode ended with requests left in the scheduler")
        arr = c0.request_arrays(h, made)
        state = h.sim._scheduler._ctx.state
        traj = state.get(TRAJ_KEY, [])
        calls, over = state.get(COUNT_KEY, [0, 0])
        if job.get("write_dir"):
            h.write_output()
    a_ = arr["arrived"]
    cut = float(a_.min() + WARM_FRAC * (a_.max() - a_.min()))
    out = {"trace": job["trace"], "start": job["start"], "draw": job["draw"], "seed": job["seed"],
           "mode": job["mode"], "n": len(made), "wall_s": time.time() - t0, **tail_metrics(arr, cut),
           "score": score_total(arr, k["taus"], k["w"], k["s_tbt"]), "calls": calls, "overrides": over}
    if job["record"] and job["mode"] != "placebo":
        if len(traj) != len(made):
            raise RuntimeError(f"{len(traj)} routing calls for {len(made)} requests")
        t_dec = np.array([x[4] for x in traj])
        if not np.allclose(t_dec, a_) or [x[5] for x in traj] != [r.num_prefill_tokens for r in made]:
            raise RuntimeError("routing calls do not line up with the requests in arrival order")
        rc = request_costs(arr, k["taus"], k["w"], k["s_tbt"])
        out.update({"X": np.stack([x[0] for x in traj]), "b": np.array([x[1] for x in traj], np.int16),
                    "a": np.array([x[2] for x in traj], np.int16), "logp": np.array([x[3] for x in traj]),
                    "t": t_dec, "r": decision_rewards(rc, k["taus"], k["w"], t_dec)})
    return out


# ------------------------------------------------------------------ learner (torch)

def _torch_net(P: dict):
    import torch
    net = torch.nn.Sequential(torch.nn.Linear(F_IN, HIDDEN), torch.nn.Tanh(), torch.nn.Linear(HIDDEN, HIDDEN),
                              torch.nn.Tanh(), torch.nn.Linear(HIDDEN, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(P["W1"].T))
        net[0].bias.copy_(torch.from_numpy(P["b1"]))
        net[2].weight.copy_(torch.from_numpy(P["W2"].T))
        net[2].bias.copy_(torch.from_numpy(P["b2"]))
        net[4].weight.copy_(torch.from_numpy(P["w3"][None, :]))
        net[4].bias.copy_(torch.from_numpy(P["b3"]))
    return net


def _params_from(net, mu, sd) -> dict:
    return {"W1": net[0].weight.detach().numpy().T.copy(), "b1": net[0].bias.detach().numpy().copy(),
            "W2": net[2].weight.detach().numpy().T.copy(), "b2": net[2].bias.detach().numpy().copy(),
            "w3": net[4].weight.detach().numpy()[0].copy(), "b3": net[4].bias.detach().numpy().copy(),
            "mu": mu, "sd": sd}


def normalise(X: np.ndarray, mu, sd) -> np.ndarray:
    return np.clip((X.astype(np.float64) - mu) / sd, -Z_CLIP, Z_CLIP)


def torch_logits(net, Xn, B):
    """Xn: N x R x F_IN normalised; B: N base picks. pi_theta's logits."""
    import torch
    z = net(Xn).squeeze(-1)
    return z + ALPHA * torch.nn.functional.one_hot(B, z.shape[1]).double()


def torch_log_mu(z, B, A, eps):
    import torch
    R = z.shape[1]
    pi = torch.softmax(z, 1)
    u = (1.0 - torch.nn.functional.one_hot(B, R).double()) / (R - 1)
    mu = (1.0 - eps) * pi + eps * u
    return torch.log(mu.gather(1, A[:, None]).squeeze(1))


def advantages(group: list, rewards: list, t_gamma: float) -> list:
    """``group``: the M >= 2 rollouts of one window (same arrivals); ``rewards`` theirs (possibly with a planted
    reward). Leave-one-out baseline per decision."""
    if len(group) < 2:
        raise ValueError("the leave-one-out baseline needs at least 2 rollouts per window")
    t = group[0]["t"]
    for e in group[1:]:
        if len(e["t"]) != len(t) or not np.array_equal(e["t"], t):
            raise RuntimeError("rollouts of one window do not share their decisions")
    G = np.stack([discounted_returns(r, e["t"], t_gamma) for e, r in zip(group, rewards)])
    M = len(group)
    base = (G.sum(axis=0, keepdims=True) - G) / (M - 1)
    return list(G - base)


def snr(groups: list, rewards: list, t_gamma: float) -> dict:
    """Mean raw advantage of override and keep actions (used decisions) and the override mean's t statistic."""
    ov, kp = [], []
    for g, rs in zip(groups, rewards):
        A = advantages(g, rs, t_gamma)
        for e, adv in zip(g, A):
            m = loss_mask(e["t"], t_gamma)
            o = (e["a"] != e["b"]) & m
            ov.append(adv[o])
            kp.append(adv[~(e["a"] != e["b"]) & m])
    ov, kp = np.concatenate(ov), np.concatenate(kp)
    se = float(ov.std(ddof=1) / math.sqrt(len(ov))) if len(ov) > 1 else float("nan")
    return {"n_override": int(len(ov)), "override_mean": float(ov.mean()) if len(ov) else float("nan"),
            "override_se": se, "t": float(ov.mean() / se) if len(ov) > 1 and se > 0 else float("nan"),
            "keep_mean": float(kp.mean()) if len(kp) else float("nan")}


def ppo_update(net, opt, batch: dict, cfg: dict, rng: np.random.Generator, mu, sd) -> dict:
    import torch
    X = torch.from_numpy(normalise(batch["X"], mu, sd))
    B = torch.from_numpy(batch["b"].astype(np.int64))
    A = torch.from_numpy(batch["a"].astype(np.int64))
    old = torch.from_numpy(batch["logp"])
    adv = torch.from_numpy(batch["adv"])
    N = len(adv)
    st = {"kl": [], "clipfrac": [], "loss": [], "grad_norm": []}
    epochs_run = 0
    for _ in range(cfg["epochs"]):
        perm = rng.permutation(N)
        for i in range(0, N, cfg["minibatch"]):
            j = torch.from_numpy(perm[i:i + cfg["minibatch"]])
            lp = torch_log_mu(torch_logits(net, X[j], B[j]), B[j], A[j], cfg["eps"])
            ratio = torch.exp(lp - old[j])
            s1, s2 = ratio * adv[j], torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv[j]
            loss = -torch.min(s1, s2).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
            opt.step()
            st["grad_norm"].append(float(gn))
            st["clipfrac"].append(float(((ratio.detach() - 1).abs() > cfg["clip"]).double().mean()))
            st["loss"].append(loss.item())
        epochs_run += 1
        with torch.no_grad():
            kl = float((old - torch_log_mu(torch_logits(net, X, B), B, A, cfg["eps"])).mean())
        st["kl"].append(kl)
        if kl > 1.5 * cfg["target_kl"]:
            break
    return {"kl": st["kl"][-1], "epochs_run": epochs_run, "clipfrac": float(np.mean(st["clipfrac"])),
            "loss": float(np.mean(st["loss"])), "grad_norm": float(np.mean(st["grad_norm"]))}


# ------------------------------------------------------------------ driver

def draw_start(rng: np.random.Generator, trace: str, length: int, grid: int) -> int:
    """A training window start: any request index, or (``grid`` > 0) a multiple of ``grid``, which bounds the number
    of window files the simulator caches (about 40 MB per 6,000-request window; the Mac's disk is small)."""
    hi = N_REQ[trace] - length
    return int(rng.integers(0, hi)) if grid <= 0 else int(grid * rng.integers(0, hi // grid + 1))


def reward_constants(c0_out: Path) -> dict:
    """Per trace: tau = median over C0's windows of the bar's best P99; s_tbt = median of C0's windows; w = C0's."""
    out = {}
    for tr in TRACES:
        summ = [json.loads(p.read_text()) for p in sorted(Path(c0_out).glob(f"*-{tr}-*/summary.json"))]
        if not summ:
            raise SystemExit(f"no C0 summaries for {tr} in {c0_out}")
        out[tr] = {"taus": {m: median(s["taus"][m] for s in summ) for m in c0.METRICS4},
                   "s_tbt": median(s["s_tbt"] for s in summ), "w": c0.WEIGHTS[tr], "windows": len(summ)}
    return out


def eval_jobs(params, consts: dict, mode: str = "greedy", eps: float = 0.0) -> list:
    return [{"trace": tr, "start": s, "length": EVAL_LEN, "draw": d, "params": params, "mode": mode, "seed": 7,
             "eps": eps, "record": False, "consts": consts, **SCEN}
            for tr in TRACES for s in EVAL_STARTS[tr] for d in EVAL_DRAWS]


def _key(e):
    return (e["trace"], e["start"], e["draw"])


EVAL_METRICS = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e", "p50_e2e", "score")


def eval_summary(res: list, ref: dict, seed: int = 0) -> dict:
    """Geometric-mean ratios to 4873 with a 95 % bootstrap CI over request ranges (each with all its draws)."""
    ranges = sorted({(e["trace"], e["start"]) for e in res})
    out = {}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ranges), (2000, len(ranges)))
    for m in EVAL_METRICS:
        per = {rg: [math.log(e[m] / ref[_key(e)][m]) for e in res if (e["trace"], e["start"]) == rg] for rg in ranges}
        L = np.array([np.mean(per[rg]) for rg in ranges])
        bs = L[picks].mean(1)
        out[m] = {"ratio": float(np.exp(L.mean())), "lo": float(np.exp(np.percentile(bs, 2.5))),
                  "hi": float(np.exp(np.percentile(bs, 97.5)))}
    out["override_rate"] = float(sum(e["overrides"] for e in res) / sum(e["calls"] for e in res))
    return out


def _read_log(path: Path, upto: int) -> None:
    """Drop log rows written after the checkpoint (a crash between checkpoint and log cannot duplicate rows)."""
    if path.exists():
        rows = [x for x in path.read_text().splitlines() if x.strip() and json.loads(x)["iter"] <= upto]
        path.write_text("".join(x + "\n" for x in rows))


def train(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULTS | {k: v for k, v in vars(a).items() if k in DEFAULTS and v is not None}
    consts = a.consts if getattr(a, "consts", None) else reward_constants(Path(a.c0_out))
    meta = {"seed": a.seed, "cfg": cfg, "consts": consts, "base": BASE, "base_sha": ev.file_sha(Path(BASE))[:12],
            "code_version": ev.code_version(), "train_draws": TRAIN_DRAWS, "eval_draws": EVAL_DRAWS,
            "rl_sha": ev.file_sha(Path(__file__))[:12], "features": cf.FEATURES, "scen": SCEN,
            "eval": {"starts": EVAL_STARTS, "len": EVAL_LEN}, "alpha": ALPHA}
    meta = json.loads(json.dumps(meta))
    mpath = out / "meta.json"
    if mpath.exists():
        if json.loads(mpath.read_text()) != meta:
            raise SystemExit(f"{mpath}: another setup or code; move it away")
    else:
        mpath.write_text(json.dumps(meta, indent=1))
    rng = np.random.default_rng(1000 + a.seed)
    ctxm = mp.get_context("forkserver")
    ctxm.set_forkserver_preload(["evolve.rl_plan.rl_residual"])
    deadline = time.time() + 3600 * a.hours
    W = cfg["windows_per_iter"] * cfg["rollouts"]
    with ctxm.Pool(a.workers, maxtasksperchild=8) as pool:
        import torch  # after the pool: workers never load torch
        torch.set_num_threads(min(8, a.workers))
        ck = out / "ckpt.pt"
        logp = out / "log.jsonl"
        if ck.exists():
            st = torch.load(ck, weights_only=False)
            P, it, ref, rng, decisions = st["params"], st["iter"], st["ref"], st["rng"], st["decisions"]
            _read_log(logp, it)
        else:
            # iteration 0: 4873 itself (zero residual, greedy) on training windows for the feature normalisation,
            # and on the eval windows as the learning curves' reference
            P0 = init_params(np.random.default_rng(a.seed), np.zeros(F_IN), np.ones(F_IN))
            save_params(P0, out / "params_0.npz")
            warm = [{"trace": TRACES[i % 2], "start": draw_start(rng, TRACES[i % 2], cfg["length"], cfg["start_grid"]),
                     "length": cfg["length"], "draw": int(rng.choice(TRAIN_DRAWS)), "params": str(out / "params_0.npz"),
                     "mode": "greedy", "seed": 0, "record": True, "consts": consts, **SCEN} for i in range(4)]
            res = pool.map(run_episode, warm + eval_jobs(str(out / "params_0.npz"), consts), chunksize=1)
            Xw = np.concatenate([e["X"].reshape(-1, F_IN) for e in res[:len(warm)]])
            mu, sd = Xw.mean(0), Xw.std(0)
            sd[sd < 1e-6] = 1.0
            mu[-1], sd[-1] = 0.0, 1.0  # the is-base flag stays 0/1
            P = init_params(np.random.default_rng(a.seed), mu, sd)
            ref = {_key(e): e for e in res[len(warm):]}
            it, decisions = 0, 0
            _read_log(logp, -1)
            logp.write_text(json.dumps({"iter": 0, "eval_ref": {f"{k[0]}:{k[1]}:p{k[2]}": {m: v[m] for m in
                                        EVAL_METRICS} for k, v in ref.items()}}) + "\n")
        net = _torch_net(P)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        if ck.exists():
            opt.load_state_dict(st["opt"])
        log = open(logp, "a")
        iter_s = None
        while time.time() + (iter_s or 0) * 1.5 < deadline and (a.max_iters is None or it < a.max_iters):
            t_it = time.time()
            it += 1
            ppath = out / "params_cur.npz"
            save_params(P, ppath)
            jobs, wins = [], []
            for i in range(cfg["windows_per_iter"]):
                tr = TRACES[i % 2]
                s_ = draw_start(rng, tr, cfg["length"], cfg["start_grid"])
                d = int(rng.choice(TRAIN_DRAWS))
                seeds = [int(x) for x in rng.integers(1 << 31, size=cfg["rollouts"])]
                wins.append([tr, s_, d, seeds])
                jobs += [{"trace": tr, "start": s_, "length": cfg["length"], "draw": d, "params": str(ppath),
                          "mode": "sample", "seed": sd_, "eps": cfg["eps"], "record": True, "consts": consts, **SCEN}
                         for sd_ in seeds]
            res = pool.map(run_episode, jobs, chunksize=1)
            groups = [res[i:i + cfg["rollouts"]] for i in range(0, len(res), cfg["rollouts"])]
            rewards = []
            for g in groups:
                rs = []
                for e in g:
                    r = e["r"].copy()
                    if cfg["plant"]:
                        r += cfg["plant"] * (e["a"] == planted_target(e["X"], e["b"]))
                    rs.append(r)
                rewards.append(rs)
            # advantages on the used decisions, normalised per trace
            parts = {k: [] for k in ("X", "b", "a", "logp", "adv", "trace")}
            for g, rs in zip(groups, rewards):
                for e, adv in zip(g, advantages(g, rs, cfg["t_gamma"])):
                    m = loss_mask(e["t"], cfg["t_gamma"])
                    for k_ in ("X", "b", "a", "logp"):
                        parts[k_].append(e[k_][m])
                    parts["adv"].append(adv[m])
                    parts["trace"].append(np.full(int(m.sum()), TRACES.index(e["trace"])))
            batch = {k: np.concatenate(v) for k, v in parts.items()}
            raw = {}
            for ti, tr in enumerate(TRACES):
                sel = batch["trace"] == ti
                raw[tr] = {"mean": float(batch["adv"][sel].mean()), "std": float(batch["adv"][sel].std())}
                batch["adv"][sel] = (batch["adv"][sel] - raw[tr]["mean"]) / (raw[tr]["std"] + 1e-8)
            ov = batch["a"] != batch["b"]
            pre = {}
            if cfg["plant"]:
                with torch.no_grad():
                    z = torch_logits(net, torch.from_numpy(normalise(batch["X"], P["mu"], P["sd"])),
                                     torch.from_numpy(batch["b"].astype(np.int64)))
                    pi = torch.softmax(z, 1).numpy()
                tgt = planted_target(batch["X"], batch["b"])
                non_b = 1.0 - pi[np.arange(len(tgt)), batch["b"]]
                pre = {"planted_share": float(np.mean(pi[np.arange(len(tgt)), tgt] / np.maximum(non_b, 1e-12))),
                       "planted_in_overrides": float(np.mean(batch["a"][ov] == tgt[ov])) if ov.any() else None}
            upd = ppo_update(net, opt, batch, cfg, rng, P["mu"], P["sd"])
            P = _params_from(net, P["mu"], P["sd"])
            decisions += int(sum(e["n"] for e in res))
            rec = {"iter": it, "decisions": decisions, "used": int(len(batch["adv"])), "wall_s": time.time() - t_it,
                   **upd, "adv_raw": raw, "windows": wins,
                   "override_rate": {tr: float(np.mean(ov[batch["trace"] == ti])) for ti, tr in enumerate(TRACES)},
                   "adv_norm_override": float(batch["adv"][ov].mean()) if ov.any() else None,
                   "adv_norm_keep": float(batch["adv"][~ov].mean()),
                   "snr": {str(T): snr(groups, rewards, T) for T in SNR_T},
                   "score": {tr: float(np.mean([e["score"] for e in res if e["trace"] == tr])) for tr in TRACES},
                   **pre}
            if it % cfg["eval_every"] == 0:
                save_params(P, out / f"params_{it}.npz")
                gr = pool.map(run_episode, eval_jobs(str(out / f"params_{it}.npz"), consts), chunksize=1)
                rec["eval"] = eval_summary(gr, ref)
                q = rec["eval"]["override_rate"]
                if q > 0:
                    pl = pool.map(run_episode, eval_jobs(None, consts, "placebo", q), chunksize=1)
                    rec["eval_placebo"] = eval_summary(pl, ref)
            torch.save({"params": P, "iter": it, "ref": ref, "rng": rng, "opt": opt.state_dict(),
                        "decisions": decisions}, out / "ckpt.tmp")
            os.replace(out / "ckpt.tmp", ck)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            iter_s = time.time() - t_it
            print(json.dumps({k: rec[k] for k in ("iter", "decisions", "wall_s", "kl", "epochs_run")}
                             | {"snr_t": {T: round(v["t"], 2) for T, v in rec["snr"].items()}}
                             | ({"planted_share": rec["planted_share"]} if "planted_share" in rec else {})),
                  flush=True)
    save_params(P, out / "params_final.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--c0-out", default="evolve/rl_plan/out/c0")
    ap.add_argument("--max-iters", type=int)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
