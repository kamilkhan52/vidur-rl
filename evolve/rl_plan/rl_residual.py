"""Large-scale model-free RL, residual on finalist 4873 (Kamil, 2026-10-14: an agent routes every request and learns
from rewards with lots of data; start exactly at 4873; pilot of about 500 CPU-h with learning curves).

Policy. At every routing call the base router 4873 picks b. Each replica c gets the logit
    z_c = f_theta(x_c) + alpha * [c == b]
where x_c are C0's 29 features of c relative to b (``c0_features``) plus an is-base flag, normalised, and f_theta is
a small MLP shared across replicas whose last layer starts at zero. With f = 0 the greedy policy is exactly 4873 and
the sampling policy keeps 4873's pick with probability ``P_KEEP`` (exploration). An override is booked as C0's router
books it (4873's attended-context FIFO entry moves to the chosen replica), so a greedy policy with f = 0 reproduces
4873's run exactly (tested).

Reward. The C7 score of an episode (a 2,000-request window), S = sum E2E + w (H_e2e + H_ttft + H_tpot + s_tbt H_tbt)
with hinges H = sum (x / tau - 1)+, is split into the cost incurred between consecutive routing decisions: E2E time
and the E2E and TTFT hinges accrue continuously over each request's life (the hinge from age tau on), the TPOT and
TBT hinges at the request's completion. The rewards of an episode sum to -S exactly (tested). tau and s_tbt are
per-trace constants (medians over C0's windows), w is C0's per-trace weight.

Learning. PPO on decisions. Every window is run ``M`` times with the same arrivals and different action samples; the
return from decision k (semi-MDP discount exp(-dt / T_GAMMA) over simulated time) is compared with the mean return of
the other M - 1 rollouts at the same decision (the same request): an input-driven baseline (Mao et al., ICLR 2019).

Data. Training: Tool and Conv, 8 replicas, load 0.75, 2,000-request windows at random starts, arrival draws p22-p39
(never used elsewhere). Learning curves: the greedy policy against 4873 on fixed windows with draws p40-p43, which are
never trained on. Validation (heldout3 p2, p3, p5) and test (test3b) are untouched.

Usage (Alpine):
    python -m evolve.rl_plan.rl_residual --out evolve/rl_plan/out/rl/s0 --seed 0 --workers 31 --hours 5
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
LEN = 2000
TRAIN_DRAWS = tuple(range(22, 40))
EVAL_DRAWS = (40, 41, 42, 43)
EVAL_STARTS = {"Tool": (1000, 9000, 17000), "Conv": (1000, 5000, 9000)}
SCEN = {"replicas": 8, "scale": 0.75, "length": LEN}  # tests shrink it
F_IN = len(cf.FEATURES) + 1  # + is_base
HIDDEN = 64
P_KEEP = 0.95
TRAJ_KEY = "rl_traj"
HIST_KEY = "rl_hist"
COUNT_KEY = "rl_counts"  # [routing calls, overrides]

DEFAULTS = {"windows_per_iter": 8, "rollouts": 4, "t_gamma": 20.0, "lr": 3e-4, "clip": 0.2, "epochs": 4,
            "minibatch": 4096, "ent_coef": 0.0, "eval_every": 10, "max_grad_norm": 0.5}


# ------------------------------------------------------------------ policy (numpy, used inside the simulator)

def init_params(rng: np.random.Generator, mu: np.ndarray, sd: np.ndarray, n_rep: int = 8) -> dict:
    def glorot(a, b):
        return rng.normal(0.0, math.sqrt(2.0 / (a + b)), (a, b)).astype(np.float64)
    return {"W1": glorot(F_IN, HIDDEN), "b1": np.zeros(HIDDEN), "W2": glorot(HIDDEN, HIDDEN), "b2": np.zeros(HIDDEN),
            "w3": np.zeros(HIDDEN), "b3": np.zeros(1),
            "alpha": np.array([math.log(P_KEEP * (n_rep - 1) / (1.0 - P_KEEP))]), "mu": mu, "sd": sd}


def rows_in(rows, b: int) -> np.ndarray:
    X = np.asarray(rows, float)
    isb = np.zeros((len(X), 1))
    isb[b, 0] = 1.0
    return np.hstack([X, isb])


def logits(P: dict, X: np.ndarray, b: int) -> np.ndarray:
    """X: R x F_IN raw inputs (``rows_in``)."""
    Z = (X - P["mu"]) / P["sd"]
    h = np.tanh(Z @ P["W1"] + P["b1"])
    h = np.tanh(h @ P["W2"] + P["b2"])
    z = h @ P["w3"] + P["b3"][0]
    z[b] += P["alpha"][0]
    return z


def build_policy(params_path: str, mode: str, seed: int, record: bool = True):
    """``mode``: 'sample' (training) or 'greedy' (evaluation; ties go to 4873's pick)."""
    base = load_base(BASE)
    with np.load(params_path) as f:
        P = {k: f[k] for k in f.files}
    rng = np.random.default_rng(seed)

    def choose(request, replicas, ctx):
        b = base.choose(request, replicas, ctx)
        hist = ctx.state.setdefault(HIST_KEY, [])
        req, reps = view_dicts(request, replicas)
        X = rows_in(cf.features(req, reps, b, cf.hist_features(hist, ctx.now)), b)
        if request.position_in_burst == 0:
            hist.append((ctx.now, request.burst_size))
            cf.prune_hist(hist, ctx.now)
        z = logits(P, X, b)
        if mode == "greedy":
            a = b if z[b] >= z.max() else int(np.argmax(z))
            lp = 0.0
        else:
            p = np.exp(z - z.max())
            p /= p.sum()
            a = int(rng.choice(len(z), p=p))
            lp = float(math.log(p[a]))
        cnt = ctx.state.setdefault(COUNT_KEY, [0, 0])
        cnt[0] += 1
        cnt[1] += a != b
        if record:
            ctx.state.setdefault(TRAJ_KEY, []).append((X.astype(np.float32), b, a, lp, ctx.now,
                                                       request.num_prefill_tokens))
        if a != b:
            move_fifo(ctx.state, request.num_prefill_tokens, b, a)
        return a

    return choose


POLICY_TEMPLATE = '''"""RL residual policy (generated by evolve.rl_plan.rl_residual; do not edit)."""
from evolve.rl_plan.rl_residual import build_policy as _build

choose = _build({params!r}, {mode!r}, {seed!r}, {record!r})
'''


def write_policy(path: Path, params_path: str, mode: str, seed: int, record: bool = True) -> Path:
    path.write_text(POLICY_TEMPLATE.format(params=str(Path(params_path).resolve()), mode=mode, seed=seed,
                                           record=record))
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
        acc = r[k] + math.exp(-dt / t_gamma) * acc
        G[k] = acc
    return G


# ------------------------------------------------------------------ episodes (in pool workers)

def p99s(arr: dict) -> dict:
    return {"p99_e2e": float(np.percentile(arr["e2e"], 99)), "p99_ttft": float(np.percentile(arr["ttft"], 99)),
            "p99_tpot": float(np.percentile(arr["tpot"], 99)) if len(arr["tpot"]) else float("nan"),
            "p99_tbt": float(np.percentile(arr["gaps"], 99)) if len(arr["gaps"]) else float("nan"),
            "mean_e2e": float(arr["e2e"].mean()), "p50_e2e": float(np.median(arr["e2e"]))}


def run_episode(job: dict) -> dict:
    """job: trace, start, draw, params, mode, seed, record, consts, replicas, scale, length (``SCEN``),
    optionally write_dir (the simulator's outputs are written there)."""
    hr.set_base_policy(BASE)
    sc = ev.Scenario(job["trace"], job["replicas"], job["scale"], job["start"], job["length"], job["draw"],
                     model="l3_8b")
    k = job["consts"][job["trace"]]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        pol = write_policy(Path(tmp) / "rl_policy.py", job["params"], job["mode"], job["seed"], job["record"])
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
        traj = h.sim._scheduler._ctx.state.get(TRAJ_KEY, [])
        calls, over = h.sim._scheduler._ctx.state.get(COUNT_KEY, [0, 0])
        if job.get("write_dir"):
            h.write_output()
    out = {"trace": job["trace"], "start": job["start"], "draw": job["draw"], "seed": job["seed"],
           "mode": job["mode"], "n": len(made), "wall_s": time.time() - t0, **p99s(arr),
           "score": score_total(arr, k["taus"], k["w"], k["s_tbt"]), "calls": calls, "overrides": over}
    if job["record"]:
        if len(traj) != len(made):
            raise RuntimeError(f"{len(traj)} routing calls for {len(made)} requests")
        t_dec = np.array([x[4] for x in traj])
        if not np.allclose(t_dec, arr["arrived"]) or [x[5] for x in traj] != [r.num_prefill_tokens for r in made]:
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
    alpha = torch.nn.Parameter(torch.tensor(float(P["alpha"][0]), dtype=torch.float64))
    return net, alpha


def _params_from(net, alpha, mu, sd) -> dict:
    return {"W1": net[0].weight.detach().numpy().T.copy(), "b1": net[0].bias.detach().numpy().copy(),
            "W2": net[2].weight.detach().numpy().T.copy(), "b2": net[2].bias.detach().numpy().copy(),
            "w3": net[4].weight.detach().numpy()[0].copy(), "b3": net[4].bias.detach().numpy().copy(),
            "alpha": np.array([float(alpha.detach())]), "mu": mu, "sd": sd}


def torch_logits(net, alpha, Xn, B):
    """Xn: N x R x F_IN normalised; B: N base picks."""
    import torch
    z = net(Xn).squeeze(-1)
    return z + alpha * torch.nn.functional.one_hot(B, z.shape[1]).double()


def advantages(group: list, t_gamma: float) -> list:
    """``group``: the M rollouts of one window (same arrivals). Leave-one-out baseline per decision."""
    t = group[0]["t"]
    for e in group[1:]:
        if len(e["t"]) != len(t) or not np.array_equal(e["t"], t):
            raise RuntimeError("rollouts of one window do not share their decisions")
    G = np.stack([discounted_returns(e["r"], e["t"], t_gamma) for e in group])
    M = len(group)
    base = (G.sum(axis=0, keepdims=True) - G) / (M - 1)
    return list(G - base)


def ppo_update(state: dict, batch: dict, cfg: dict, rng: np.random.Generator) -> dict:
    import torch
    net, alpha, opt = state["net"], state["alpha"], state["opt"]
    mu, sd = state["mu"], state["sd"]
    X = torch.from_numpy(((batch["X"].astype(np.float64) - mu) / sd))
    B = torch.from_numpy(batch["b"].astype(np.int64))
    A = torch.from_numpy(batch["a"].astype(np.int64))
    old = torch.from_numpy(batch["logp"])
    adv = batch["adv"]
    adv = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
    N = len(adv)
    stats = {"kl": [], "clipfrac": [], "entropy": [], "loss": []}
    for _ in range(cfg["epochs"]):
        perm = rng.permutation(N)
        for i in range(0, N, cfg["minibatch"]):
            j = torch.from_numpy(perm[i:i + cfg["minibatch"]])
            lp_all = torch.log_softmax(torch_logits(net, alpha, X[j], B[j]), dim=1)
            lp = lp_all.gather(1, A[j][:, None]).squeeze(1)
            ratio = torch.exp(lp - old[j])
            s1, s2 = ratio * adv[j], torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv[j]
            ent = -(lp_all.exp() * lp_all).sum(1).mean()
            loss = -torch.min(s1, s2).mean() - cfg["ent_coef"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [alpha], cfg["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                stats["kl"].append(float((old[j] - lp).mean()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > cfg["clip"]).double().mean()))
                stats["entropy"].append(float(ent))
                stats["loss"].append(float(loss))
    return {k: float(np.mean(v)) for k, v in stats.items()} | {"alpha": float(alpha.detach())}


# ------------------------------------------------------------------ driver

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


def eval_jobs(params: str, consts: dict, mode: str = "greedy") -> list:
    return [{"trace": tr, "start": s, "draw": d, "params": params, "mode": mode, "seed": 0, "record": False,
             "consts": consts, **SCEN} for tr in TRACES for s in EVAL_STARTS[tr] for d in EVAL_DRAWS]


def _key(e):
    return (e["trace"], e["start"], e["draw"])


def eval_summary(res: list, ref: dict) -> dict:
    """Greedy policy vs 4873 on the eval windows: geometric-mean ratios with a bootstrap CI over windows."""
    out = {}
    for m in ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e", "p50_e2e", "score"):
        L = np.array([math.log(e[m] / ref[_key(e)][m]) for e in res])
        bs = np.random.default_rng(0).integers(0, len(L), (2000, len(L)))
        out[m] = {"ratio": float(np.exp(L.mean())), "lo": float(np.exp(np.percentile(L[bs].mean(1), 2.5))),
                  "hi": float(np.exp(np.percentile(L[bs].mean(1), 97.5)))}
    out["override_rate"] = float(sum(e["overrides"] for e in res) / sum(e["calls"] for e in res))
    return out


def train(a) -> None:
    import torch
    torch.set_num_threads(min(8, a.workers))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DEFAULTS | {k: v for k, v in vars(a).items() if k in DEFAULTS and v is not None}
    consts = a.consts if getattr(a, "consts", None) else reward_constants(Path(a.c0_out))
    meta = {"seed": a.seed, "cfg": cfg, "consts": consts, "base": BASE, "base_sha": ev.file_sha(Path(BASE))[:12],
            "code_version": ev.code_version(), "train_draws": TRAIN_DRAWS, "eval_draws": EVAL_DRAWS,
            "rl_sha": ev.file_sha(Path(__file__))[:12], "features": cf.FEATURES, "scen": SCEN,
            "eval_starts": EVAL_STARTS}
    mpath = out / "meta.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if {k: old[k] for k in ("seed", "cfg", "base_sha", "code_version", "features")} != \
                {k: json.loads(json.dumps(meta[k])) for k in ("seed", "cfg", "base_sha", "code_version", "features")}:
            raise SystemExit(f"{mpath}: another setup; move it away")
    else:
        mpath.write_text(json.dumps(meta, indent=1))
    rng = np.random.default_rng(1000 + a.seed)
    ctxm = mp.get_context("fork")
    deadline = time.time() + 3600 * a.hours
    log = open(out / "log.jsonl", "a")
    with ctxm.Pool(a.workers, maxtasksperchild=4) as pool:
        ck = out / "ckpt.pt"
        if ck.exists():
            st = torch.load(ck, weights_only=False)
            P, it, ref, rng = st["params"], st["iter"], st["ref"], st["rng"]
        else:
            # iteration 0: 4873 itself (zero residual, greedy) on training windows for the feature normalisation,
            # and on the eval windows as the learning curves' reference
            P0 = init_params(np.random.default_rng(a.seed), np.zeros(F_IN), np.ones(F_IN), SCEN["replicas"])
            save_params(P0, out / "params_0.npz")
            warm = [{"trace": TRACES[i % 2], "start": int(rng.integers(0, N_REQ[TRACES[i % 2]] - SCEN["length"])),
                     "draw": int(rng.choice(TRAIN_DRAWS)), "params": str(out / "params_0.npz"), "mode": "greedy",
                     "seed": 0, "record": True, "consts": consts, **SCEN} for i in range(cfg["windows_per_iter"])]
            res = pool.map(run_episode, warm + eval_jobs(str(out / "params_0.npz"), consts), chunksize=1)
            Xw = np.concatenate([e["X"].reshape(-1, F_IN) for e in res[:len(warm)]])
            mu, sd = Xw.mean(0), Xw.std(0)
            sd[sd < 1e-6] = 1.0
            mu[-1], sd[-1] = 0.0, 1.0  # the is-base flag stays 0/1
            P = init_params(np.random.default_rng(a.seed), mu, sd, SCEN["replicas"])
            ref = {_key(e): e for e in res[len(warm):]}
            it = 0
            log.write(json.dumps({"iter": 0, "eval_ref": {f"{k[0]}:{k[1]}:p{k[2]}": {m: v[m] for m in v if m in (
                "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "score")} for k, v in ref.items()}}) + "\n")
        net, alpha = _torch_net(P)
        opt = torch.optim.Adam(list(net.parameters()) + [alpha], lr=cfg["lr"])
        if ck.exists():
            opt.load_state_dict(st["opt"])
        state = {"net": net, "alpha": alpha, "opt": opt, "mu": P["mu"], "sd": P["sd"]}
        decisions = int(st["decisions"]) if ck.exists() else 0
        iter_s = None
        while time.time() + (iter_s or 0) * 1.5 < deadline and (a.max_iters is None or it < a.max_iters):
            t_it = time.time()
            it += 1
            ppath = out / "params_cur.npz"
            save_params(P, ppath)
            jobs = []
            for i in range(cfg["windows_per_iter"]):
                tr = TRACES[i % 2]
                st_ = int(rng.integers(0, N_REQ[tr] - SCEN["length"]))
                d = int(rng.choice(TRAIN_DRAWS))
                jobs += [{"trace": tr, "start": st_, "draw": d, "params": str(ppath), "mode": "sample",
                          "seed": int(rng.integers(1 << 31)), "record": True, "consts": consts, **SCEN}
                         for _ in range(cfg["rollouts"])]
            res = pool.map(run_episode, jobs, chunksize=1)
            groups = [res[i:i + cfg["rollouts"]] for i in range(0, len(res), cfg["rollouts"])]
            advs = [x for g in groups for x in advantages(g, cfg["t_gamma"])]
            batch = {k: np.concatenate([e[k] for e in res]) for k in ("X", "b", "a", "logp")}
            batch["adv"] = np.concatenate(advs)
            upd = ppo_update(state, batch, cfg, rng)
            P = _params_from(net, alpha, P["mu"], P["sd"])
            decisions += len(batch["adv"])
            rec = {"iter": it, "decisions": decisions, "wall_s": time.time() - t_it, **upd,
                   "train_override_rate": float(np.mean(batch["b"] != batch["a"])),
                   "train_score_by_trace": {tr: float(np.mean([e["score"] for e in res if e["trace"] == tr]))
                                            for tr in TRACES}}
            if it % cfg["eval_every"] == 0:
                save_params(P, out / f"params_{it}.npz")
                ev_res = pool.map(run_episode, eval_jobs(str(out / f"params_{it}.npz"), consts), chunksize=1)
                rec["eval"] = eval_summary(ev_res, ref)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"params": P, "iter": it, "ref": ref, "rng": rng, "opt": opt.state_dict(),
                        "decisions": decisions}, out / "ckpt.tmp")
            os.replace(out / "ckpt.tmp", ck)
            iter_s = time.time() - t_it
            print(json.dumps({k: v for k, v in rec.items() if k != "train_score_by_trace"}), flush=True)
    save_params(P, out / "params_final.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--c0-out", default="evolve/rl_plan/out/c0")
    ap.add_argument("--max-iters", type=int)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
