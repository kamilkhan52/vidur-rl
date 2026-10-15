"""The RL residual on 4873: zero residual reproduces 4873; rewards sum to -S; actor and learner agree; training runs."""
import argparse
import filecmp
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evolve.rl_plan import c0
from evolve.rl_plan import headroom as hr
from evolve.rl_plan import rl_residual as rl
from evolve.sim_fork import build_simulator
from evolve import evaluate as ev

SCEN = {"replicas": 4, "scale": 1.5}
SC = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1, model="l3_8b")


@pytest.fixture(scope="module", autouse=True)
def base_4873():
    old = hr.POLICY, hr.argv_for.__defaults__
    hr.set_base_policy(c0.BASE)
    yield
    hr.POLICY, hr.argv_for.__defaults__ = old


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    d = tmp_path_factory.mktemp("plain")
    h = build_simulator(hr.argv_for(SC, d))
    h.run()
    h.write_output()
    s = hr.score_output(d, SC)
    taus = {m: s[f"p99_{m}"] for m in c0.METRICS4}
    consts = {t: {"taus": taus, "s_tbt": 0.05, "w": 1000.0} for t in ("Tool", "Conv")}
    return d, consts


def _job(params, mode, seed, consts, **kw):
    return {"trace": "Conv", "start": 6000, "length": 200, "draw": 1, "params": str(params), "mode": mode,
            "seed": seed, "record": True, "consts": consts, **SCEN, **kw}


def _csv(d):
    return next(Path(d).rglob("request_metrics.csv"))


def _random_params(tmp_path, scale=1.0, seed=3):
    P = rl.init_params(np.random.default_rng(seed), np.zeros(rl.F_IN), np.ones(rl.F_IN))
    P["mu"], P["sd"] = np.zeros(rl.F_IN), np.full(rl.F_IN, 50.0)
    P["w3"] = np.random.default_rng(seed + 1).normal(0, scale, rl.HIDDEN)
    P["b3"] = np.array([-rl.ALPHA + 6.0])  # residual large enough to override often
    path = tmp_path / "p.npz"
    rl.save_params(P, path)
    return P, path


def test_overlap_cum_matches_brute_force():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 10, 50)
    e = s + rng.uniform(-1, 5, 50)
    t = np.sort(rng.uniform(-1, 16, 30))
    brute = np.array([sum(max(0.0, min(ei, ti) - si) for si, ei in zip(s, e) if ei > si) for ti in t])
    assert np.allclose(rl._overlap_cum(s, e, t), brute)


def test_zero_residual_greedy_is_4873_and_rewards_sum_to_minus_score(plain, tmp_path):
    d, consts = plain
    P = rl.init_params(np.random.default_rng(0), np.zeros(rl.F_IN), np.ones(rl.F_IN))
    rl.save_params(P, tmp_path / "p0.npz")
    out = tmp_path / "out"
    out.mkdir()
    e = rl.run_episode(_job(tmp_path / "p0.npz", "greedy", 0, consts, write_dir=str(out)))
    assert e["overrides"] == 0 and e["calls"] == e["n"] == len(e["a"])
    assert filecmp.cmp(_csv(d), _csv(out), shallow=False)
    assert math.isclose(e["r"].sum(), -e["score"], rel_tol=1e-9)
    # placebo with eps = 0 is 4873 too
    out2 = tmp_path / "out2"
    out2.mkdir()
    rl.run_episode(_job(None, "placebo", 3, consts, eps=0.0, record=False, write_dir=str(out2)))
    assert filecmp.cmp(_csv(d), _csv(out2), shallow=False)
    # sampling with zero residual keeps 4873's pick with probability (1 - eps) pi(b)
    eps = 0.1
    es = rl.run_episode(_job(tmp_path / "p0.npz", "sample", 7, consts, eps=eps))
    assert 0.0 < es["overrides"] / es["n"] < 0.3
    assert math.isclose(es["r"].sum(), -es["score"], rel_tol=1e-9)
    pib = math.exp(rl.ALPHA) / (math.exp(rl.ALPHA) + 3)
    assert np.allclose(np.exp(es["logp"][es["a"] == es["b"]]), (1 - eps) * pib)


def test_actor_logp_equals_learner_logp(plain, tmp_path):
    import torch
    _, consts = plain
    P, path = _random_params(tmp_path)
    e = rl.run_episode(_job(path, "sample", 11, consts, eps=0.05))
    assert e["overrides"] > 0  # overrides are booked and the episode completes
    net = rl._torch_net(P)
    X = torch.from_numpy(rl.normalise(e["X"], P["mu"], P["sd"]))
    B, A = torch.from_numpy(e["b"].astype(np.int64)), torch.from_numpy(e["a"].astype(np.int64))
    lp = rl.torch_log_mu(rl.torch_logits(net, X, B), B, A, 0.05).detach().numpy()
    assert np.allclose(lp, e["logp"], atol=1e-5)  # X is stored as float32
    # numpy actor logits equal the learner's
    z_np = np.stack([rl.logits(P, x.astype(np.float64), b) for x, b in zip(e["X"][:20], e["b"][:20])])
    z_t = rl.torch_logits(net, X[:20], B[:20]).detach().numpy()
    assert np.allclose(z_np, z_t, atol=1e-6)


def test_advantages_leave_one_out(plain, tmp_path):
    _, consts = plain
    _, path = _random_params(tmp_path, scale=0.3)
    g = [rl.run_episode(_job(path, "sample", s, consts, eps=0.05)) for s in (1, 2, 3)]
    rs = [e["r"] for e in g]
    A = rl.advantages(g, rs, 20.0)
    G = [rl.discounted_returns(e["r"], e["t"], 20.0) for e in g]
    assert np.allclose(A[0], G[0] - (G[1] + G[2]) / 2)
    assert np.allclose(rl.discounted_returns(rs[0], g[0]["t"], math.inf), np.cumsum(rs[0][::-1])[::-1])
    bad = dict(g[1], t=g[1]["t"] + 1e-3)
    with pytest.raises(RuntimeError):
        rl.advantages([g[0], bad], rs[:2], 20.0)
    with pytest.raises(ValueError):
        rl.advantages([g[0]], rs[:1], 20.0)
    s = rl.snr([g], [rs], 20.0)
    assert s["n_override"] > 0


def test_train_smoke(plain, tmp_path, monkeypatch):
    _, consts = plain
    monkeypatch.setattr(rl, "SCEN", SCEN)
    monkeypatch.setattr(rl, "N_REQ", {"Tool": 1200, "Conv": 1200})
    monkeypatch.setattr(rl, "EVAL_STARTS", {"Tool": (100,), "Conv": (300, 600)})
    monkeypatch.setattr(rl, "EVAL_DRAWS", (40,))
    monkeypatch.setattr(rl, "EVAL_LEN", 200)
    a = argparse.Namespace(out=str(tmp_path / "run"), seed=0, workers=2, hours=1.0, c0_out=None, max_iters=2,
                           consts=consts, **{k: None for k in rl.DEFAULTS})
    a.windows_per_iter, a.rollouts, a.eval_every, a.minibatch, a.length = 2, 2, 1, 256, 300
    a.eps, a.plant, a.t_gamma, a.lr = 0.2, 5000.0, 20.0, 3e-3
    rl.train(a)
    log = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [0, 1, 2]
    assert all("eval" in r and r["eval"]["p99_e2e"]["ratio"] > 0 for r in log[1:])
    assert all(0 < r["planted_share"] < 1 and r["used"] > 0 and "snr" in r for r in log[1:])
    assert (tmp_path / "run" / "params_final.npz").exists()
    # resume continues from the checkpoint
    a.max_iters = 3
    rl.train(a)
    log = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [0, 1, 2, 3]
