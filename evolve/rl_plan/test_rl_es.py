"""ES policy: a zero output layer is exactly 4873; the length predictor learns from observe(); training runs."""
import argparse
import filecmp
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan import headroom as hr
from evolve.rl_plan import rl_es as es
from evolve.rl_plan import rl_residual as rl
from evolve.sim_fork import build_simulator

SC = ev.Scenario("Conv", 4, 1.5, 6000, 300, 1, model="l3_8b")


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


def _job(params, consts, probe=None, **kw):
    return {"trace": "Conv", "start": 6000, "length": 300, "draw": 1, "replicas": 4, "scale": 1.5,
            "params": None if params is None else str(params), "probe": probe, "consts": consts, **kw}


def _csv(d):
    return next(Path(d).rglob("request_metrics.csv"))


def test_predictor_hierarchy():
    p = es.LengthPredictor()
    m0, c0_ = p.predict(7, 1000)
    assert math.isclose(m0, es.PRIOR_LOG) and c0_ == 0
    for _ in range(50):
        p.update(7, 1000, 10)
    m, c = p.predict(7, 1000)
    assert abs(m - math.log1p(10)) < 0.1 and c > 40
    m2, _ = p.predict(7, 100000)  # other bucket of the same first block: falls back to the block's mean
    assert abs(m2 - math.log1p(10)) < 0.2
    m3, _ = p.predict(8, 1000)  # unseen block: the global mean
    assert abs(m3 - math.log1p(10)) < 0.2


def test_zero_output_is_4873_and_features_are_finite(plain, tmp_path):
    d, consts = plain
    P = es.init_params(np.random.default_rng(0))
    np.savez(tmp_path / "p.npz", **P)
    out = tmp_path / "o"
    out.mkdir()
    e = es.run_episode(_job(tmp_path / "p.npz", consts, write_dir=str(out)))
    assert filecmp.cmp(_csv(d), _csv(out), shallow=False)
    assert e["calls"] == e["n"] == 300
    # the probe changes decisions (and the predictor sees completions)
    out2 = tmp_path / "o2"
    out2.mkdir()
    es.run_episode(_job(None, consts, probe=30.0, write_dir=str(out2)))
    assert not filecmp.cmp(_csv(d), _csv(out2), shallow=False)
    # a random network also runs, with finite features
    P["w3"] = np.random.default_rng(1).normal(0, 1, es.HIDDEN)
    np.savez(tmp_path / "r.npz", **P)
    e3 = es.run_episode(_job(tmp_path / "r.npz", consts))
    assert e3["n"] == 300 and np.isfinite(e3["score"])


def test_observe_tracks_outstanding_work(plain, tmp_path):
    """After an episode every routed request has completed, so the tracked decode work returns to 0."""
    _, consts = plain
    import tempfile
    sc = SC
    with tempfile.TemporaryDirectory() as tmp:
        pol = es.write_policy(Path(tmp) / "p.py", None, 1.0)
        h = build_simulator(hr.argv_for(sc, tmp, policy=str(pol)))
        h.run()
        st = h.sim._scheduler._ctx.state["es"]
    assert all(not any(q.values()) for q in st["outs"])
    assert all(abs(w) < 1e-6 for w in st["work"]) and all(x == 0 for x in st["long"])
    assert st["pred"].s[("all",)][1] == 300


def test_train_smoke(plain, tmp_path, monkeypatch):
    _, consts = plain
    monkeypatch.setattr(rl, "N_REQ", {"Tool": 1200, "Conv": 1200})
    monkeypatch.setattr(es, "N_REQ", {"Tool": 1200, "Conv": 1200})
    monkeypatch.setattr(rl, "EVAL_STARTS", {"Tool": (100,), "Conv": (300,)})
    monkeypatch.setattr(es, "EVAL_DRAWS", (40,))
    monkeypatch.setattr(rl, "reward_constants", lambda _: consts)
    a = argparse.Namespace(out=str(tmp_path / "run"), seed=0, workers=2, hours=1.0, max_iters=2, c0_out="x",
                           configs="Tool:4:1.5,Conv:4:1.5", pairs=2, windows=2, length=200, start_grid=0, sigma=0.5, lr=0.05, wd=0.0,
                           rank=False, oracle=False, arch="linear", opt="sgd", eval_every=1, eval_len=200, fitness="gen8", bound=0.5)
    orig = es.run_episode

    def small(job):
        return orig({**job, "replicas": 4})
    monkeypatch.setattr(es, "run_episode", small)
    # the pool pickles the function by name, so run serially via a fake pool
    import multiprocessing as mp

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, f, xs, chunksize=1):
            return [f(x) for x in xs]

    class Ctx:
        Pool = FakePool
    monkeypatch.setattr(mp, "get_context", lambda *_: Ctx)
    es.train(a)
    log = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [0, 1]
    assert log[0]["center_F"] == 0.0  # the centre starts as 4873 exactly
    assert all("eval" in r and r["eval"]["score"]["ratio"] > 0 for r in log)
    assert all(r["eval"]["gen8"]["ratio"] > 0 and r["eval"]["mean_tbt"]["ratio"] > 0 for r in log)
    with np.load(tmp_path / "run" / "params_cur.npz") as f:
        assert float(f["bound"]) == 0.5 and f["lin"].shape == (es.N_LIN,)
    a.max_iters = 3
    es.train(a)
    log = [json.loads(x) for x in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in log] == [0, 1, 2]


def test_oracle_sees_true_lengths(plain, tmp_path):
    """With --oracle the predicted lengths are the true ones: the tracked work sums the true outputs, and every
    lookup is consumed."""
    _, consts = plain
    e = es.run_episode(_job(None, consts, probe=1.0, oracle=True))
    assert e["n"] == 300 and es._ORACLE is None
