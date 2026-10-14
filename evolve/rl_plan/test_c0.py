"""Step C0 (``c0.py``, ``c0_features.py``, ``c0_policy.py``):
- the logger only observes: its parent run equals a plain run of the base router byte for byte, and every logged
  decision has a view matching it and all R candidates scored;
- a rollout's per-request E2E, TTFT, TPOT and token gaps equal what the simulator writes for those requests;
- the deployed router books an override exactly as sim_fork's forced decision does (same outputs);
- leave-one-range-out finds a planted gain and passes items 1-2; on pure noise item 1 fails;
- item 3's verdict on synthetic runs; the log mode refuses a window without its bar runs.
Simulations on Conv 4R s1.5 w6000+200 p1 with the finalist 4873 as base (~8 s per run)."""

import filecmp
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan import c0_features as cf
from evolve.rl_plan import headroom as hr
from evolve.sim_fork import after_os_fork, build_simulator

SC = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1, model="l3_8b")


@pytest.fixture(scope="module", autouse=True)
def base_4873():
    old = hr.POLICY, hr.argv_for.__defaults__
    hr.set_base_policy(c0.BASE)
    yield
    hr.POLICY, hr.argv_for.__defaults__ = old


def _csv(d):
    return next(Path(d).rglob("request_metrics.csv"))


def _same_outputs(a, b):
    assert filecmp.cmp(_csv(a), _csv(b), shallow=False)
    ga, gb = np.load(_csv(a).parent / "token_gaps.npz"), np.load(_csv(b).parent / "token_gaps.npz")
    for f in ga.files:
        assert np.array_equal(ga[f], gb[f])


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    d = tmp_path_factory.mktemp("plain")
    h = build_simulator(hr.argv_for(SC, d))
    h.run()
    h.write_output()
    return d, hr.score_output(d, SC)


def test_features():
    hist = [(0.0, 1), (55.0, 2), (95.0, 3)]
    f = cf.hist_features(hist, 100.0)
    assert f["log_n_10s"] == pytest.approx(np.log1p(3)) and f["log_n_60s"] == pytest.approx(np.log1p(5))
    assert f["dt_prev"] == pytest.approx(5.0) and cf.hist_features([], 1.0)["dt_prev"] == cf.DT_CAP
    req = {"num_prefill_tokens": 1000, "burst_size": 1, "session_last_replica": 2,
           "cached_prefix_tokens": [0, 512, 0], "inflight_prefix_tokens": [0, 0, 256], "active_prefix_tokens": []}
    reps = [{"outstanding_requests": i, "pending_prefill_tokens": 100 * i, "waiting": 0, "running": i,
             "kv_usage": 0.1 * i} for i in range(3)]
    rows = cf.features(req, reps, 1, f)
    assert len(rows) == 3 and all(len(r) == len(cf.FEATURES) for r in rows)
    F = dict(zip(cf.FEATURES, rows[1]))
    assert all(F[k] == 0 for k in cf.FEATURES if k.startswith("d_"))
    F2 = dict(zip(cf.FEATURES, rows[2]))
    assert F2["is_session_last"] == 1 and F2["inflight_frac"] == pytest.approx(0.256)
    assert F2["log_uncached"] == pytest.approx(np.log1p(744)) and F2["d_outstanding"] == 1


def test_logger_neutral_and_rows(plain, tmp_path):
    d, s = plain
    taus = {m: s[f"p99_{m}"] for m in c0.METRICS4}
    log = tmp_path / "log.jsonl"
    r = c0.log_window(SC, log, taus, s["cut"], every=15, workers=4, header={"t": 1}, max_points=3,
                      write_dir=tmp_path / "out")
    assert r["points"] == r["new"] == 3
    _same_outputs(d, tmp_path / "out")
    rows = [json.loads(x) for x in log.read_text().splitlines()[1:]]
    assert [x["j"] for x in rows] == [0, 15, 30]
    for x in rows:
        assert sorted(x["cands"], key=int) == [str(c) for c in range(SC.replicas)]
        assert x["req"]["arrived_at"] == x["t"] and x["req"]["position_in_burst"] == 0
        assert len(x["reps"]) == SC.replicas and "prefix_hashes" not in x["req"]
        assert len({t["n"] for t in x["cands"].values()}) == 1
    # a resume reuses the rows (no new scoring) and checks them
    r2 = c0.log_window(SC, log, taus, s["cut"], every=15, workers=4, header={"t": 1}, max_points=3)
    assert r2["new"] == 0


def _arrays_child(h, k):
    h.run_to_decision(k)
    after_os_fork(h)
    affected = hr._inflight(h)
    g = h.sim._request_generator
    made, orig = [], g.get_next_request

    def wrapped():
        r = orig()
        if r is not None:
            made.append(r)
        return r
    g.get_next_request = wrapped
    hr.apply_decision(h, None)
    h.run()
    h.write_output()
    return {k_: v.tolist() for k_, v in c0.request_arrays(h, affected + made).items()}


def test_request_arrays_match_outputs(tmp_path):
    h = build_simulator(hr.argv_for(SC, tmp_path))
    arr = hr._run_children([lambda: _arrays_child(h, 120)], 1, 600)[0]
    df = pd.read_csv(_csv(tmp_path)).set_index("Request Id").loc[arr["ids"]]
    # request_metrics.csv rounds to 6 decimals; the rollouts use the exact values
    assert np.allclose(arr["e2e"], df["request_e2e_time"], rtol=0, atol=5.1e-7)
    assert np.allclose(arr["ttft"], df["prefill_e2e_time"], rtol=0, atol=5.1e-7)
    m = df["request_num_decode_tokens"].to_numpy() > 1
    tp = (df["request_e2e_time"] - df["prefill_e2e_time"]).to_numpy()[m] / (df["request_num_decode_tokens"].to_numpy()[m] - 1)
    assert np.allclose(arr["tpot"], tp, rtol=0, atol=1.1e-6)
    gz = np.load(_csv(tmp_path).parent / "token_gaps.npz")
    keep = np.isin(gz["request_id"], arr["ids"])
    assert np.array_equal(np.sort(gz["gap"][keep]), np.sort(arr["gaps"])) and len(arr["gaps"]) > 0


def test_router_override_equals_forced_decisions(tmp_path):
    """The deployed router (base + a fixed test model) and a reference that recomputes the same features from the
    logger's own view capture and arrival history, then forces the choice through sim_fork: same outputs."""
    model = c0.LeastPendingAltModel()
    pol = c0.write_policy(tmp_path, model, 0.5, {})["policy"]
    h = build_simulator(hr.argv_for(SC, tmp_path / "A", policy=pol))
    h.run()
    h.write_output()
    st = h.sim._scheduler._ctx.state["c0_stats"]
    c0.install_capture()
    ref = build_simulator(hr.argv_for(SC, tmp_path / "B"))
    hist, n_dec, n_ov = [], 0, 0
    while ref.next_decision():
        q = ref.sim._scheduler._request_queue
        t = q[0].arrived_at
        hf = cf.hist_features(hist, t)
        hist.append((t, len(q)))
        cf.prune_hist(hist, t)
        pe, req, reps = c0.capture_view(ref)
        rows = cf.features(req, reps, pe, hf)
        alts = [c for c in range(SC.replicas) if c != pe]
        pred = model.predict(np.asarray([rows[c] for c in alts]))
        choice = alts[int(np.argmax(pred))] if pred.max() > 0.5 else pe
        n_dec += 1
        n_ov += choice != pe
        hr.apply_decision(ref, choice)
    ref.write_output()
    assert st == {"decisions": n_dec, "overrides": n_ov} and 10 < n_ov < n_dec
    _same_outputs(tmp_path / "A", tmp_path / "B")


def _synthetic(signal: float, seed=0, D=600, R=8, n_ranges=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(D, R, len(cf.FEATURES)))
    G = signal * X[:, :, 0] + rng.normal(0, 200, size=(D, R))
    base = rng.integers(0, R, D)
    G[np.arange(D), base] = 0.0
    ranges = np.array([f"{'Tool' if i % 2 else 'Conv'}:{i}" for i in range(n_ranges)])[np.arange(D) % n_ranges]
    return {"X": X, "G": G, "range": ranges, "trace": np.array([r.split(":")[0] for r in ranges]), "base": base}


def test_loro_finds_planted_gain_and_rejects_noise():
    ds = _synthetic(400.0)
    res = c0.items12(ds, c0.loro(ds), n_boot=300)
    assert res["item1"]["ok"] and res["item2"]["ok"], res
    assert res["item1"]["overrides"] >= c0.MIN_OVERRIDES
    null = c0.items12(_synthetic(0.0, seed=1), c0.loro(_synthetic(0.0, seed=1)), n_boot=300)
    assert not null["item1"]["ok"]


def _run(scale, rng=None, sd=0.0):
    base = {"p99_e2e": 100.0, "p99_ttft": 40.0, "p99_tpot": 0.08, "p99_tbt": 0.2, "mean_e2e": 30.0, "p50_e2e": 20.0}
    out = {}
    for k, v in base.items():
        f = scale.get(k, 1.0) if isinstance(scale, dict) else scale
        out[k] = v * f * (float(np.exp(rng.normal(0, sd))) if rng is not None else 1.0)
    return out


def test_item3_verdict():
    rng = np.random.default_rng(0)
    tags = sorted(sc.tag for sc in c0.heldout_c0())
    assert len(tags) == 6

    def res(new_scale):
        return {t: {"new": _run(new_scale), "base": _run(1.0), "noise": [_run(1.0, rng, 0.01) | {"seed": i} for i in range(20)]}
                for t in tags}
    good = c0.item3_verdict(res(0.95))
    assert good["ok"] and good["better"]["geo_ratio"] == pytest.approx(0.95)
    bad = c0.item3_verdict(res({"p99_e2e": 0.9, "p99_ttft": 0.9, "p99_tpot": 0.9, "p99_tbt": 0.9, "p50_e2e": 1.1}))
    assert bad["better"]["ok"] and not bad["not_worse"]["p50_e2e"]["ok"] and not bad["ok"]
    assert c0.item3_verdict({tags[0]: {"new": _run(1.0), "base": _run(1.0), "noise": []}})["status"] == \
        "invalid_scenarios"
    part = res(0.95)
    part[tags[0]]["noise"] = part[tags[0]]["noise"][:19]
    assert c0.item3_verdict(part)["status"] == "incomplete"
    dup = res(0.95)
    dup[tags[0]]["noise"][3]["seed"] = 4  # a repeated seed: incomplete, never a smaller noise spread
    assert c0.item3_verdict(dup)["status"] == "incomplete"


def test_windows_and_log_refuses_without_bar(tmp_path):
    w0, wb = c0.windows("c0"), c0.windows("c0b")
    assert len(w0) == 24 and len(wb) == 34 and {w.tag for w in w0} <= {w.tag for w in wb}
    assert len({c0.range_id(w.tag) for w in w0}) == 12 and len({c0.range_id(w.tag) for w in wb}) == 17
    with pytest.raises(SystemExit, match="no bar run|taus"):
        c0.main(["--mode", "log", "--window", "Tool:0:20", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="not a c0 window"):
        c0.main(["--mode", "log", "--window", "Tool:2000:20", "--out", str(tmp_path)])


def test_logged_rows_equal_deployed_router_features(plain, tmp_path):
    """The logger's own rows, read back as ``fit`` reads them, give the feature rows the deployed router computes
    at the same decisions (margin +inf: the router follows 4873, so both walk the same path)."""
    d, s = plain
    taus = {m: s[f"p99_{m}"] for m in c0.METRICS4}
    log = tmp_path / "log.jsonl"
    c0.log_window(SC, log, taus, s["cut"], every=7, workers=4, header={"t": 1}, max_points=4)
    rows = [json.loads(x) for x in log.read_text().splitlines()[1:]]
    c0.FeatureRecorder.seen = []
    pol = c0.write_policy(tmp_path, c0.FeatureRecorder(), float("inf"), {})["policy"]
    h = build_simulator(hr.argv_for(SC, tmp_path / "A", policy=pol))
    h.run()
    seen = c0.FeatureRecorder.seen
    assert len(seen) == h.sim._scheduler._ctx.state["c0_stats"]["decisions"]
    for r in rows:
        F = cf.features(r["req"], r["reps"], r["pe"], r["hist"])
        alts = [c for c in range(SC.replicas) if c != r["pe"]]
        assert np.array_equal(np.asarray([F[c] for c in alts]), seen[r["k"] - 1])


def _folds(gains_by_range, preds_by_range=None, override_all=True):
    folds = []
    for i, g in enumerate(gains_by_range):
        g = np.asarray(g, float)
        pr = np.asarray(preds_by_range[i], float) if preds_by_range else g.copy()
        folds.append({"range": f"Tool:{i}", "pred": pr, "gain": g,
                      "override": np.full(len(g), override_all),
                      "best": np.zeros(len(g), int), "alt_pred": pr, "alt_gain": g, "margin": 0.0})
    return {"folds": folds}


def test_item1_overrides_in_one_range_do_not_pass():
    """Review r1 B1: a resample without overrides counts as no gain, so 40 overrides from one range cannot pass."""
    rng = np.random.default_rng(0)
    lo = _folds([rng.normal(500, 50, 40)] + [rng.normal(0, 1, 30) for _ in range(11)])
    for f in lo["folds"][1:]:
        f["override"][:] = False
    ds = {"G": np.zeros((10, 8))}
    r = c0.items12(ds, lo, n_boot=2000)
    assert r["item1"]["overrides"] == 40 and r["item1"]["resamples_without_overrides"] > 0
    assert not r["item1"]["ok"]


def test_item2_needs_within_range_skill():
    """Review r1 M4: predictions and gains sharing only a per-range offset pass the pooled Spearman but not item 2."""
    rng = np.random.default_rng(1)
    gains, preds = [], []
    for i in range(12):
        off = 100.0 * i
        gains.append(off + rng.normal(0, 50, 60))
        preds.append(off + rng.normal(0, 50, 60))
    r = c0.items12({"G": np.zeros((10, 8))}, _folds(gains, preds), n_boot=2000)
    assert r["item2"]["spearman_pooled"] > 0.5 and not r["item2"]["ok"]
    skill = [g + rng.normal(0, 20, len(g)) for g in gains]
    assert c0.items12({"G": np.zeros((10, 8))}, _folds(gains, skill), n_boot=2000)["item2"]["ok"]


def test_check_windows_refuses_incomplete_logs(tmp_path):
    sc = c0.windows("c0")[0]
    d = tmp_path / sc.tag
    d.mkdir()
    head = {"code_version": ev.code_version(), "decision_sha": c0.c0_sha(), "scenario": sc.tag}
    (d / "log.jsonl").write_text(json.dumps({"header": head}) + "\n" + json.dumps({"k": 5, "j": 0}) + "\n")
    summ = {"meta": {}, "taus": {}, "w": 1.0, "s_tbt": 1.0}
    (d / "summary.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit, match="not complete"):
        c0.check_windows({sc.tag: d}, 6)
    (d / "summary.json").write_text(json.dumps(summ | {"complete": {"6": {"points": 2, "header": head}}}))
    with pytest.raises(SystemExit, match="1 rows"):
        c0.check_windows({sc.tag: d}, 6)
    (d / "summary.json").write_text(json.dumps(summ | {"complete": {"6": {"points": 1, "header": head}}}))
    assert len(c0.check_windows({sc.tag: d}, 6)[sc.tag]["rows"]) == 1
    with pytest.raises(SystemExit, match="not logged"):
        c0.check_windows({sc.tag: d, "x": tmp_path / "missing"}, 6)
