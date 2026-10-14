"""Closed-loop headroom pilot (review G r1, finding 8):
- neutrality: always keeping pi_e's choice through the fork/forcing path reproduces the plain run byte for byte, and
  score_output equals evaluate.run_one on the same window;
- deviation-path exactness: forcing replica r at decision k (headroom's parent path) equals a run in which pi_e's own
  ``_select`` returned r at decision k (a true counterfactual: the policy booked its memory itself);
- parent consistency: after lookahead deviations (children forked first), the continued parent equals a fresh run
  with the same forced decision sequence;
- scoring on a constructed case with a known answer, censoring, the decision record;
- censoring probe (review G r2): a pure time cutoff at T = tau_e cannot see the routed request's or later arrivals'
  E2E tail; T+drain censors nothing and bounds the censored scores from above;
- one child serving several horizons (grandchild drains) scores each exactly as a single-horizon rollout;
- open-loop horizon selection runs and resumes; an interrupted closed loop resumes to the uninterrupted result;
- resampled output lengths leak no true length; fork failures are reported and every child is reaped;
- review G r4: arrival jitter keeps every request's fields together; noise runs and the per-point floor perturb the
  run; the tail-weight sweep and its pre-specified choice; the verdict filters controls by code sha and noise kind and
  the head by weight; per-mode decision shas; block bootstrap; the cross-window claim;
- review G r5: the jitter changes routing, not just arrival stamps; world-B (cross-fit) rollouts; the cross-fitted
  weight rule (strict improvement, minimum deviation rate, rule_met); trade / noise-model / pace-guard labels; the
  pooled statistic; the pilot verdict recomputes labels and checks the windows; the tails tie-break;
- review G r6: the weight rule and the cross-fitted gain are mechanisms, not gates (synthetic true-null pilots reach
  "no headroom", true positives "headroom"); data-based bounds; centred noise band; the below-band-and-worse case is
  not negative; one setup across windows;
- step B: K sampled futures are common to all candidates and averaged, K = 1 reproduces the single-future sampled
  closed loop, the K-future loop resumes; transferred thresholds and weight are used and own-window settings refused;
  the pilot's shas and row keys are unchanged; the gate's clauses on synthetic windows.
All on Conv 4R s1.5 w6000+200 p1 (~8 s per full run)."""

import filecmp
import json
import math
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.stats as st

from evolve import evaluate as ev
from evolve.rl_plan import headroom as hr
from evolve.sim_fork import build_simulator

SC = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1, model="l3_8b")
K = 120  # a decision after the warm-up


def _csv(d):
    return next(Path(d).rglob("request_metrics.csv"))


def _same_outputs(a, b):
    """request_metrics.csv byte-identical and the per-token gaps equal."""
    assert filecmp.cmp(_csv(a), _csv(b), shallow=False)
    ga, gb = np.load(_csv(a).parent / "token_gaps.npz"), np.load(_csv(b).parent / "token_gaps.npz")
    assert sorted(ga.files) == sorted(gb.files)
    for f in ga.files:
        assert np.array_equal(ga[f], gb[f])


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    d = tmp_path_factory.mktemp("plain")
    h = build_simulator(hr.argv_for(SC, d))
    h.run()
    h.write_output()
    return d, hr.score_output(d, SC)


def _plan(s, score="tail"):
    return hr.Plan(score, s["p99_e2e"], s["p99_ttft"], 25 * s["mean_e2e"], "elapsed", s["mean_tpot"])


def test_neutral_loop_and_score_output_matches_run_one(plain, tmp_path):
    d, s = plain
    assert s["ok"] and s["completed"] == s["n_expected"] == 200
    r = ev.run_one("evolved", SC, policy=hr.POLICY)
    assert r["ok"] and r["bundle"] == s["bundle"]
    for k, v in r.items():
        if k != "wall_s":
            assert s[k] == v or (v != v and s[k] != s[k]), k  # NaN == NaN
    hz = hr.Horizon.parse("T=10+drain")
    hr.closed_loop(SC, hz, _plan(s), s["cut"], tmp_path / "B", tmp_path / "b.jsonl", workers=2,
                   max_decisions=10, candidates="pe")
    _same_outputs(d, tmp_path / "B")
    rows = [json.loads(x) for x in open(tmp_path / "b.jsonl")][1:]  # after the header
    assert len(rows) == 10 and all(x["chosen"] == x["pe"] and x["decided_by"] == "keep" for x in rows)


def test_deviation_path_equals_policy_choosing_it(plain, tmp_path):
    d, _ = plain
    # pi_e's own choice at decision K (record-only), then a different replica r
    h = build_simulator(hr.argv_for(SC, tmp_path / "B"))
    assert h.run_to_decision(K)
    probe = h.fork()
    pe = hr.apply_decision(probe, None)["policy_choice"]
    r = (pe + 1) % SC.replicas
    rec = hr.apply_decision(h, r)  # headroom's parent path
    assert rec == {"policy_choice": pe, "executed": r}
    h.run()
    h.write_output()

    a = build_simulator(hr.argv_for(SC, tmp_path / "A"))
    pol = a.sim._scheduler._policy  # a fresh module per simulator
    orig, calls = pol._select, [0]

    def patched(*args):
        calls[0] += 1
        pick = orig(*args)  # same ctx.rng draws as the unpatched call
        return r if calls[0] == K else pick
    pol._select = patched
    a.run()
    a.write_output()
    assert calls[0] == 200
    _same_outputs(tmp_path / "A", tmp_path / "B")
    assert not filecmp.cmp(_csv(d), _csv(tmp_path / "B"), shallow=False)  # the deviation changed the run


def _worst(by, pe, plan):  # always deviate (to the worst-scoring other replica): exercises the parent path
    best, info = hr.select_candidate(by, pe, plan)
    other = max((c for c in by if c != pe), key=lambda c: info["scores"][c])
    return other, dict(info, decided_by="test")


def test_parent_continues_like_fresh_forced_run_and_resumes(plain, tmp_path):
    _, s = plain
    hz = hr.Horizon.parse("T=5+drain")
    info = hr.closed_loop(SC, hz, _plan(s), s["cut"], tmp_path / "L", tmp_path / "l.jsonl", workers=2,
                          max_decisions=3, chooser=_worst)
    assert info["planned"] == 3 and info["deviations"] == 3
    forced = {x["k"]: x["chosen"] for x in list(map(json.loads, open(tmp_path / "l.jsonl")))[1:]}

    f = build_simulator(hr.argv_for(SC, tmp_path / "F"))
    while f.next_decision():
        k = f.decisions + 1
        if k in forced:
            assert hr.apply_decision(f, forced[k])["executed"] == forced[k]
        else:
            f.run(lambda x, n=f.decisions: x.decisions > n)
    f.write_output()
    _same_outputs(tmp_path / "L", tmp_path / "F")

    # interrupted at the 3rd planned decision (after 2 were logged), then resumed: same as uninterrupted
    calls = [0]

    def crash(by, pe, plan):
        calls[0] += 1
        if calls[0] == 3:
            raise KeyboardInterrupt
        return _worst(by, pe, plan)
    log = tmp_path / "r.jsonl"
    with pytest.raises(KeyboardInterrupt):
        hr.closed_loop(SC, hz, _plan(s), s["cut"], tmp_path / "R1", log, workers=2, max_decisions=3, chooser=crash)
    with open(log, "a") as fh:
        fh.write('{"k": 99, "trunc')  # a partial line from a crash mid-write is dropped
    info = hr.closed_loop(SC, hz, _plan(s), s["cut"], tmp_path / "R2", log, workers=2, max_decisions=3,
                          chooser=_worst)
    assert info["replayed"] == 2 and info["planned"] == 3 and info["deviations"] == 3
    _same_outputs(tmp_path / "L", tmp_path / "R2")
    with pytest.raises(SystemExit, match="other parameters"):
        hr.closed_loop(SC, hz, _plan(s), s["cut"], tmp_path / "R3", log, workers=2, header={"horizon": "other"})


def test_censoring_probe(plain, tmp_path):
    """At T = tau_e with a pure cutoff, the routed request's and later arrivals' E2E hinge is identically 0; with
    +drain nothing is censored, the affected set is the same and every score term is >= the censored one."""
    _, s = plain
    plan = _plan(s)
    h = build_simulator(hr.argv_for(SC, tmp_path))
    assert h.run_to_decision(K)
    hz = [hr.Horizon("time", s["p99_e2e"], drain=False), hr.Horizon("time", s["p99_e2e"], drain=True)]
    res = hr.evaluate_candidates(h, list(range(SC.replicas)), hz, plan, workers=2)
    assert sum(r["h"][0]["n_unfinished"] for r in res) > 0  # the probe censors something
    for r in res:
        cen, dr = r["h"]
        assert cen["post_hinge"] == 0.0 and cen["routed_e2e"] <= s["p99_e2e"]
        assert dr["n"] == cen["n"] and dr["n_unfinished"] == 0 and not dr["routed_censored"]
        for k in ("sum", "hinge", "hinge_ttft", "post_hinge", "routed_e2e"):
            assert dr[k] >= cen[k] - 1e-9, k


def test_nested_horizons_equal_single_rollouts(plain, tmp_path):
    _, s = plain
    plan = _plan(s)
    h = build_simulator(hr.argv_for(SC, tmp_path))
    assert h.run_to_decision(K)
    hz = [hr.Horizon.parse(x) for x in ("H=8", "T=10+drain", "T=10", "full")]
    multi = hr.evaluate_candidates(h, [None, 1], hz, plan, workers=2)
    for j, x in enumerate(hz):
        single = hr.evaluate_candidates(h, [None, 1], [x], plan, workers=2)
        for a, b in zip(multi, single):
            ta, tb = dict(a["h"][j]), dict(b["h"][0])
            ta.pop("cpu_s"), tb.pop("cpu_s")
            cont = ta.pop("cont", None)
            assert ta == tb, x.label
            if x.mode != "full":  # the same requests, scored at the end of the full run
                assert cont["n"] == ta["n"] and cont["n_unfinished"] == 0
    assert multi[0]["h"][3]["n"] > multi[0]["h"][1]["n"]  # full covers more arrivals than 10 s


def test_open_loop_selection_and_resume(plain, tmp_path):
    _, s = plain
    plan = _plan(s)
    hz = [hr.Horizon.parse(x) for x in ("H=8", "T=10+drain", "T=1000+drain")]  # the last reaches the window end
    ref = hr.Horizon.parse("full")
    log = tmp_path / "o.jsonl"
    st = hr.open_loop(SC, hz, ref, plan, s["cut"], log, workers=2, every=20, max_points=2, mean_e2e=s["mean_e2e"])
    labels = ("H8", "T10s+drain", "T1000s+drain")
    assert st["_run"]["points"] == 2 and set(st) == set(labels) | {"full", "paired_informative", "_run"}
    for x in labels[:2]:
        inf = st[x]["informative"]
        assert inf["points"] == 2 and st[x]["reached_end"]["points"] == 0
        assert 0 <= inf["pair_sign_agree"]["mean"] <= 1 and inf["cpu_s"]["mean"] > 0
        assert inf["ref_gain"]["lo"] <= inf["ref_gain"]["mean"] <= inf["ref_gain"]["hi"]
        assert inf["regret"]["mean"] >= 0 and "spearman_vs_cont" in inf
    end = st["T1000s+drain"]  # identical to full by construction: all in reached_end, perfect agreement
    assert end["informative"]["points"] == 0 and end["reached_end"]["points"] == 2
    assert end["reached_end"]["top1"]["mean"] == 1.0 and end["reached_end"]["regret"]["mean"] == 0.0
    assert end["reached_end"]["ref_gain"]["mean"] == pytest.approx(st["full"]["one_step_gain"]["mean"])
    assert st["full"]["one_step_gain"]["n"] == 2 and st["full"]["noise_points"] == 2
    assert st["paired_informative"]["H8 - T10s+drain"]["regret"]["n"] == 2
    assert set(st["full"]["weights"]) == {"25m", "100m", "400m", "tails"}
    assert st["full"]["weight_choice"]["label"] in st["full"]["weights"]
    rows = [json.loads(x) for x in log.read_text().splitlines()[1:]]
    assert all(r["noise"]["perturbed"].startswith("arrivals") for r in rows)
    # the jitter floor perturbs the full rollout (the r3 tiebreak floor was identically 0 on Tool)
    # ... by more than the arrival shift alone can (each E2E moves by <= d when nothing else changes): routing changed
    assert any(abs(r["noise"]["delta"]["sum"]) > 2 * hr.NOISE_DELTA * r["h"]["full"][str(r["pe"])]["n"] for r in rows)
    for r in rows:  # world B: every candidate, the same jitter, pi_e's choice = the floor rollout
        assert sorted(map(int, r["xfit"])) == list(range(SC.replicas))
        assert r["xfit"][str(r["pe"])]["sum"] == pytest.approx(r["h"]["full"][str(r["pe"])]["sum"]
                                                               + r["noise"]["delta"]["sum"])
    assert st["full"]["xfit_points"] == 2 and st["full"]["one_step_gain_xfit"]["n"] == 2
    assert st["full"]["weight_choice"]["rule_met"] in (True, False) and "gain_xfit" in st["full"]["weight_choice"]
    n_lines = len(log.read_text().splitlines())
    st2 = hr.open_loop(SC, hz, ref, plan, s["cut"], log, workers=2, every=20, max_points=2,
                       mean_e2e=s["mean_e2e"])  # resume: no new rows
    assert len(log.read_text().splitlines()) == n_lines
    assert {k: v for k, v in st2.items() if k != "_run"} == {k: v for k, v in st.items() if k != "_run"}


def _req(arr, done=None, prefill=None, total=10, processed=0, prefill_tokens=5):
    return SimpleNamespace(id=0, arrived_at=arr, _completed=done is not None, _completed_at=done or 0.0,
                           _prefill_completed_at=prefill or 0.0, total_tokens=total, num_processed_tokens=processed,
                           num_prefill_tokens=prefill_tokens)


def test_scoring_constructed_case():
    # censoring at cutoff 100: finished, unfinished after prefill, unfinished before prefill
    reqs = [_req(10, done=30, prefill=12), _req(50, prefill=60, total=10, processed=7), _req(90)]
    e2e, ttft, fin = hr.request_costs(reqs, 100.0)
    assert e2e.tolist() == [20, 50, 10] and ttft.tolist() == [2, 10, 10] and fin.tolist() == [True, False, False]
    e2e_r, _, _ = hr.request_costs(reqs, 100.0, "remaining", tpot_ref=0.5)
    assert e2e_r.tolist() == [20, 50 + 3 * 0.5, 10 + 5 * 0.5]  # 10 - max(7, 5) and 10 - max(0, 5) decode tokens left
    with pytest.raises(RuntimeError):
        hr.request_costs(reqs, None)
    t = hr.rollout_terms(np.array([20.0, 50.0]), np.array([2.0, 30.0]), tau_e=40.0, tau_f=20.0)
    assert t == {"sum": 70.0, "hinge": 0.25, "hinge_ttft": 0.5}

    # candidate 0 = pi_e. 1: lower summed E2E, worse tail. 2: higher sum, better tail.
    by = {0: {"sum": 100.0, "hinge": 1.0, "hinge_ttft": 0.0},
          1: {"sum": 90.0, "hinge": 1.5, "hinge_ttft": 0.0},
          2: {"sum": 104.0, "hinge": 0.5, "hinge_ttft": 0.0}}
    plan = hr.Plan("tail", weight=10.0)  # scores: 110, 105, 109
    best, info = hr.select_candidate(by, 0, plan)
    assert best == 1 and info["decided_by"] == "mean" and info["gain"] == pytest.approx(5.0)
    assert info["term_gain"] == pytest.approx({"mean": 10.0, "e2e_tail": -5.0, "ttft_tail": 0.0})
    best, info = hr.select_candidate(by, 0, hr.Plan("tail", weight=30.0))  # scores: 130, 135, 119
    assert best == 2 and info["decided_by"] == "e2e_tail" and info["gain"] == pytest.approx(11.0)
    assert hr.select_candidate(by, 0, hr.Plan("tail", weight=30.0, min_gain=11.5))[0] == 0  # gain too small
    assert hr.select_candidate(by, 0, hr.Plan("lex"))[0] == 2
    assert hr.select_candidate(by, 0, hr.Plan("sum"))[0] == 1
    best, info = hr.select_candidate({3: dict(by[0]), 0: by[0]}, 0, plan)  # a tie keeps pi_e's choice
    assert best == 0 and info["decided_by"] == "keep" and info["gain"] == 0.0


def test_horizon_parse():
    assert hr.Horizon.parse("T=1x").resolve(95.0) == hr.Horizon("time", 95.0, drain=False)
    assert hr.Horizon.parse("T=1x+drain").resolve(95.0) == hr.Horizon("time", 95.0, drain=True)
    assert hr.Horizon.parse("T=1x+drain").label == "T1x+drain" and hr.Horizon.parse("T=20").label == "T20s"
    assert hr.Horizon.parse("H=64") == hr.Horizon("decisions", 64)
    assert hr.Horizon.parse("full").label == "full"
    for bad in ("T=0", "H=0", "X=3", "H=1.5", "T=", "T=+drain", "T=inf", "T=ax"):
        with pytest.raises(ValueError):
            hr.Horizon.parse(bad)


def test_resampled_lengths_hide_true_lengths(tmp_path):
    h = build_simulator(hr.argv_for(SC, tmp_path))
    assert h.run_to_decision(K)
    f = h.fork()
    pool, cap = hr.output_pool(SC), ev.max_request_tokens(SC)
    assert len(pool) > 1000 and cap == 131072
    infl = hr._inflight(f)
    gen = [hr.generated_tokens(r) for r in infl]
    g = f.sim._request_generator
    i = g.num_requests_generated
    true_future = np.array(g._num_decode_tokens[i:], copy=True)
    hr.resample_outputs(f, np.random.default_rng(0), pool, cap)
    assert len(infl) > 10
    pool_set = set(pool.tolist())
    for r, done in zip(infl, gen):
        L = r.original_num_decode_tokens
        assert L > done and r.original_num_prefill_tokens + L <= cap
        assert r.total_tokens == r.original_num_prefill_tokens + L and r.total_tokens > r.num_processed_tokens
        assert L in pool_set or L == done + 1
    new_future = np.asarray(g._num_decode_tokens[i:])
    assert not np.array_equal(new_future, true_future)
    assert np.all(np.asarray(g._num_prefill_tokens[i:]) + new_future <= cap)
    # block-hash lists fit the new lengths (Request asserts it for arrivals; the KV cache needs every full block)
    for r in infl:
        assert len(r._block_hash_ids) == r.total_tokens // r._block_size
    for j in range(i, g.num_requests):
        assert len(g._block_hash_ids[j]) == (g._num_prefill_tokens[j] + new_future[j - i]) // g._block_size[j]
    # the parent's requests and trace are untouched
    assert [r.original_num_decode_tokens for r in hr._inflight(h)] != [r.original_num_decode_tokens for r in infl]
    assert np.array_equal(h.sim._request_generator._num_decode_tokens[i:], true_future)
    # end to end: resampled rollouts run to completion under every horizon kind
    plan = hr.Plan("tail", 90.0, 60.0, 250.0)
    hz = [hr.Horizon.parse(x) for x in ("H=8", "T=10+drain", "T=10", "full")]
    res = hr.evaluate_candidates(h, [None, 1], hz, plan, workers=2, sample=(7, pool, cap))
    assert all(r["h"][3]["n_unfinished"] == 0 and "cont" in r["h"][1] for r in res)
    true = hr.evaluate_candidates(h, [None], hz, plan, workers=1)
    assert true[0]["h"][3]["sum"] != res[0]["h"][3]["sum"]  # the lengths did change the rollout


def test_fit_hashes():
    assert hr.fit_hashes([1, 2, 3], 40, 16, 5) == [1, 2]
    assert hr.fit_hashes([1, 2], 70, 16, 5) == [1, 2, -5000003, -5000004]


def _no_children():
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


def test_fork_failures_reported_and_reaped():
    def fail():
        raise ValueError("boom")

    def ok():
        return {"x": 1}

    def hang():
        time.sleep(60)

    with pytest.raises(RuntimeError, match="killed by signal 9"):
        hr._collect_all([hr._fork_child(lambda: os.kill(os.getpid(), signal.SIGKILL)), hr._fork_child(hang)], 30)
    _no_children()
    with pytest.raises(RuntimeError, match="(?s)exit status 0.*ValueError: boom"):
        hr._collect_all([hr._fork_child(fail), hr._fork_child(hang)], 30)
    _no_children()
    t = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        hr._collect_all([hr._fork_child(ok), hr._fork_child(hang)], 1.0)
    assert time.time() - t < 10
    _no_children()
    assert hr._collect_all([hr._fork_child(ok), hr._fork_child(ok)], 30) == [{"x": 1}, {"x": 1}]
    _no_children()


def test_summary_reuse_refuses_other_setups(tmp_path):
    meta = {"scenario": SC.tag, "policy_sha": "a", "code_version": ev.code_version(), "bundle": "b"}
    p = tmp_path / "summary.json"
    assert hr.load_summary(p, meta)["rows"] == {}
    hr._atomic_json(p, {"format": hr.FORMAT, "meta": meta, "rows": {"pi_e": {"p99_e2e": 1.0}}})
    assert hr.load_summary(p, dict(meta))["rows"]["pi_e"] == {"p99_e2e": 1.0}
    for k in meta:
        with pytest.raises(SystemExit, match=k):
            hr.load_summary(p, dict(meta, **{k: "other"}))
    p.write_text(json.dumps({"scenario": SC.tag, "pi_e": {}}))  # r1 format: no provenance
    with pytest.raises(SystemExit, match="older"):
        hr.load_summary(p, meta)
    # lookahead rows are keyed by every parameter
    base = {"horizon": "T1x", "score": "tail", "future": "true", "max_decisions": None, "weight": 1.0}
    assert hr._row_key(base) != hr._row_key(dict(base, max_decisions=5)) != hr._row_key(dict(base, weight=2.0))
    assert hr._row_key(dict(base, decision_sha="a")) != hr._row_key(dict(base, decision_sha="b"))
    assert hr.decision_sha() == hr.decision_sha("closed") and hr.decision_sha().endswith(hr.SIM_FORK_SHA)
    shas = {m: hr.decision_sha(m) for m in ("closed", "openloop", "placebo", "noise")}
    assert len(set(shas.values())) == 4 and all(v.endswith(hr.SIM_FORK_SHA) for v in shas.values())
    assert hr.decision_sha().split(":")[0] != hr.post_sha()
    with pytest.raises(ValueError):
        hr.decision_sha("pilot")


def test_placebo_and_noise_runs(plain, tmp_path):
    d, s = plain
    hr.placebo_run(SC, 0.0, 0, s["cut"], tmp_path / "P0")  # rate 0: the forcing path is neutral
    _same_outputs(d, tmp_path / "P0")
    info = hr.placebo_run(SC, 0.5, 1, s["cut"], tmp_path / "P1")
    assert 0.3 < info["deviation_frac"] < 0.7 and hr.score_output(tmp_path / "P1", SC)["ok"]
    info = hr.noise_run(SC, 3, s["cut"], tmp_path / "N", hr.NOISE_DELTA)
    assert hr.score_output(tmp_path / "N", SC)["ok"] and info["jittered_from"] > 0
    sn = hr.score_output(tmp_path / "N", SC)  # more than new arrival stamps: a P99 moves by > 1e-3 (log)
    assert max(abs(math.log(sn[f"p99_{m}"] / s[f"p99_{m}"])) for m in ("e2e", "ttft")) > 1e-3
    hr.noise_run(SC, 3, s["cut"], tmp_path / "N2", hr.NOISE_DELTA)  # seeded: reproducible
    _same_outputs(tmp_path / "N", tmp_path / "N2")


def test_jitter_arrivals():
    n = 50
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1.0, n))
    g = SimpleNamespace(num_requests=n - 5, num_requests_generated=0, _arrived_at=t.copy(),
                        _num_prefill_tokens=np.arange(n) * 10, _num_decode_tokens=np.arange(n) * 100,
                        _block_hash_ids=[[j] for j in range(n)], _block_size=[16] * n, _session_id=list(range(n)),
                        _prompt_extra=[f"r{j}" for j in range(n)], _other=[1, 2], name="x")
    before = {k: v for k, v in vars(g).items()}
    hr.jitter_arrivals(g, 10, float(t[9]), 0.05, np.random.default_rng(0))
    a = g._arrived_at
    assert np.array_equal(a[:10], t[:10]) and np.array_equal(a[n - 5:], t[n - 5:])  # outside [i0, num_requests)
    assert np.all(np.diff(a[:n - 5]) >= 0) and a[10:n - 5].min() >= t[9]
    assert not np.array_equal(a, t)
    ids = [h[0] for h in g._block_hash_ids]  # every field moved with its request
    assert sorted(ids) == list(range(n)) and ids != list(range(n))
    assert g._num_prefill_tokens.tolist() == [10 * j for j in ids] and g._num_decode_tokens.tolist() == [
        100 * j for j in ids]
    assert g._session_id == ids and g._prompt_extra == [f"r{j}" for j in ids] and g._other == [1, 2]
    assert all(abs(a[k] - t[j]) <= 0.05 or a[k] == t[9] for k, j in enumerate(ids))
    assert np.array_equal(before["_arrived_at"], t) and before["_block_hash_ids"][10] == [10]  # replaced, not edited


def _vs(e2e, ttft, tpot=0.0, tbt=0.0):
    return {"log_p99_e2e": e2e, "log_p99_ttft": ttft, "log_p99_tpot": tpot, "log_p99_tbt": tbt}


SHAS = {"closed": "c1", "placebo": "p1", "noise": "n1"}


def _ctl(mode, vs, sha=None, **kw):
    p = {"mode": mode, "decision_sha": sha or SHAS[mode]} | kw
    return {"params": p, "vs_pi_e": vs}


def _controls(rng, sd=0.02, n_off=0.0, p_sd=None):
    base = {f"placebo{i}": _ctl("placebo", _vs(*rng.normal(0, p_sd or sd, 4)), rate=0.2) for i in range(20)}
    base |= {f"noise{i}": _ctl("noise", _vs(*(rng.normal(0, sd, 4) + n_off)), noise_kind="jitter", noise_delta=0.005)
             for i in range(20)}
    return base


HEAD = {"params": {"mode": "closed", "horizon": "full", "plan_every": 1, "max_decisions": None, "future": "true",
                   "score": "tail", "weight": 10.0, "decision_sha": "c1"}, "deviation_frac": 0.2}


def test_window_verdict_rule():
    base = _controls(np.random.default_rng(0))
    head = HEAD
    V = lambda rows, **kw: hr.window_verdict(rows, SHAS, 0.005, **kw)  # noqa: E731
    lab = lambda *vs: V(base | {"L": dict(head, vs_pi_e=_vs(*vs))})["label"]  # noqa: E731
    assert V(base)["label"] == "missing"
    assert lab(-0.2, -0.2) == "positive"
    assert lab(-0.2, -0.2, 0.0, 0.3) == "trade"  # tails bought with token pace
    assert lab(-0.3, 0.1) == "trade" and lab(0.2, -0.2) == "trade"  # review G r5: not "negative"
    assert lab(0.2, 0.0) == "negative" and lab(0.2, 0.2) == "negative"
    assert lab(0.0, 0.0) == "null"
    assert lab(-0.2, 0.0) == "inconclusive"
    v = V(base | {"L": dict(head, vs_pi_e=_vs(-0.2, -0.2))})
    assert v["band"]["e2e"]["noise_p2.5"] < 0 < v["band"]["e2e"]["noise_p97.5"] and v["detectable_pct"]["e2e"] < 0
    assert v["clauses"]["better"] == {"e2e": True, "ttft": True} and v["clauses"]["pace_ok"]["tpot"]
    # pace guard relative to the band: a TPOT ratio at the band's upper edge passes, above it fails
    edge = v["clauses"]["pace_limit"]["tpot"]
    assert lab(-0.2, -0.2, edge - 1e-6) == "positive" and lab(-0.2, -0.2, edge + 1e-3) == "trade"
    # data-based bound: L - P97.5(reference), not measured from 0; the reference is the 380 pairwise differences
    v0 = V(base | {"L": dict(head, vs_pi_e=_vs(0.0, 0.0))})
    assert v0["n_noise"] == 20 and v0["n_reference"] == 380 and v0["noise_centre"] == "pairwise"
    assert v0["ruled_out"]["e2e"]["log"] == pytest.approx(-v0["band"]["e2e"]["noise_p97.5"])
    assert v0["ruled_out"]["e2e"]["pct"] < 0 and v0["detectable_pct"]["e2e"] < 0
    # review G r6/r7: the jitter's offset cancels in the pairwise reference; L = 0 stays null, L = offset is worse
    off = _controls(np.random.default_rng(0), n_off=0.1)
    vo = hr.window_verdict(off | {"L": dict(head, vs_pi_e=_vs(0.0, 0.0))}, SHAS, 0.005)
    assert vo["label"] == "null" and vo["noise_offset"]["e2e"] == pytest.approx(0.1, abs=0.02)
    assert vo["zero_in_raw_noise_band"] == {"e2e": False, "ttft": False}
    assert hr.window_verdict(off | {"L": dict(head, vs_pi_e=_vs(0.1, 0.1))}, SHAS, 0.005)["label"] == "negative"
    # review G r6 finding 4: below the noise band but within the placebo's reach, the other tail worse: not negative
    wide = _controls(np.random.default_rng(0), p_sd=0.06)
    vw = hr.window_verdict(wide | {"L": dict(head, vs_pi_e=_vs(-0.07, 0.1))}, SHAS, 0.005)
    assert vw["clauses"]["below_noise"]["e2e"] and not vw["clauses"]["better"]["e2e"]
    assert vw["label"] == "inconclusive"
    few = {k: v for k, v in base.items() if k != "noise0"}
    assert V(few | {"L": dict(head, vs_pi_e=_vs(-0.2, -0.2))})["label"] == "missing"
    # review G r4 finding 4: other noise kinds and stale-code controls are never pooled
    L = {"L": dict(head, vs_pi_e=_vs(-0.03, -0.03))}
    ref = V(base | L)
    other = {f"tok{i}": _ctl("noise", _vs(0, 0), noise_kind="token") for i in range(20)}
    stale = {f"old{i}": _ctl("placebo", _vs(-0.2, -0.2), sha="p0", rate=0.2) for i in range(5)}
    got = V(base | L | other | stale)
    assert got["label"] == ref["label"] and got["n_noise"] == 20 and got["n_placebo"] == 20
    assert got["stale_controls"] == 5 and got["noise_by_kind"] == {"jitter:0.005": 20, "token:None": 20}
    assert hr.window_verdict(base | L | other, SHAS)["label"] == "missing"  # two kinds, none named
    stale_head = {"L0": dict(head, params=dict(head["params"], decision_sha="c0"), vs_pi_e=_vs(-0.2, -0.2))}
    assert V(base | L | stale_head)["label"] == ref["label"]  # a closed row of older code is not a head
    # finding 3: two weight arms; the verdict takes the pre-specified one
    arm = {"L2": dict(head, params=dict(head["params"], weight=40.0), vs_pi_e=_vs(-0.2, -0.2))}
    assert V(base | L | arm)["label"] == "missing"
    assert V(base | L | arm, weight=40.0)["label"] == "positive" and V(base | L | arm, weight=10.0) == ref


def _xrow(k, a, b, pe=0):
    t = lambda x: {"sum": x[0], "hinge": x[1], "hinge_ttft": x[2]}  # noqa: E731
    return {"k": k, "pe": pe, "h": {"full": {str(c): t(v) for c, v in a.items()}},
            "xfit": {str(c): t(v) for c, v in b.items()} if b is not None else None}


def test_weight_sweep_and_choice():
    # pi_e = 0. In A, c1 has a lower sum and a worse E2E hinge (argmin at 25m); c2 the reverse (argmin from 100m)
    A = {0: (100.0, 1.0, 0.5), 1: (90.0, 1.2, 0.5), 2: (103.0, 0.9, 0.5)}
    B = {0: (100.0, 1.0, 0.5), 1: (95.0, 1.1, 0.5), 2: (104.0, 0.8, 0.5)}
    rows = [_xrow(k, A, B) for k in range(1, 11)]
    sw = hr.weight_sweep(rows, "full", mean_e2e=1.0)
    g = sw["grid"]
    assert g["25m"]["d_hinge"]["mean"] == pytest.approx(0.1) and g["25m"]["frac_hinge_worse"] == 1.0
    assert g["100m"]["d_hinge"]["hi"] == pytest.approx(-0.2) and g["100m"]["frac_hinge_better"] == 1.0
    assert g["100m"]["gain_xfit"]["mean"] == pytest.approx((100 + 150) - (104 + 130))  # cross-fitted, in B
    assert g["100m"]["gain_insample"]["mean"] == pytest.approx((100 + 150) - (103 + 140))
    assert sw["choice"]["label"] == "100m" and sw["choice"]["rule_met"] is True
    # review G r5 (a): a weight at which nothing deviates is never chosen (it used to pass trivially)
    A2 = {0: (100.0, 1.0, 0.5), 1: (120.0, 0.99, 0.5)}  # deviates only at tails
    B2 = {0: (100.0, 1.0, 0.5), 1: (121.0, 0.9, 0.5)}
    sw2 = hr.weight_sweep([_xrow(k, A2, B2) for k in range(1, 11)], "full", 1.0)
    assert sw2["grid"]["25m"]["deviates"]["mean"] == 0.0 and sw2["choice"]["label"] == "tails"
    assert sw2["choice"]["rule_met"] is True and sw2["choice"]["fallback"] is False
    # an improvement that is not strict (CI touching 0) or a worse other tail: rule not met, tails flagged
    B3 = {0: (100.0, 1.0, 0.5), 1: (95.0, 1.2, 0.5), 2: (104.0, 1.0, 0.4)}
    B4 = {0: (100.0, 1.0, 0.5), 1: (95.0, 1.2, 0.5), 2: (104.0, 1.0, 0.6)}
    sw3 = hr.weight_sweep([_xrow(k, A, B3 if k % 2 else B4) for k in range(1, 11)], "full", 1.0)
    assert sw3["choice"]["label"] == "tails" and sw3["choice"]["rule_met"] is False and sw3["choice"]["fallback"]
    assert hr.weight_sweep([_xrow(1, A, None)], "full", 1.0)["choice"] is None  # no world-B rollouts


def test_tails_tie_break():
    # equal hinges: the summed E2E decides even at the tails weight (review G r5 nit)
    by = {0: {"sum": 10000.0, "hinge": 50.0, "hinge_ttft": 20.0}, 1: {"sum": 9500.0, "hinge": 50.0, "hinge_ttft": 20.0}}
    assert hr.select_candidate(by, 0, hr.Plan("tail", weight=1e9 * 20.0))[0] == 1


def test_pooled_stat():
    rng = np.random.default_rng(3)
    Ns = [[_vs(*rng.normal(0, 0.02, 4)) for _ in range(20)] for _ in range(4)]
    null = hr.pooled_stat([(_vs(0.0, 0.0), N) for N in Ns])
    for m in ("e2e", "ttft"):
        assert null[m]["q5"] <= null[m]["Z"] <= null[m]["q95"] and null[m]["n_null"] == 20 ** 4
        assert -10 < null[m]["detectable_pct"] < 0 and null[m]["bound_pct"] < 0
    better = hr.pooled_stat([(_vs(-0.05, -0.05), N) for N in Ns])
    assert all(better[m]["Z"] < better[m]["q5"] for m in ("e2e", "ttft"))
    # review G r6: the bound depends on the observed Z (data-based), the detectable shift does not
    for d in (0.0, -0.01, -0.02):
        st = hr.pooled_stat([(_vs(d, d), N) for N in Ns])["e2e"]
        assert st["detectable_log"] == pytest.approx(null["e2e"]["detectable_log"])
        inv = (st["Z"] - st["q95"]) / st["bound_log"]
        assert st["bound_log"] == pytest.approx((st["Z"] - st["q95"]) / inv) and st["bound_log"] < d


CV = "cv-test"


def _window(tag, L, rule_met=True, xlo=-1.0, n_off=0.0, seed=0, psha=None, choice=None, meta=None, L_sd=None):
    rng = np.random.default_rng(seed)
    shas = {m: hr.decision_sha(m) for m in ("closed", "placebo", "noise")}
    rows = {f"placebo{i}": _ctl("placebo", _vs(*rng.normal(0, 0.02, 4)), sha=shas["placebo"], rate=0.2)
            for i in range(20)}
    rows |= {f"noise{i}": _ctl("noise", _vs(*(rng.normal(0, 0.02, 4) + n_off)), sha=shas["noise"],
                               noise_kind="jitter", noise_delta=hr.NOISE_DELTA) for i in range(20)}
    if L_sd is not None:  # the lookahead's own P99 lottery draw around its true change L
        L = tuple(np.asarray(L) + rng.normal(0, L_sd, 2))
    rows["L"] = dict(HEAD, params=dict(HEAD["params"], decision_sha=shas["closed"], weight=100.0), vs_pi_e=_vs(*L))
    ch = choice or {"label": "100m", "weight_m": 100.0, "weight": 100.0, "rule_met": rule_met,
                    "fallback": not rule_met, "gain_xfit": {"n": 10, "mean": xlo + 1, "lo": xlo, "hi": xlo + 2}}
    rows["L"]["params"]["weight"] = ch["weight"]
    rows["ol"] = {"params": {"mode": "openloop", "max_points": None, "future": "true",
                             "decision_sha": hr.decision_sha("openloop")},
                  "post_sha": psha or hr.post_sha(), "stats": {"full": {"weight_choice": ch}}}
    rows["verdict"] = {"label": "positive"}  # a stale stored label: never trusted
    return {"meta": {"scenario": tag, "code_version": CV, "bundle": "b"} | (meta or {}), "rows": rows}


@pytest.fixture
def cv(monkeypatch):
    monkeypatch.setattr(hr.ev, "code_version", lambda: CV)


def test_pilot_verdict(cv):
    tags = hr.pilot_windows()
    assert len(tags) == 4
    P = lambda ws: hr.pilot_verdict({t: w for t, w in zip(tags, ws)})  # noqa: E731
    nulls = [_window(t, (0.0, 0.0), seed=i) for i, t in enumerate(tags)]
    v = P(nulls)
    assert v["claim"] == "no headroom", v["reasons"]
    assert all(x < 0 for x in v["effect_size_ruled_out_pct"].values())
    assert all(w["mechanism"].startswith("(ii)") and w["ruled_out"]["e2e"]["pct"] < 0 for w in v["windows"].values())
    assert P([_window(t, (-0.2, -0.2), seed=i) for i, t in enumerate(tags)])["claim"] == "headroom"
    trade = nulls[:3] + [_window(tags[3], (-0.3, 0.15), seed=3)]
    assert P(trade)["claim"] == "inconclusive" and P(trade)["windows"][tags[3]]["label"] == "trade"
    # review G r6: rule_met False and a positive cross-fitted gain are mechanisms, not gates
    v = P(nulls[:3] + [_window(tags[3], (0.2, 0.0), rule_met=False, seed=3)])
    assert v["claim"] == "no headroom" and v["windows"][tags[3]]["mechanism"].startswith("(i)")
    v = P(nulls[:3] + [_window(tags[3], (0.0, 0.0), xlo=0.5, seed=3)])
    assert v["claim"] == "no headroom" and v["windows"][tags[3]]["gain_xfit"]["lo"] == 0.5
    stale = P(nulls[:3] + [_window(tags[3], (0.0, 0.0), seed=3, psha="old")])
    assert stale["claim"] == "inconclusive" and stale["windows"][tags[3]]["label"] == "stale"
    nox = _window(tags[3], (0.0, 0.0), seed=3)
    nox["rows"]["ol"]["stats"]["full"]["weight_choice"] = None
    assert P(nulls[:3] + [nox])["windows"][tags[3]]["label"] == "no_xfit"
    wrong = nulls[:3] + [_window(ev.Scenario("Tool", 8, 0.75, 4000, 2000, 1, model="l3_8b").tag, (0.0, 0.0))]
    assert hr.pilot_verdict({i: w for i, w in enumerate(wrong)})["claim"] == "invalid_windows"
    assert hr.pilot_verdict({i: w for i, w in enumerate(nulls[:3])})["claim"] == "invalid_windows"
    # one setup across windows, at the current code_version
    other = P(nulls[:3] + [_window(tags[3], (0.0, 0.0), seed=3, meta={"bundle": "other"})])
    assert other["claim"] == "invalid_windows" and "setup" in other["reason"]
    old = P([_window(t, (0.0, 0.0), seed=i, meta={"code_version": "old"}) for i, t in enumerate(tags)])
    assert old["claim"] == "invalid_windows" and "code_version" in old["reason"]


def _regime_rows(regime, seed, gain_h=1.0, cost_sum=30.0, R=8, npts=165):
    """Review G r6's synthetic open-loop rows: chaos per rollout sd 300 (sum), 1.5 (each hinge); regime 0: no candidate
    truly better than pi_e; regime 1: one candidate per point has a true E2E-hinge gain and a higher summed E2E."""
    rng = np.random.default_rng(seed)
    sd = {"sum": 300.0, "hinge": 1.5, "hinge_ttft": 1.5}
    out = []
    for k in range(npts):
        true = {c: {"sum": 1000.0, "hinge": 50.0, "hinge_ttft": 20.0} for c in range(R)}
        if regime == 1:
            true[1 + int(rng.integers(R - 1))] = {"sum": 1000.0 + cost_sum, "hinge": 50.0 - gain_h,
                                                  "hinge_ttft": 20.0}
        w = lambda: {str(c): {t: true[c][t] + rng.normal(0, sd[t]) for t in sd} for c in range(R)}  # noqa: E731
        out.append({"k": k, "pe": 0, "h": {"full": w()}, "xfit": w()})
    return out


SD = {"e2e": 0.022, "ttft": 0.041, "tpot": 0.02, "tbt": 0.02}  # one jittered pi_e draw, Tool w0 (G r6)


def _window_b(tag, delta, rng, choice=None, rho=0.1, h=0.022):
    """Review G r7's model B: a window log P99 = mu + g(arrivals, shared by policies) + c(trajectory chaos); pi_e's own
    unjittered run is one draw c0. Noise N_j = h + g_j + c_j - c0 (h: the jitter's harm), placebo P_j = c'_j - c0,
    lookahead L = delta + c_la - c0 on every metric (TPOT/TBT drawn too, no true pace change)."""
    sg = {m: SD[m] * math.sqrt(rho) for m in SD}
    sc = {m: SD[m] * math.sqrt(1 - rho) for m in SD}
    dl = {"e2e": delta[0], "ttft": delta[1], "tpot": 0.0, "tbt": 0.0}
    c0 = {m: rng.normal(0, sc[m]) for m in SD}
    vs = lambda d: {f"log_p99_{m}": float(d[m]) for m in SD}  # noqa: E731
    shas = {m: hr.decision_sha(m) for m in ("closed", "placebo", "noise")}
    rows = {f"placebo{i}": _ctl("placebo", vs({m: rng.normal(0, sc[m]) - c0[m] for m in SD}), sha=shas["placebo"],
                                rate=0.2) for i in range(20)}
    rows |= {f"noise{i}": _ctl("noise", vs({m: h + rng.normal(0, sg[m]) + rng.normal(0, sc[m]) - c0[m] for m in SD}),
                               sha=shas["noise"], noise_kind="jitter", noise_delta=hr.NOISE_DELTA) for i in range(20)}
    ch = choice or {"label": "100m", "weight_m": 100.0, "weight": 100.0, "rule_met": True, "fallback": False,
                    "gain_xfit": {"n": 10, "mean": 0.0, "lo": -1.0, "hi": 1.0}}
    rows["L"] = dict(HEAD, params=dict(HEAD["params"], decision_sha=shas["closed"], weight=ch["weight"]),
                     vs_pi_e=vs({m: dl[m] + rng.normal(0, sc[m]) - c0[m] for m in SD}))
    rows["ol"] = {"params": {"mode": "openloop", "max_points": None, "future": "true",
                             "decision_sha": hr.decision_sha("openloop")},
                  "post_sha": hr.post_sha(), "stats": {"full": {"weight_choice": ch}}}
    return {"meta": {"scenario": tag, "code_version": CV, "bundle": "b"}, "rows": rows}


def _oc(delta, n, seed, choices=None):
    """Claims, window labels and E2E bound coverage over n synthetic pilots (model B)."""
    tags = hr.pilot_windows()
    rng = np.random.default_rng(seed)
    claims, labels, cov_w, cov_p = [], [], [], []
    for _ in range(n):
        v = hr.pilot_verdict({t: _window_b(t, delta, rng, (choices or [None] * 4)[i]) for i, t in enumerate(tags)})
        claims.append(v["claim"])
        labels += [w["label"] for w in v["windows"].values()]
        cov_w += [w["ruled_out"]["e2e"]["log"] <= delta[0] for w in v["windows"].values()]
        cov_p.append(v["pooled"]["e2e"]["bound_log"] <= delta[0])
    return claims, labels, float(np.mean(cov_w)), float(np.mean(cov_p))


def test_synthetic_operating_characteristic(cv):
    """Review G r6 finding 1 and r7: synthetic pilots through the real ``pilot_verdict`` under model B (pi_e's own
    draw random, arrival share rho = 0.1, jitter harm +0.022, TPOT/TBT drawn). The weight comes from the reviewer's
    synthetic open-loop rows (regime 0: chaos only -> fallback; regime 1: real per-decision hinge gains -> rule met,
    cross-fitted gain > 0): neither gates the claim. G r7 rates (300 pilots): true null "no headroom" ~70 % (G r8 re-run), window
    bound coverage ~98 %, pooled ~95 %; true -20 % on both tails "headroom" ~66 %; a trade never "no headroom"."""
    choices = {0: [hr.weight_sweep(_regime_rows(0, s), "full", 1.0)["choice"] for s in range(4)],
               1: [hr.weight_sweep(_regime_rows(1, s, gain_h=3.0), "full", 1.0)["choice"] for s in range(4)]}
    assert not all(c["rule_met"] for c in choices[0])  # chaos only: mostly the fallback ...
    assert any(c["rule_met"] and c["gain_xfit"]["lo"] > 0 for c in choices[1])  # ... real gains: rule met, xfit > 0
    claims, labels, cov_w, cov_p = [], [], [], []
    for regime, chs in choices.items():
        c, lab, w, p = _oc((0.0, 0.0), 30, 10 + regime, chs)
        claims += c
        labels += lab
        cov_w.append(w)
        cov_p.append(p)
    assert "headroom" not in claims and claims.count("no headroom") >= 0.6 * len(claims), claims
    assert labels.count("negative") <= 0.1 * len(labels) and np.mean(cov_w) >= 0.94 and np.mean(cov_p) >= 0.85
    pos, plab, _, _ = _oc((-0.2, -0.2), 30, 20)
    assert pos.count("headroom") >= 0.45 * len(pos) and "no headroom" not in pos, pos
    trade, tlab, _, _ = _oc((-0.15, 0.15), 15, 30)
    assert "no headroom" not in trade and "headroom" not in trade and tlab.count("trade") >= 0.5 * len(tlab)


def test_boot_ci():
    ci = hr.boot_ci([1.0, 2.0, 3.0, float("nan"), 4.0])
    assert ci["n"] == 4 and ci["mean"] == 2.5 and 1.0 <= ci["lo"] < 2.5 < ci["hi"] <= 4.0
    assert hr.boot_ci([])["mean"] is None
    # serially correlated points: the block bootstrap CI is wider than the i.i.d. one
    x = np.repeat(np.random.default_rng(0).normal(0, 1, 10), 20)
    iid, blk = hr.boot_ci(x), hr.boot_ci(x, block=20)
    assert blk["block"] == 20 and blk["mean"] == iid["mean"] and blk["hi"] - blk["lo"] > 2 * (iid["hi"] - iid["lo"])
    assert hr.lag1(x) > 0.9 and hr.lag1([1.0, 1.0, 1.0]) is None


# ------------------------------------------------------------------ step B (K sampled futures, transfer, gate)

def test_futures_crn_and_average(plain, tmp_path):
    """K futures: every candidate sees the same K sampled futures (each equal to a single-future rollout with that
    seed), the candidate's terms are their mean, and the futures differ."""
    _, s = plain
    plan = _plan(s)
    h = build_simulator(hr.argv_for(SC, tmp_path))
    assert h.run_to_decision(K)
    pool, cap = hr.output_pool(SC), ev.max_request_tokens(SC)
    seeds = hr.future_seeds(h.decisions, 2)
    assert seeds[0] == 1000003 * h.decisions + 7 and len(set(seeds + hr.future_seeds(h.decisions + 1, 2))) == 4
    hz = hr.Horizon.parse("T=10+drain")
    samples = [(x, pool, cap) for x in seeds]
    pe, mean, per, none = hr.evaluate_futures(h, [None, 1], hz, plan, 2, 600, samples)
    assert none is None
    pe2, mean2, per2, aud = hr.evaluate_futures(h, [None, 1], hz, plan, 2, 600, samples, audit=True)
    true = hr._by(hr.evaluate_candidates(h, [None, 1], [hz], plan, 2), 0, pe)
    strip_ = lambda t: {k: v for k, v in t.items() if k != "cpu_s"}  # noqa: E731
    assert pe2 == pe and {c: strip_(t) for c, t in aud.items()} == {c: strip_(t) for c, t in true.items()}
    assert all(mean2[c]["sum"] == mean[c]["sum"] for c in mean)  # the audit is not averaged in
    assert len(per) == 2 and set(mean) == set(per[0]) == {pe, 1}
    strip = lambda t: {k: v for k, v in t.items() if k != "cpu_s"}  # noqa: E731
    for j, smp in enumerate(samples):  # CRN: the same future j for every candidate
        single = hr._by(hr.evaluate_candidates(h, [None, 1], [hz], plan, 2, sample=smp), 0, pe)
        assert {c: strip(t) for c, t in single.items()} == {c: strip(t) for c, t in per[j].items()}
    for c in mean:
        for k in hr._AVG:
            assert mean[c][k] == pytest.approx((per[0][c][k] + per[1][c][k]) / 2)
        assert mean[c]["n"] == per[0][c]["n"] == per[1][c]["n"]
    assert any(per[0][c]["sum"] != per[1][c]["sum"] for c in mean)  # the futures differ
    one = hr.mean_terms([per[0][1]])  # K = 1: the terms themselves
    assert all(one[k] == per[0][1][k] for k in hr._AVG)


def test_closed_loop_crn_k1_equals_sampled_mode_and_resumes(plain, tmp_path):
    _, s = plain
    plan, hz = _plan(s), hr.Horizon.parse("T=5+drain")
    a = hr.closed_loop(SC, hz, plan, s["cut"], tmp_path / "A", tmp_path / "a.jsonl", workers=2,
                       future="sampled-lengths", max_decisions=3, chooser=_worst)
    b = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "B", tmp_path / "b.jsonl", workers=2, n_futures=1,
                           max_decisions=3, chooser=_worst)
    _same_outputs(tmp_path / "A", tmp_path / "B")
    ra = [json.loads(x) for x in open(tmp_path / "a.jsonl")][1:]
    rb = [json.loads(x) for x in open(tmp_path / "b.jsonl")][1:]
    assert a["deviations"] == b["deviations"] == 3
    for x, y in zip(ra, rb):
        assert (x["k"], x["pe"], x["chosen"]) == (y["k"], y["pe"], y["chosen"])
        assert {c: v["score"] for c, v in x["cands"].items()} == {c: v["score"] for c, v in y["cands"].items()}
        assert y["seeds"] == [1000003 * (x["k"] - 1) + 7]
    # K = 2: the decision is the argmin of the mean score over the two futures (default chooser)
    log = tmp_path / "c.jsonl"
    c = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "C", log, workers=2, n_futures=2, max_decisions=2)
    rows = [json.loads(x) for x in open(log)][1:]
    assert c["planned"] == 2 and c["n_futures"] == 2 and len(rows) == 2
    for r in rows:
        assert len(r["seeds"]) == 2
        for v in r["cands"].values():
            assert len(v["score_k"]) == 2 and v["score"] == pytest.approx(np.mean(v["score_k"]))
        sc_ = {int(k): v["score"] for k, v in r["cands"].items()}
        assert r["chosen"] == (r["pe"] if r["decided_by"] == "keep" else min(sc_, key=sc_.get))
    c2 = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "C2", log, workers=2, n_futures=2, max_decisions=2)
    assert c2["replayed"] == 2
    _same_outputs(tmp_path / "C", tmp_path / "C2")
    with pytest.raises(ValueError, match="scalar"):
        hr.closed_loop_crn(SC, hz, hr.Plan("lex"), s["cut"], tmp_path / "D", tmp_path / "d.jsonl")


def _pilot_dir(root, start, weight, tau, label="100m", meta=None, sc=SC):
    """A minimal pilot window directory: summary with pi_e and a current open-loop row carrying a weight choice."""
    d = Path(root) / f"w{start}"
    d.mkdir()
    tag = hr.replace(sc, window_start=start).tag
    rows = {"pi_e": {"p99_e2e": tau, "p99_ttft": tau / 2, "mean_tpot": tau / 1000, "mean_e2e": tau / 5},
            "ol": {"params": {"mode": "openloop", "max_points": None, "future": "true",
                              "decision_sha": hr.decision_sha("openloop")}, "post_sha": hr.post_sha(),
                   "stats": {"full": {"weight_choice": {"label": label, "weight": weight, "rule_met": True}}}}}
    m = {"scenario": tag, "code_version": "cv", "bundle": "b"} | (meta or {})
    (d / "summary.json").write_text(json.dumps({"format": hr.FORMAT, "meta": m, "rows": rows}))
    return str(d)


def test_transferred_settings_used_and_own_window_refused(tmp_path):
    meta = {"scenario": SC.tag, "code_version": "cv", "bundle": "b"}
    d0, d1 = _pilot_dir(tmp_path, 0, 100.0, 90.0), _pilot_dir(tmp_path, 2000, 300.0, 110.0, label="tails")
    x = hr.transfer_settings(hr._split_dirs([f"{d0},{d1}"]), SC, meta)
    assert (x["tau_e"], x["tau_f"], x["weight"]) == (100.0, 50.0, 200.0) and x["tpot_ref"] == pytest.approx(0.1)
    assert x["src"]["windows"] == [hr.replace(SC, window_start=0).tag, hr.replace(SC, window_start=2000).tag]
    assert x["src"]["weight_labels"] == ["100m", "tails"]
    # this window's own pi_e and open-loop choice differ: with transfers they are never used
    own_pe = {"p99_e2e": 50.0, "p99_ttft": 20.0, "mean_tpot": 0.05, "mean_e2e": 10.0}
    (tmp_path / "own").mkdir()
    own_rows = json.loads(Path(_pilot_dir(tmp_path / "own", 6000, 999.0, 50.0), "summary.json").read_text())["rows"]
    psha = hr.post_sha()
    st, w_pre = hr.resolve_settings("closed", "tail", None, own_pe, own_rows, "true", psha, x, x)
    assert st == {"tau_e": 100.0, "tau_f": 50.0, "tpot_ref": x["tpot_ref"], "weight": 200.0} and w_pre is None
    st0, w0 = hr.resolve_settings("closed", "tail", None, own_pe, own_rows, "true", psha)  # pilot behaviour
    assert st0 == {"tau_e": 50.0, "tau_f": 20.0, "tpot_ref": 0.05, "weight": 999.0} and w0["weight"] == 999.0
    st1, _ = hr.resolve_settings("closed", "tail", None, own_pe, own_rows, "true", psha, None, x)  # thresholds only
    assert st1["tau_e"] == 100.0 and st1["weight"] == 999.0
    for wspec in ("auto", "25m", "5.0"):  # own-window or explicit weight with a transferred one: refused
        with pytest.raises(SystemExit, match="refused"):
            hr.resolve_settings("closed", "tail", wspec, own_pe, own_rows, "true", psha, x, x)
    with pytest.raises(SystemExit, match="open loop"):
        hr.resolve_settings("openloop", "tail", None, own_pe, own_rows, "true", psha, None, x)
    with pytest.raises(SystemExit, match="auto"):  # pilot behaviour: closed auto without an open-loop row
        hr.resolve_settings("closed", "tail", None, own_pe, {}, "true", psha)
    # sources that would leak or do not match are refused
    with pytest.raises(SystemExit, match="overlaps"):
        hr.transfer_settings([_pilot_dir(tmp_path, 6100, 1.0, 1.0)], SC, meta)
    with pytest.raises(SystemExit, match="another setup"):
        hr.transfer_settings([_pilot_dir(tmp_path, 4000, 1.0, 1.0, meta={"bundle": "x"})], SC, meta)
    other = hr.replace(SC, trace="Tool")
    with pytest.raises(SystemExit, match="not a window"):
        hr.transfer_settings([_pilot_dir(tmp_path, 3000, 1.0, 1.0, sc=other)], SC, meta)
    nochoice = _pilot_dir(tmp_path, 5000, 1.0, 1.0)
    summ = json.loads(Path(nochoice, "summary.json").read_text())
    summ["rows"]["ol"]["stats"]["full"]["weight_choice"] = None
    Path(nochoice, "summary.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit, match="no pre-specified weight"):
        hr.transfer_settings([nochoice], SC, meta)
    # the CLI refuses bad combinations before any simulation
    for argv in (["--mode", "openloop", "--thresholds-from", d0], ["--tail-weight-from", d0, "--tail-weight", "5"],
                 ["--n-futures", "4"], ["--n-futures", "0"]):
        with pytest.raises(SystemExit):
            hr.main(argv + ["--out", str(tmp_path / "cli")])
    assert not (tmp_path / "cli").exists()


def test_default_keys_unchanged():
    """Step B adds code without re-keying the pilot: the per-mode decision shas and post_sha of the pilot's stored
    rows are still current (checked against the Alpine summaries when present)."""
    shas = {m: hr.decision_sha(m) for m in ("closed", "closed_crn", "openloop", "placebo", "noise")}
    assert len(set(shas.values())) == 5
    d = ev.REPO / "evolve/rl_plan/out_alpine/v41_tool_w0/summary.json"
    if not d.exists():
        pytest.skip("pilot summaries not present")
    rows = json.loads(d.read_text())["rows"]
    seen = set()
    for k, v in rows.items():
        p = hr._params(v)
        if p.get("mode") in ("closed", "openloop", "placebo", "noise"):
            assert p["decision_sha"] == shas[p["mode"]] and v["post_sha"] == hr.post_sha(), k
            assert hr._row_key(p) == k
            seen.add(p["mode"])
    assert seen == {"closed", "openloop", "placebo", "noise"}


def test_pooled_test_matches_pooled_stat():
    rng = np.random.default_rng(5)
    Ns = [[_vs(*rng.normal(0, 0.02, 4)) for _ in range(20)] for _ in range(4)]
    for L in (_vs(0.0, 0.0), _vs(-0.03, 0.01, 0.05, -0.02)):
        a, b = hr.pooled_stat([(L, N) for N in Ns]), hr.pooled_test([(L, N) for N in Ns])
        for m in ("e2e", "ttft"):
            assert all(a[m][k] == b[m][k] for k in ("Z", "z", "q2.5", "q5", "q50", "q95", "q97.5", "p_low", "n_null"))
    big = [[_vs(*rng.normal(0, 0.02, 4)) for _ in range(380)] for _ in range(4)]  # sampled null, same draws
    a, b = hr.pooled_stat([(_vs(0.01, 0.0), N) for N in big]), hr.pooled_test([(_vs(0.01, 0.0), N) for N in big])
    assert a["e2e"]["p_low"] == b["e2e"]["p_low"] and b["e2e"]["n_null"] == 100_000
    worse = hr.pooled_test([(_vs(0.0, 0.0, 0.08, 0.0), N) for N in Ns])
    assert worse["tpot"]["p_high"] < 0.025 < worse["tbt"]["p_high"] and worse["tpot"]["p_low"] > 0.9


def _bwindow(tag, L, seed=0, rate=0.03, p_off=0.0, transfer=True, n_noise=20, p_rate=None, k=4, rule=None):
    rng = np.random.default_rng(seed)
    sh = {m: hr.decision_sha(m) for m in ("closed_crn", "placebo", "noise")}
    rows = {f"placebo{i}": _ctl("placebo", _vs(*(rng.normal(0, 0.02, 4) + p_off)), sha=sh["placebo"],
                                rate=p_rate if p_rate is not None else rate) for i in range(20)}
    rows |= {f"noise{i}": _ctl("noise", _vs(*rng.normal(0, 0.02, 4)), sha=sh["noise"], noise_kind="jitter",
                               noise_delta=hr.NOISE_DELTA) for i in range(n_noise)}
    trace = tag.split("-")[2]
    src = transfer if isinstance(transfer, list) else [t for t in hr.pilot_windows() if t.split("-")[2] == trace]
    p = {"mode": "closed", "horizon": "full", "plan_every": 1, "max_decisions": None, "future": "sampled-lengths",
         "score": "tail", "weight": 100.0, "tau_e": 90.0, "tau_f": 40.0, "n_futures": k,
         "decision_sha": sh["closed_crn"]}
    if transfer:
        p |= {"thresholds_from": {"windows": src}, "weight_from": {"windows": src}}
    if rule:
        p["deviation_rule"] = rule
    rows["B"] = {"params": p, "deviation_frac": rate, "vs_pi_e": _vs(*L)}
    rows["verdict"] = {"label": "missing"}
    return {"meta": {"scenario": tag, "code_version": CV, "bundle": "b"}, "rows": rows}


def test_gate_verdict(cv):
    tags = hr.b_windows()
    assert len(tags) == 4 and all("w4000+2000" in t or "w6000+2000" in t for t in tags)
    G = lambda ws, **kw: hr.gate_verdict({t: w for t, w in zip(tags, ws)}, **kw)  # noqa: E731
    mk = lambda L, **kw: [_bwindow(t, L, seed=i, **kw) for i, t in enumerate(tags)]  # noqa: E731
    v = G(mk((-0.10, 0.0, 0.0, 0.0)))
    assert v["gate"] == "pass", v
    assert v["clauses"]["e2e_gain"]["gain_pct"] == pytest.approx(100 * (1 - math.exp(-0.1)))
    assert v["clauses"]["e2e_vs_noise"]["p"] < 0.025 and v["clauses"]["e2e_vs_placebo"]["p"] < 0.025
    assert all(w["n_futures"] == 4 and w["n_placebo"] == 20 and w["n_noise"] == 20 for w in v["windows"].values())
    small = G(mk((-0.03, 0.0, 0.0, 0.0)))  # significant but below 5 %
    assert small["gate"] == "fail" and not small["clauses"]["e2e_gain"]["ok"]
    assert small["clauses"]["e2e_vs_noise"]["ok"]
    plc = G(mk((-0.10, 0.0, 0.0, 0.0), p_off=-0.1))  # random overrides gain as much: placebo clause fails
    assert plc["gate"] == "fail" and not plc["clauses"]["e2e_vs_placebo"]["ok"] and plc["clauses"]["e2e_vs_noise"]["ok"]
    for j, m in ((1, "ttft"), (2, "tpot"), (3, "tbt")):
        L = [-0.10, 0.0, 0.0, 0.0]
        L[j] = 0.08
        w = G(mk(tuple(L)))
        assert w["gate"] == "fail" and not w["clauses"]["not_worse"][m]["ok"]
        assert all(w["clauses"]["not_worse"][x]["ok"] for x in ("ttft", "tpot", "tbt") if x != m)
    # incomplete: too few noise runs, placebo at another rate, own-window settings, settings from the wrong windows
    ws = mk((-0.10, 0.0, 0.0, 0.0))
    assert G(ws[:3] + [_bwindow(tags[3], (-0.1, 0, 0, 0), n_noise=19)])["gate"] == "incomplete"
    assert G(ws[:3] + [_bwindow(tags[3], (-0.1, 0, 0, 0), p_rate=0.05)])["gate"] == "incomplete"
    assert G(ws[:3] + [_bwindow(tags[3], (-0.1, 0, 0, 0), transfer=False)])["gate"] == "incomplete"
    wrong = G(ws[:3] + [_bwindow(tags[3], (-0.1, 0, 0, 0), transfer=hr.pilot_windows()[:2])])
    assert wrong["gate"] == "incomplete" and wrong["windows"][tags[3]]["status"] == "wrong_transfer"
    assert G(ws, n_futures=1)["gate"] == "incomplete"  # the rows are K = 4
    assert hr.gate_verdict({t: _bwindow(t, (0, 0, 0, 0)) for t in hr.pilot_windows()})["gate"] == "invalid_windows"
    old = [dict(w, meta=dict(w["meta"], code_version="old")) for w in ws]
    assert G(old)["gate"] == "invalid_windows"


# ------------------------------------------------------------------ step B2 (lcb deviation rule)

def test_select_candidate_lcb():
    """Paired LCB over the futures, Bonferroni over the alternatives: a noisy candidate with the best mean is not
    taken (the mean rule would take it), a consistent one is; no bound above min_gain keeps pi_e."""
    plan = hr.Plan("sum")
    sk = {0: [100.0, 100.0, 100.0, 100.0], 1: [80.0, 120.0, 70.0, 90.0], 2: [95.0, 96.0, 94.0, 95.0],
          3: [101.0, 99.0, 100.0, 100.0]}
    by = {c: {"sum": float(np.mean(v))} for c, v in sk.items()}
    assert hr.select_candidate(by, 0, plan)[0] == 1  # the mean rule picks the noisy best mean
    best, info = hr.select_candidate_lcb(by, sk, 0, plan, 0.05)
    t = st.t.ppf(1 - 0.05 / 3, 3)
    assert info["t_crit"] == pytest.approx(t) and best == 2 and info["decided_by"] == "sum"
    d = np.array([5.0, 4.0, 6.0, 5.0])
    assert info["lcb"][2] == pytest.approx(d.mean() - t * d.std(ddof=1) / 2) and info["lcb"][1] < 0
    assert info["gain"] == pytest.approx(5.0) and set(info["lcb"]) == {1, 2, 3}
    # two pass: the larger bound wins, not the larger mean
    sk2 = {**sk, 3: [96.0, 88.0, 94.0, 90.0]}  # mean gain 8 > 5, but the wider bound
    by2 = {c: {"sum": float(np.mean(v))} for c, v in sk2.items()}
    b2, i2 = hr.select_candidate_lcb(by2, sk2, 0, plan, 0.05)
    assert 0 < i2["lcb"][3] < i2["lcb"][2] and b2 == 2 and np.mean(sk2[3]) < np.mean(sk2[2])
    # nothing passes min_gain: keep
    keep, ik = hr.select_candidate_lcb(by, sk, 0, hr.replace(plan, min_gain=10.0), 0.05)
    assert keep == 0 and ik["decided_by"] == "keep" and ik["gain"] == 0.0
    # zero variance: the gain itself is the bound; an identical rollout is not an override
    sk3 = {0: [10.0, 12.0], 1: [9.0, 11.0], 2: [10.0, 12.0]}
    b3, i3 = hr.select_candidate_lcb({c: {"sum": float(np.mean(v))} for c, v in sk3.items()}, sk3, 0, plan, 0.05)
    assert b3 == 1 and i3["lcb"][1] == pytest.approx(1.0) and i3["lcb"][2] == 0.0
    with pytest.raises(ValueError, match="futures"):
        hr.select_candidate_lcb({0: {"sum": 1.0}, 1: {"sum": 0.0}}, {0: [1.0], 1: [0.0]}, 0, plan, 0.05)
    with pytest.raises(ValueError, match="scalar"):
        hr.select_candidate_lcb(by, sk, 0, hr.Plan("lex"), 0.05)
    with pytest.raises(ValueError, match="alpha"):
        hr.select_candidate_lcb(by, sk, 0, plan, 0.0)


def test_closed_loop_crn_lcb_rule_logs_and_resumes(plain, tmp_path):
    """B2 end to end on a small window: every logged decision is the lcb rule applied to the logged per-future
    scores, the bounds and t_crit are logged, and a resumed run replays without new rollouts."""
    _, s = plain
    plan, hz = _plan(s), hr.Horizon.parse("T=5+drain")
    log = tmp_path / "l.jsonl"
    r = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "A", log, workers=2, n_futures=3, max_decisions=3,
                           rule="lcb", alpha=0.05, audit_true=True)
    assert r["planned"] == 3 and r["deviation_rule"] == {"rule": "lcb", "alpha": 0.05}
    head = json.loads(open(log).readline())
    assert "lcb" in json.dumps(head) and "audit_true" in json.dumps(head)  # the default header carries both
    # the audit is the true-future score: equal to a true-future rollout from the same decision
    first = json.loads(open(log).readlines()[1])
    h = build_simulator(hr.argv_for(SC, tmp_path / "X"))
    while h.next_decision() and h.decisions + 1 < first["k"]:
        h.run(lambda x, n=h.decisions: x.decisions > n)
    tb = hr._by(hr.evaluate_candidates(h, list(range(SC.replicas)), [hz], plan, 2), 0, first["pe"])
    assert all(first["cands"][str(c)]["score_true"] == pytest.approx(sum(hr.score_parts(tb[c], plan).values()))
               for c in tb)
    rows = [json.loads(x) for x in open(log)][1:]
    assert len(rows) == 3
    for row in rows:
        pe = row["pe"]
        sk = {int(c): v["score_k"] for c, v in row["cands"].items()}
        by = {int(c): {k: v[k] for k in ("sum", "hinge", "hinge_ttft")} for c, v in row["cands"].items()}
        best, info = hr.select_candidate_lcb(by, sk, pe, plan, 0.05)
        assert row["chosen"] == best and row["t_crit"] == pytest.approx(st.t.ppf(1 - 0.05 / (SC.replicas - 1), 2))
        assert all(row["cands"][str(c)]["lcb"] == pytest.approx(info["lcb"][c]) for c in info["lcb"])
        assert "lcb" not in row["cands"][str(pe)]
    r2 = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "B", log, workers=2, n_futures=3, max_decisions=3,
                            rule="lcb", alpha=0.05, audit_true=True)
    with pytest.raises(SystemExit):  # a log of one rule never resumes under the other
        hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "M", log, workers=2, n_futures=3, max_decisions=3)
    assert r2["replayed"] == 3 and r2["deviations"] == r["deviations"]
    _same_outputs(tmp_path / "A", tmp_path / "B")
    with pytest.raises(ValueError, match="lcb"):
        hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "C", tmp_path / "c.jsonl", n_futures=1, rule="lcb")
    with pytest.raises(ValueError, match="lcb"):
        hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "C", tmp_path / "c.jsonl", n_futures=2, rule="lcb",
                           chooser=_worst)
    with pytest.raises(ValueError, match="rule"):
        hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "C", tmp_path / "c.jsonl", n_futures=2, rule="max")


def test_gate_selects_rows_by_rule(cv):
    """The gate takes only rows of its rule: B2 (lcb) rows pass under rule lcb, are missing under the mean rule,
    and windows whose rule parameters differ are refused."""
    tags = hr.b_windows()
    lcb = {"rule": "lcb", "alpha": 0.05}
    ws = [_bwindow(t, (-0.10, 0.0, 0.0, 0.0), seed=i, k=8, rule=lcb) for i, t in enumerate(tags)]
    G = lambda ws, **kw: hr.gate_verdict({t: w for t, w in zip(tags, ws)}, n_futures=8, **kw)  # noqa: E731
    v = G(ws, rule="lcb")
    assert v["gate"] == "pass" and v["rule"] == "lcb" and all(w["deviation_rule"] == lcb for w in v["windows"].values())
    assert G(ws)["gate"] == "incomplete"  # mean rule: no B row
    mixed = ws[:3] + [_bwindow(tags[3], (-0.10, 0.0, 0.0, 0.0), seed=3, k=8, rule={"rule": "lcb", "alpha": 0.1})]
    assert G(mixed, rule="lcb")["gate"] == "invalid_windows"
    a2 = [_bwindow(t, (-0.10, 0.0, 0.0, 0.0), seed=i, k=8, rule={"rule": "lcb", "alpha": 0.2}) for i, t in enumerate(tags)]
    assert G(a2, rule="lcb")["gate"] == "invalid_windows" and G(a2, rule="lcb", lcb_alpha=0.2)["gate"] == "pass"
    both = [dict(w, rows=dict(w["rows"], B_mean=_bwindow(t, (0.0, 0, 0, 0), k=8)["rows"]["B"]))
            for w, t in zip(ws, tags)]
    assert G(both, rule="lcb")["gate"] == "pass" and G(both)["gate"] == "fail"  # each rule finds its own head


def test_cli_refuses_lcb_without_futures(tmp_path):
    for argv in (["--deviation-rule", "lcb"], ["--deviation-rule", "lcb", "--n-futures", "1"],
                 ["--deviation-rule", "lcb", "--n-futures", "4", "--future", "sampled-lengths", "--mode",
                  "openloop"], ["--deviation-rule", "lcb", "--n-futures", "4", "--lcb-alpha", "1.5"],
                 ["--lcb-alpha", "0.1"], ["--audit-true"], ["--audit-true", "--n-futures", "4", "--future",
                  "sampled-lengths", "--mode", "placebo"]):
        with pytest.raises(SystemExit):
            hr.main(argv + ["--out", str(tmp_path)])


def _arow(k, pe, chosen, sk, true):
    return {"k": k, "pe": pe, "chosen": chosen,
            "cands": {str(c): {"score": float(np.mean(v)), "score_k": v, "score_true": true[c]} for c, v in sk.items()}}


def test_audit_stats():
    """The pre-registered audit: true gain per override, precision vs the sampled-futures prediction, and the
    mean-rule and random-alternative baselines on the same decisions."""
    rows = [  # pe = 0 everywhere; lower scores are better
        _arow(1, 0, 1, {0: [10.0, 10.0], 1: [8.0, 9.0], 2: [11.0, 12.0]}, {0: 10.0, 1: 7.0, 2: 12.0}),  # right, +3
        _arow(2, 0, 2, {0: [10.0, 10.0], 1: [9.0, 9.0], 2: [9.0, 8.0]}, {0: 10.0, 1: 10.0, 2: 11.0}),  # wrong, -1
        _arow(3, 0, 0, {0: [10.0, 10.0], 1: [9.0, 11.5], 2: [12.0, 12.0]}, {0: 10.0, 1: 10.0, 2: 9.0}),  # kept
    ]
    a = hr.audit_stats({"w": rows})
    assert a["overrides"] == 2 and a["underpowered"] and a["mean_true_gain"]["mean"] == pytest.approx(1.0)
    assert a["precision"] == 0.5 and a["windows"]["w"]["overrides"] == 2
    p1 = hr._norm_cdf(1.5 / np.std([2.0, 1.0], ddof=1))
    p2 = hr._norm_cdf(1.5 / np.std([1.0, 2.0], ddof=1))
    assert a["predicted_precision_upper"] == pytest.approx((p1 + p2) / 2) and a["reading"] == "underpowered"
    assert a["calibration_all_pairs"]["pairs"] == 6 and a["calibration_all_pairs"]["realised"] == pytest.approx(2 / 6)
    # mean rule: decision 1 -> 1 (+3), decision 2 -> 2 (mean 8.5 < 9; -1), decision 3: 1's mean 10.25 > 10, keep
    assert a["mean_rule"]["overrides"] == 2 and a["mean_rule"]["mean_true_gain"]["mean"] == pytest.approx(1.0)
    # random alternative: per-decision means of g over {1, 2}: (3 - 2)/2, (0 - 1)/2, (0 + 1)/2
    assert a["random_alternative"]["mean_true_gain"]["mean"] == pytest.approx((0.5 - 0.5 + 0.5) / 3)
    assert a["random_alternative"]["precision"] == pytest.approx((0.5 + 0.0 + 0.5) / 3)
    tie = hr.audit_stats({"w": [_arow(1, 0, 1, {0: [2.0, 2.0], 1: [1.0, 1.0]}, {0: 5.0, 1: 5.0})]})
    assert tie["precision"] == 0.0 and tie["predicted_precision_upper"] == 1.0  # a tie is not right; sd 0, gain > 0
    many = hr.audit_stats({"a": rows * 10, "b": rows * 6}, block=1)  # gains +3/-1 alternate: i.i.d. blocks here
    assert many["overrides"] == 32 and not many["underpowered"]
    assert 0 < many["mean_true_gain"]["lo"] < 1.0 < many["mean_true_gain"]["hi"] and many["reading"] == "right_not_enough"
    assert many["block"] == 1 and many["audit_sha"] == hr.audit_sha()
    mix = [_arow(i, 0, 1, {0: [10.0, 10.0], 1: [8.0, 9.0]}, {0: 10.0, 1: 10.0 + (3.0 if i % 7 < 3 else -2.5)})
           for i in range(70)]
    assert hr.audit_stats({"w": mix})["reading"] == "inconclusive" and hr.audit_stats({"w": mix})["block"] == hr.BLOCK
    good = [_arow(i, 0, 1, {0: [10.0, 10.0], 1: [8.0, 9.0]}, {0: 10.0, 1: 7.0 + (i % 3)}) for i in range(40)]
    bad = [_arow(i, 0, 1, {0: [10.0, 10.0], 1: [8.0, 9.0]}, {0: 10.0, 1: 11.0 + (i % 3)}) for i in range(40)]
    assert hr.audit_stats({"w": good})["reading"] == "right_not_enough"
    assert hr.audit_stats({"w": bad})["reading"] == "not_better"
    with pytest.raises(ValueError, match="score_true"):
        r = _arow(1, 0, 1, {0: [1.0, 1.0], 1: [0.0, 0.0]}, {0: 1.0, 1: 0.0})
        del r["cands"]["1"]["score_true"]
        hr.audit_stats({"w": [r]})


def test_audit_logs_reads_the_gated_log(plain, tmp_path):
    """The audit reads the gated B2 row's own log (rule, alpha, K, sha, audit) and refuses an incomplete one."""
    _, s = plain
    plan, hz = _plan(s), hr.Horizon.parse("T=5+drain")
    params = {"mode": "closed", "horizon": "full", "plan_every": 1, "max_decisions": None,
              "future": "sampled-lengths", "score": "tail", "n_futures": 2, "thresholds_from": {"windows": ["x"]},
              "weight_from": {"windows": ["x"]}, "decision_sha": hr.decision_sha("closed_crn"),
              "deviation_rule": {"rule": "lcb", "alpha": 0.05}, "audit_true": True}
    key = hr._row_key(params)
    r = hr.closed_loop_crn(SC, hz, plan, s["cut"], tmp_path / "A", tmp_path / f"{key}.jsonl", workers=2,
                           n_futures=2, max_decisions=2, rule="lcb", audit_true=True)
    other = dict(params, audit_true=None)  # an unaudited lcb row is ignored
    summ = {"rows": {key: {"params": params, "planned": r["planned"]}, "o": {"params": other, "planned": 9}}}
    (tmp_path / "summary.json").write_text(json.dumps(summ))
    logs = hr.audit_logs([str(tmp_path)], 2)
    assert len(logs[str(tmp_path)]) == 2 and hr.audit_stats(logs)["random_alternative"]["decisions"] == 2
    with pytest.raises(SystemExit, match="0 audited"):
        hr.audit_logs([str(tmp_path)], 2, lcb_alpha=0.1)
    with pytest.raises(SystemExit, match="0 audited"):
        hr.audit_logs([str(tmp_path)], 8)
    summ["rows"][key]["planned"] = 3
    (tmp_path / "summary.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit, match="planned 3"):
        hr.audit_logs([str(tmp_path)], 2)


def test_base_policy_swap_keeps_decision_code_and_matches_run_one(tmp_path):
    """Step C12: ``--policy`` plans from another base router. The decision shas stay those of the pi_e runs (only
    ``argv_for``'s default and ``POLICY`` are rebound), the base run equals ``evaluate.run_one`` with that router, and
    always keeping its choice through the fork path reproduces it."""
    base = ev.REPO / "evolve/finalists/v4/v4_487363424749.py"
    old_policy, old_defaults = hr.POLICY, hr.argv_for.__defaults__
    shas = {m: hr.decision_sha(m) for m in ("closed", "closed_crn", "openloop", "placebo", "noise")}
    try:
        hr.set_base_policy(base)
        assert hr.POLICY == str(base.resolve()) and hr.argv_for.__defaults__ == (str(base.resolve()),)
        assert {m: hr.decision_sha(m) for m in shas} == shas
        d = tmp_path / "A"
        h = build_simulator(hr.argv_for(SC, d))
        h.run()
        h.write_output()
        s = hr.score_output(d, SC)
        r = ev.run_one("evolved", SC, policy=hr.POLICY)
        assert s["ok"] and r["ok"]
        for k in ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e"):
            assert s[k] == r[k], k
        hr.closed_loop(SC, hr.Horizon.parse("T=10+drain"), _plan(s), s["cut"], tmp_path / "B", tmp_path / "b.jsonl",
                       workers=2, max_decisions=5, candidates="pe")
        _same_outputs(d, tmp_path / "B")
        with pytest.raises(SystemExit):
            hr.set_base_policy(tmp_path / "missing.py")
    finally:
        hr.POLICY, hr.argv_for.__defaults__ = old_policy, old_defaults
