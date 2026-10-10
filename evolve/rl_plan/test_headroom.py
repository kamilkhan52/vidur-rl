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
- resampled output lengths leak no true length; fork failures are reported and every child is reaped.
All on Conv 4R s1.5 w6000+200 p1 (~8 s per full run)."""

import filecmp
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
            assert ta == tb, x.label
    assert multi[0]["h"][3]["n"] > multi[0]["h"][1]["n"]  # full covers more arrivals than 10 s


def test_open_loop_selection_and_resume(plain, tmp_path):
    _, s = plain
    plan = _plan(s)
    hz = [hr.Horizon.parse("H=8"), hr.Horizon.parse("T=10+drain")]
    ref = hr.Horizon.parse("full")
    log = tmp_path / "o.jsonl"
    st = hr.open_loop(SC, hz, ref, plan, s["cut"], log, workers=2, every=20, max_points=2)
    assert st["_run"]["points"] == 2 and set(st) == {"H8", "T10s+drain", "full", "_run"}
    assert st["full"]["top1_agree"] == 1.0 and st["full"]["ref_gain_mean"] >= max(st[x]["ref_gain_mean"]
                                                                                   for x in ("H8", "T10s+drain"))
    for x in ("H8", "T10s+drain"):
        assert 0 <= st[x]["pair_sign_agree"] <= 1 and st[x]["standalone_cpu_s_per_decision"] > 0
    n_lines = len(log.read_text().splitlines())
    st2 = hr.open_loop(SC, hz, ref, plan, s["cut"], log, workers=2, every=20, max_points=2)  # resume: no new rows
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
    # the parent's requests and trace are untouched
    assert [r.original_num_decode_tokens for r in hr._inflight(h)] != [r.original_num_decode_tokens for r in infl]
    assert np.array_equal(h.sim._request_generator._num_decode_tokens[i:], true_future)


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
    assert hr._row_key(dict(base, code_sha=hr.CODE_SHA)) != hr._row_key(dict(base, code_sha={"headroom": "x"}))
    assert set(hr.CODE_SHA) == {"headroom", "sim_fork"}
