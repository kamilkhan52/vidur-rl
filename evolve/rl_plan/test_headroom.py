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
  not negative; one setup across windows.
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
    cross-fitted gain > 0): neither gates the claim. G r7 rates (300 pilots): true null "no headroom" ~77 %, window
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
