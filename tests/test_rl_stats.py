"""evolve.analysis.rl_stats: run/candidate summaries from a tiny fake RL run, and the e413 held-out ratios."""

import json
from pathlib import Path

import pytest

from evolve.analysis import rl_stats

ROOT = Path(__file__).resolve().parents[1]


def _learner(dec, steps, q_last, loss_mean, agree, buf_end=None, buf_loaded=None):
    L = {"mode": "training", "decisions": dec, "gradient_steps": steps, "epsilon_now": 0.05, "explore_frac": 0.05,
         "teacher_acted_frac": 0.1, "teacher_agreement": agree, "action_share": [0.5, 0.25, 0.25],
         "reward": {"mean": -0.5, "p5": -1.0, "p95": 0.1, "first": -0.6, "last": -0.4},
         "interval_reward": None,
         "loss": {"mean": loss_mean, "p5": loss_mean / 2, "p95": loss_mean * 2, "first": loss_mean, "last": loss_mean},
         "q_mean": {"mean": q_last / 2, "p5": q_last, "p95": 0.0, "first": 0.0, "last": q_last},
         "train_seconds": 1.0, "policy_seconds": 0.5, "unfinished_decisions_dropped": 0}
    if buf_end is not None:
        L.update({"buffer_size_end": buf_end, "buffer_loaded_from_previous_episode": buf_loaded})
    return L


def _episode(tag, learner):
    return {"episode": tag, "p99_e2e": 20.0, "p99_ttft": 5.0, "p99_tpot": 0.05, "mean_e2e": 4.0, "wall_s": 50.0,
            "learner": learner}


@pytest.fixture
def fake_run(tmp_path):
    run = tmp_path / "evolve" / "runs" / "rl_fake"
    (run / "programs").mkdir(parents=True)
    (run / "programs" / "aaaaaaaaaaaa.py").write_text(
        "BETA = 0.1\nCONFIG = {'algorithm': 'dqn', 'beta': BETA, 'hidden': [16], 'teacher_mix': 0.3}\n"
        "def replica_features(request, replica, replicas, ctx):\n    return [0.0]\n"
        "def reward(outcome):\n    return 0.0\n")
    details = [{"scenario": "7b-8R-Tool-s1.0-full-p1", "metric": m, "p99": 0.9, "best_heuristic_p99": 1.0,
                "log_ratio": -0.105} for m in ("e2e", "ttft", "tpot")]
    a = {"id": "aaaaaaaaaaaa", "parent": None, "time": 1.79e9, "valid": True, "stage_reached": "1", "task": "rl",
         "rl_protocol": "loto_conv", "objective": "balanced", "fitness": 0.2,
         "proposer": {"seed": "seed_x", "cost_usd": 0.0},
         "result": {"code_version": "c000", "details": details, "eval_wall_s": 100.0,
                    "training": {"ok": True, "train_wall_s": 100.0, "episodes": [
                        _episode("7b-8R-Tool-s1.0-w0+4000-p4", _learner(1000, 100, -1.0, 0.2, 0.6)),
                        _episode("7b-8R-Synth-s0.75-full-p4", _learner(2000, 300, -8.0, 0.4, 0.7))]}}}
    # candidate b: no episodes in the archive row; its learner stats come from the stats file on disk
    b = {"id": "bbbbbbbbbbbb", "parent": "aaaaaaaaaaaa", "time": 1.79e9 + 3600, "valid": True, "stage_reached": "1",
         "task": "rl", "rl_protocol": "loto_conv", "objective": "balanced", "fitness": -1.0,
         "proposer": {"cost_usd": 0.5}, "result": {"code_version": "c000", "details": details}}
    (run / "archive.jsonl").write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    tdir = run / "training" / "bbbbbbbbbbbb"
    tdir.mkdir(parents=True)
    (tdir / "net_ep0.pt.stats.json").write_text(json.dumps(_learner(1500, 200, -0.3, 0.1, 0.4, 1550, 0)))
    (tdir / "net_ep0.pt").write_bytes(b"not a checkpoint")  # unreadable: must not crash
    return tmp_path, run


def test_summarize_fake_run(fake_run):
    repo, run = fake_run
    sources = rl_stats.discover(repo, [str(run)])
    assert [(n, p.name) for n, p, _ in sources] == [("rl_fake", "archive.jsonl")]
    s = rl_stats.summarize(sources, repo=repo, log=lambda *_: None)
    assert set(s) >= {"runs", "candidates", "finalists", "offline", "warnings", "not_available"}
    r = s["runs"]["rl_fake"]
    assert (r["n_candidates"], r["n_valid"], r["best_id"]) == (2, 2, "aaaaaaaaaaaa")
    assert r["best_fitness"] == pytest.approx(0.2)
    assert r["median_fitness"] == pytest.approx(-0.4)
    assert r["proposer_cost_usd"] == pytest.approx(0.5)
    assert r["models"] == ["7b"] and r["not_l3_8b"]
    assert r["protocols"] == {"loto_conv": 2}
    assert r["flag_counts"]["q_drift"] == {"true": 1, "known": 2}
    assert any("not l3_8b" in w for w in s["warnings"])
    assert any("double-storage" in w for w in s["warnings"])

    ca = s["candidates"]["rl_fake/aaaaaaaaaaaa"]
    assert ca["summary"]["algorithm"] == "dqn" and ca["summary"]["beta"] == 0.1  # BETA resolved from the module
    assert ca["summary"]["hidden"] == [16] and ca["summary"]["teacher"]["uses_teacher"]
    la = ca["learner"]
    assert la["n_episodes"] == 2 and la["decisions"] == 2000 and la["gradient_steps"] == 300
    assert la["q_last_final"] == -8.0 and la["flags"]["q_drift"]
    assert la["loss_last_over_first"] == pytest.approx(2.0) and la["flags"]["loss_rising"]
    assert la["teacher_agreement_final"] == 0.7
    assert la["action_max_share_max"] == pytest.approx(0.5) and la["flags"]["action_collapse"]
    assert la["sim_wall_s"] == 100.0 and la["grad_share_of_wall"] == pytest.approx(0.02)
    assert la["dup_excess"] is None  # no buffer keys in the archive learner
    assert ca["search_vs_bar"]["ttft"]["geo"] == pytest.approx(0.9)

    cb = s["candidates"]["rl_fake/bbbbbbbbbbbb"]
    assert cb["episodes_source"] == "stats_files"
    assert cb["learner"]["dup_excess"] == 50 and cb["learner"]["flags"]["dup_storage"]
    assert cb["checkpoint"]["error"]  # bad checkpoint reported, not raised
    assert cb["design"]["config"] is None

    out = repo / "o.md"
    rl_stats.write_md(s, out, offline_requested=False)
    md = out.read_text()
    assert "NOT AVAILABLE" in md and "rl_fake" in md


def test_e413_heldout_ratios_match_results():
    f = ROOT / "evolve" / "finalists" / "heldout_loto_rl_loto_r2_e413fe9e193c.json"
    if not f.exists() or not (ROOT / "evolve" / ".cache" / "baselines.json").exists():
        pytest.skip("held-out JSON or baseline cache not on this machine")
    h = rl_stats.heldout_summary(f, ROOT, rl_stats.load_baselines(ROOT), rl_stats._rule_key(ROOT), {})
    got = [round(h["vs_heuristics"]["overall"][m]["geo"], 3) for m in ("e2e", "ttft", "tpot")]
    assert got == [0.901, 0.753, 1.098]
