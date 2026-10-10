"""The learned (RL) router harness: contract checks, training/inference round trip,
Double-DQN target, and that it sees only router-visible information."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import base_args, run_simulator
from vidur.scheduler.global_scheduler.evolved_global_scheduler import PolicyError
from vidur.scheduler.global_scheduler.learned_global_scheduler import resolve_config

ROOT = Path(__file__).resolve().parents[1]
SEED_DESIGN = ROOT / "evolve" / "rl_designs" / "seed_ttft_bandit.py"
TEACHER = ROOT / "evolve" / "policies" / "lor.py"


def run(trace, out, design, mode="training", ckpt_in="", ckpt_out="", teacher="", replicas=3):
    return run_simulator(base_args(
        trace, out, num_replicas=replicas, global_scheduler_config_type="learned",
        learned_global_scheduler_config_policy_path=str(design),
        learned_global_scheduler_config_mode=mode,
        learned_global_scheduler_config_checkpoint_in=ckpt_in or None,
        learned_global_scheduler_config_checkpoint_out=ckpt_out or None,
        learned_global_scheduler_config_teacher_path=teacher or None,
    ))


def design_with(tmp_path, **cfg):
    src = SEED_DESIGN.read_text()
    extra = "\nCONFIG.update(" + repr(cfg) + ")\n"
    p = tmp_path / "design.py"
    p.write_text(src + extra)
    return p


def test_training_then_inference_round_trip(conv_session_trace, tmp_path):
    design = design_with(tmp_path, learning_starts=8, batch_size=8, train_every=1)
    ck = tmp_path / "net.pt"
    sim = run(conv_session_trace, tmp_path / "train", design, ckpt_out=str(ck))
    assert sim._scheduler.is_empty()
    stats = json.loads(Path(str(ck) + ".stats.json").read_text())
    assert stats["gradient_steps"] > 0 and stats["reward"]["mean"] < 0
    assert stats["unfinished_decisions_dropped"] == 0
    inf = run(conv_session_trace, tmp_path / "inf", design, mode="inference", ckpt_in=str(ck))
    assert inf._scheduler.is_empty()
    d = inf._scheduler.diagnostics()
    assert d["gradient_steps"] == stats["gradient_steps"]  # frozen: no learning in inference
    assert d["decisions"] == 2 * stats["decisions"]  # counters carry over from the checkpoint
    assert d["explore_frac"] == 0.0


def test_inference_is_deterministic(conv_session_trace, tmp_path):
    design = design_with(tmp_path, learning_starts=8, batch_size=8)
    ck = tmp_path / "net.pt"
    run(conv_session_trace, tmp_path / "train", design, ckpt_out=str(ck))
    a = run(conv_session_trace, tmp_path / "a", design, mode="inference", ckpt_in=str(ck))
    b = run(conv_session_trace, tmp_path / "b", design, mode="inference", ckpt_in=str(ck))
    assert a._scheduler.diagnostics()["action_share"] == b._scheduler.diagnostics()["action_share"]


def test_dqn_with_teacher_runs_and_tracks_agreement(conv_session_trace, tmp_path):
    design = design_with(tmp_path, algorithm="dqn", beta=0.1, learning_starts=8, batch_size=8,
                         teacher_mix=0.5, imitation_weight=1.0, target_update=5)
    ck = tmp_path / "net.pt"
    run(conv_session_trace, tmp_path / "t", design, ckpt_out=str(ck), teacher=str(TEACHER))
    stats = json.loads(Path(str(ck) + ".stats.json").read_text())
    assert stats["teacher_agreement"] is not None and 0 < stats["teacher_acted_frac"] < 1


def test_config_validation():
    assert resolve_config(None)["algorithm"] == "bandit"
    with pytest.raises(PolicyError):
        resolve_config({"algorithm": "ppo"})
    with pytest.raises(PolicyError):
        resolve_config({"lr": 5.0})
    with pytest.raises(PolicyError):
        resolve_config({"not_a_key": 1})
    with pytest.raises(PolicyError):
        resolve_config({"hidden": [64, 0]})


def test_bad_features_rejected(tiny_trace, tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("CONFIG = {}\ndef replica_features(r, v, vs, ctx):\n    return [float('nan')]\n"
                 "def reward(o):\n    return 0.0\n")
    with pytest.raises(PolicyError):
        run(tiny_trace, tmp_path / "o", p)


def test_outcome_and_views_hide_output_length(tiny_trace, tmp_path):
    p = tmp_path / "spy.py"
    p.write_text(
        "CONFIG = {'learning_starts': 8, 'batch_size': 8}\n"
        "SEEN = []\n"
        "def replica_features(r, v, vs, ctx):\n"
        "    SEEN.append(sorted(vars(r)) if hasattr(r, '__dict__') else sorted(r.__dataclass_fields__))\n"
        "    return [v.pending_prefill_tokens / 1e4]\n"
        "def reward(o):\n"
        "    return -o.ttft\n")
    sim = run(tiny_trace, tmp_path / "o", p)
    fields = sim._scheduler._policy.SEEN[0]
    assert not any("decode" in f for f in fields)


def test_double_dqn_target_matches_reference(tiny_trace, tmp_path):
    """One training step computes y = r + g * Q_target(s', argmax_a Q_online(s', a))."""
    import torch

    design = design_with(tmp_path, algorithm="dqn", learning_starts=10**5)  # never trains during the run
    sim = run(tiny_trace, tmp_path / "o", design)
    s = sim._scheduler
    torch.manual_seed(0)
    for p in s._target.parameters():  # make target differ from online
        p.data.add_(torch.randn_like(p) * 0.5)
    B, N, F = 4, 3, s._feature_dim
    rng = np.random.default_rng(0)
    batch = [(rng.normal(size=(N, F)).astype(np.float32), i % N, float(-i), rng.normal(size=(N, F)).astype(np.float32),
              0.9 if i < 3 else 0.0, -1, np.ones(N, dtype=bool)) for i in range(B)]
    s._buffer = batch
    s._cfg = dict(s._cfg, batch_size=B, lr=0.0, imitation_weight=0.0)
    s._rng.randrange = lambda n, it=iter(range(B)): next(it)  # take the batch in order
    with torch.no_grad():
        ys = []
        for x, a, r, nx, g, _, _m in batch:
            q_on = s._net(torch.from_numpy(nx)).squeeze(-1)
            q_tg = s._target(torch.from_numpy(nx)).squeeze(-1)
            ys.append(r + g * q_tg[int(q_on.argmax())].item())
        qs = [s._net(torch.from_numpy(x)).squeeze(-1)[a].item() for x, a, *_ in batch]
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor(qs), torch.tensor(ys)).item()
    seen = {}
    orig = torch.nn.functional.smooth_l1_loss

    def spy(q, y):
        seen["loss"] = orig(q, y).item()
        return orig(q, y)

    torch.nn.functional.smooth_l1_loss = spy
    try:
        s._train_step()
    finally:
        torch.nn.functional.smooth_l1_loss = orig
    assert math.isclose(seen["loss"], expected, rel_tol=1e-5)


def test_interval_reward_only_design(conv_session_trace, tmp_path):
    """A design with only interval_reward stores a transition at every next decision."""
    design = ROOT / "evolve" / "rl_designs" / "seed_dqn_interval.py"
    p = tmp_path / "d.py"
    p.write_text(design.read_text() + "\nCONFIG.update({'learning_starts': 8, 'batch_size': 8})\n")
    ck = tmp_path / "net.pt"
    sim = run(conv_session_trace, tmp_path / "o", p, ckpt_out=str(ck))
    assert sim._scheduler.is_empty()
    stats = json.loads(Path(str(ck) + ".stats.json").read_text())
    assert stats["reward"] is None and stats["interval_reward"]["mean"] <= 0
    assert stats["gradient_steps"] > 0 and stats["unfinished_decisions_dropped"] == 0


def test_candidate_mask_is_respected(conv_session_trace, tmp_path):
    """A design that only allows replica 0 must route everything there, greedy or exploring."""
    p = tmp_path / "d.py"
    p.write_text(SEED_DESIGN.read_text() + "\nCONFIG.update({'learning_starts': 8, 'batch_size': 8, "
                 "'epsilon_start': 1.0, 'epsilon_end': 1.0})\n"
                 "def candidate_mask(request, replicas, ctx):\n"
                 "    return [r.index == 0 for r in replicas]\n")
    sim = run(conv_session_trace, tmp_path / "o", p)
    assert sim._scheduler.diagnostics()["action_share"][0] == 1.0


def test_teacher_phase_out_schedule():
    from vidur.scheduler.global_scheduler.learned_global_scheduler import LearnedGlobalScheduler

    cfg = resolve_config({"teacher_mix": 0.5, "teacher_mix_end": 0.0, "imitation_weight": 2.0,
                          "imitation_weight_end": 0.5, "teacher_decay_decisions": 100})
    s = LearnedGlobalScheduler.__new__(LearnedGlobalScheduler)  # only the schedule is exercised
    s._cfg = cfg
    for n, mix, iw in ((0, 0.5, 2.0), (50, 0.25, 1.25), (100, 0.0, 0.5), (500, 0.0, 0.5)):
        s._n_decisions = n
        assert abs(s._scheduled("teacher_mix") - mix) < 1e-12
        assert abs(s._scheduled("imitation_weight") - iw) < 1e-12
    with pytest.raises(PolicyError):
        resolve_config({"teacher_mix_end": 2.0})


def test_replay_buffer_persists_across_episodes(conv_session_trace, tmp_path):
    design = design_with(tmp_path, learning_starts=8, batch_size=8)
    ck1, ck2 = tmp_path / "e1.pt", tmp_path / "e2.pt"
    run(conv_session_trace, tmp_path / "a", design, ckpt_out=str(ck1))
    s1 = json.loads(Path(str(ck1) + ".stats.json").read_text())
    assert Path(str(ck1) + ".buffer.npz").exists() and s1["buffer_size_end"] > 0
    run(conv_session_trace, tmp_path / "b", design, ckpt_in=str(ck1), ckpt_out=str(ck2))
    s2 = json.loads(Path(str(ck2) + ".stats.json").read_text())
    assert s2["buffer_loaded_from_previous_episode"] == s1["buffer_size_end"]
    assert s2["buffer_size_end"] > s1["buffer_size_end"]


def test_each_decision_is_stored_once(conv_session_trace, tmp_path):
    """v4 fix: a bandit with an outcome reward used to store a decision twice when its request finished before
    the next routing decision (rl_inventory §3.3)."""
    design = design_with(tmp_path, buffer_size=500000)  # buffer never wraps on this trace
    ck = tmp_path / "net.pt"
    run(conv_session_trace, tmp_path / "train", design, ckpt_out=str(ck))
    stats = json.loads(Path(str(ck) + ".stats.json").read_text())
    assert stats["buffer_size_end"] == stats["decisions"] - stats["unfinished_decisions_dropped"]
