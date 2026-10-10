"""The learned (RL) router harness: contract checks, training/inference round trip,
Double-DQN target, and that it sees only router-visible information."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from conftest import base_args, run_simulator
from vidur.scheduler.global_scheduler.evolved_global_scheduler import PolicyError
from vidur.scheduler.global_scheduler.learned_global_scheduler import resolve_config, rng_seeds

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


# --------------------------------------------------------------------------- RL_LOGGING_SPEC (2026-10-10)

# Digests of the runs below, computed with the learner as committed before the logging change (this repo's af87600 =
# 7fb6d38^ for learned_global_scheduler.py; code_version c16266a8ef49) injected into the current tree, on the Mac mini.
# The DQN + teacher digest is also what the v3 learner (1823e99) gives; the bandit digests are not (v3 stored early
# completions twice), so they pin the store-once fix. Floating-point results depend on the torch / numpy build, so the
# comparison is skipped on other versions; the log-on == log-off checks always run.
GOLDEN_OFF = ("2.10.0", "1.26.4", "3e58176078ab8b9a8adefca297f3d3c26f91cfeba4eedb5520ac7a34cc8b840b")
GOLDEN_CFG = dict(algorithm="dqn", beta=0.1, learning_starts=8, batch_size=8, teacher_mix=0.5, imitation_weight=1.0,
                  target_update=5)
# seed_ttft_bandit (outcome reward), two episodes, the second reloading the first's network and replay buffer
GOLDEN_BANDIT = ("2.10.0", "1.26.4", "6af7f650e4c84f2d29826bd99b805e5637cf2b40cad98449ee83bbc91d3985ba",
                 "8b900b57558e89dd19ff5d5785dfff7de5b400998d643c2594c5be83978792d2")
GOLDEN_BANDIT_CFG = dict(learning_starts=8, batch_size=8)
TIMING_KEYS = ("train_seconds", "policy_seconds", "logging")


def _digest(ck):
    """sha256 over the network tensors, the replay buffer arrays and the learner stats (wall-clock keys and the
    additive logging block excluded)."""
    import hashlib

    import torch

    h = hashlib.sha256()
    c = torch.load(ck, weights_only=False)
    for k in sorted(c["net"]):
        h.update(k.encode())
        h.update(c["net"][k].numpy().tobytes())
    z = np.load(str(ck) + ".buffer.npz")
    for k in sorted(z.files):
        h.update(k.encode())
        h.update(np.ascontiguousarray(z[k]).tobytes())
    s = json.loads(Path(str(ck) + ".stats.json").read_text())
    h.update(json.dumps({k: v for k, v in s.items() if k not in TIMING_KEYS}, sort_keys=True).encode())
    return h.hexdigest()


def test_logging_off_matches_golden_and_logging_is_observe_only(conv_session_trace, tmp_path):
    import torch

    off = tmp_path / "off.pt"
    run(conv_session_trace, tmp_path / "a", design_with(tmp_path, **GOLDEN_CFG), ckpt_out=str(off),
        teacher=str(TEACHER))
    d_off = _digest(off)
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on = on_dir / "on.pt"
    run(conv_session_trace, tmp_path / "b", design_with(on_dir, **GOLDEN_CFG, log=True, log_interval=4),
        ckpt_out=str(on), teacher=str(TEACHER))
    assert _digest(on) == d_off  # logging never changes an action, a gradient step or a stored transition
    s_off = json.loads(Path(str(off) + ".stats.json").read_text())
    s_on = json.loads(Path(str(on) + ".stats.json").read_text())
    assert "logging" not in s_off and "train_seed" not in s_off  # off: no additive keys
    assert not any(Path(str(off) + suf).exists() for suf in (".updates.npz", ".trace.npz", ".buffer.meta.npz"))
    lg = s_on["logging"]
    assert lg["buffer_start"] == 0 and lg["buffer_rows_added"] == s_on["buffer_size_end"]
    assert sum(lg["who_counts"].values()) == s_on["decisions"]
    assert lg["decision_latency_us"]["n"] == s_on["decisions"]
    u = np.load(str(on) + ".updates.npz")
    assert int(u["n_updates"].sum()) == s_on["gradient_steps"] and u["target_syncs"].sum() >= 1
    for k in ("td_sd", "grad_norm_p95", "loss_td_mean", "loss_imit_mean", "q_top2_gap", "replay_age_mean"):
        assert k in u.files
    t = np.load(str(on) + ".trace.npz")
    assert len(t["a"]) == s_on["buffer_size_end"] and t["slot"].tolist() == list(range(len(t["a"])))
    n = len(t["a"])  # who flag agrees with the learner's own counters (every stored decision is in the trace)
    assert abs((t["who"] == 1).sum() / n - s_on["explore_frac"]) < 0.05 + 1.0 / n
    assert ((t["who"] == 2) <= (t["a"] == t["ta"])).all()  # a teacher-acted row executed the teacher's action
    assert set(t["out_rid"].tolist()) >= set(t["rid"].tolist())
    if (torch.__version__.split("+")[0], np.__version__) == GOLDEN_OFF[:2]:
        assert d_off == GOLDEN_OFF[2], "log-off run differs from the pre-change learner"


def test_bandit_two_episodes_match_golden_and_logging_is_observe_only(conv_session_trace, tmp_path):
    import torch

    digests = {}
    for tag, extra in (("off", {}), ("on", dict(log=True, log_interval=3, log_trace_frac=0.5))):
        d = tmp_path / tag
        d.mkdir()
        design = design_with(d, **GOLDEN_BANDIT_CFG, **extra)
        c1, c2 = d / "e1.pt", d / "e2.pt"
        run(conv_session_trace, d / "o1", design, ckpt_out=str(c1))
        run(conv_session_trace, d / "o2", design, ckpt_in=str(c1), ckpt_out=str(c2))
        digests[tag] = (_digest(c1), _digest(c2))
    assert digests["on"] == digests["off"]  # also across the buffer reload with its .meta.npz sidecar
    if (torch.__version__.split("+")[0], np.__version__) == GOLDEN_BANDIT[:2]:
        assert digests["off"] == GOLDEN_BANDIT[2:], "log-off bandit run differs from the pre-change learner"


def test_rng_seeds_streams():
    """train_seed None keeps the legacy seeds; any train_seed (0 included) gives learner and teacher streams that
    differ from each other, from the legacy streams and across training and scenario seeds."""
    assert rng_seeds(None, 67) == (67, 68)  # legacy, including the teacher = seed + 1 quirk
    legacy = {v for s in range(-5, 200) for v in rng_seeds(None, s)}
    seen = set()
    for ts in (0, 1, 2, 3, 4, 1000):
        for sc in (0, 1, 4, 5, 67):
            lr, tr = rng_seeds(ts, sc)
            assert lr != tr and lr not in legacy and tr not in legacy
            assert lr not in seen and tr not in seen
            seen |= {lr, tr}
    assert rng_seeds(3, 7) == rng_seeds(3, 7)


def _trace_run(trace, d, train_seed, gen_seed="42"):
    """One training run with jittered arrivals (so arrivals depend on the request generator seed) and the observe-only
    decision trace; returns (stats, trace arrays, net)."""
    import torch

    d.mkdir()
    ck = d / "net.pt"
    cfg = dict(learning_starts=10**5, epsilon_start=0.5, epsilon_end=0.5, log=True)
    if train_seed is not None:
        cfg["train_seed"] = train_seed
    run_simulator(base_args(
        trace, d / "o", num_replicas=3, global_scheduler_config_type="learned",
        learned_global_scheduler_config_policy_path=str(design_with(d, **cfg)),
        learned_global_scheduler_config_mode="training",
        learned_global_scheduler_config_checkpoint_out=str(ck),
        trace_request_generator_config_arrival_jitter="3.0", trace_request_generator_config_seed=gen_seed))
    st = json.loads(Path(str(ck) + ".stats.json").read_text())
    t = np.load(str(ck) + ".trace.npz")
    o = np.argsort(t["rid"])
    tr = {"rid": t["rid"][o], "t": t["t"][o], "who": t["who"][o], "out_rid": np.sort(t["out_rid"])}
    return st, tr, torch.load(ck, weights_only=False)["net"]


def test_train_seed_seeds_the_learner_only(conv_session_trace, tmp_path):
    import torch

    runs = {}
    for name, s in (("a1", 1), ("b1", 1), ("c2", 2), ("z0", 0), ("none", None)):
        runs[name] = _trace_run(conv_session_trace, tmp_path / name, s)
    other_sim = _trace_run(conv_session_trace, tmp_path / "gen43", None, gen_seed="43")[1]
    for name, (st, _, _) in runs.items():
        assert st.get("train_seed") == {"a1": 1, "b1": 1, "c2": 2, "z0": 0, "none": None}[name]
    ref = runs["none"][1]
    for name, (_, tr, _) in runs.items():  # the workload is the same: request ids and (jittered) arrival times
        assert np.array_equal(tr["rid"], ref["rid"]) and np.array_equal(tr["out_rid"], ref["out_rid"])
        assert np.array_equal(tr["t"], ref["t"]), name
    assert not np.array_equal(other_sim["t"], ref["t"])  # the check has teeth: arrivals follow the generator seed
    # exploration follows train_seed: same seed -> same explore decisions; different seed (or None) -> different
    who = {n: (tr["who"] == 1) for n, (_, tr, _) in runs.items()}
    assert who["a1"].sum() > 5
    assert np.array_equal(who["a1"], who["b1"])
    assert not np.array_equal(who["a1"], who["c2"])
    assert not np.array_equal(who["z0"], who["none"])  # train_seed 0 is not the legacy stream
    # network initialisation follows train_seed
    nets = {n: net for n, (_, _, net) in runs.items()}
    assert all(torch.equal(nets["a1"][k], nets["b1"][k]) for k in nets["a1"])
    assert any(not torch.equal(nets["a1"][k], nets["c2"][k]) for k in nets["a1"])
    for bad in ({"train_seed": -1}, {"train_seed": 1.5}, {"train_seed": True}, {"log": 1}, {"log_trace_frac": 0.0}):
        with pytest.raises(PolicyError):
            resolve_config(bad)


def test_manifest_identifies_the_learner(tmp_path, monkeypatch):
    import hashlib

    from evolve import rl_task

    learner = ROOT / "vidur" / "scheduler" / "global_scheduler" / "learned_global_scheduler.py"
    run_file = rl_task.run_design(str(SEED_DESIGN), tmp_path, {"train_seed": 3})
    m = json.loads(rl_task.write_manifest(tmp_path, str(SEED_DESIGN), run_file, rl_task.SMOKE_EPISODES, None,
                                          {"train_seed": 3}).read_text())
    assert m["learner_sha256"] == hashlib.sha256(learner.read_bytes()).hexdigest()
    assert m["design_sha256"] == hashlib.sha256(SEED_DESIGN.read_bytes()).hexdigest()
    assert m["code_version"] and m["config"]["train_seed"] == 3
    if m["git_commit"] is not None:  # a git checkout
        assert len(m["git_commit"]) == 40 and isinstance(m["git_dirty"], bool)
    monkeypatch.setattr(rl_task, "_git", lambda *a: None)  # no git: recorded as unknown, never fatal
    p = rl_task.learner_provenance()
    assert p["git_commit"] is None and p["git_dirty"] is None and p["learner_sha256"] == m["learner_sha256"]
