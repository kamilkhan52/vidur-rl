"""evolve.analysis.rl_offline: pure helpers on tiny synthetic fixtures."""

import numpy as np

from evolve.analysis import rl_offline as ro

SRC = '''
def replica_features(request, replica, replicas, ctx):
    a = 1
    return [
        # --- request / cluster (shared) ---
        a,
        a * 2,
        # --- cache state: here ---
        a + 1,
        # --- the deterministic zero-mean externality price of this action ---
        0.0,
    ]
'''


def test_feature_groups_follow_headers():
    g = ro.feature_groups(SRC, 4)
    assert g == [("request / cluster", [0, 1]), ("cache state", [2]),
                 ("the deterministic zero-mean externality price of this action", [3])]
    assert ro.feature_groups(SRC, 5) is None  # length mismatch -> None
    assert ro.feature_groups("x = 1", 4) is None


def test_episode_labels_and_wrap_detection():
    lab = ro.episode_labels([3, 5, 9], [0, 3, 5], 9)
    assert lab.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert ro.episode_labels([3, 5, 9], [0, 2, 5], 9) is None  # not reloaded
    assert ro.episode_labels([3, 5, 8], [0, 3, 5], 9) is None  # buffer length mismatch


def test_dedup_removes_exact_duplicates_only():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    r = np.array([0.5, 0.5, 0.6, 0.5])
    assert ro.dedup_index(x, r).tolist() == [0, 2, 3]


def test_state_feature_mask():
    x3 = np.zeros((5, 3, 2))
    x3[:, :, 0] = np.arange(5)[:, None]  # same on every replica
    x3[:, :, 1] = np.arange(3)[None, :]  # differs across replicas
    assert ro.state_feature_mask(x3).tolist() == [True, False]


def test_group_folds_partition_whole_episodes():
    lab = np.repeat(np.arange(8), [5, 3, 7, 2, 6, 4, 1, 8])
    folds = ro.group_folds(lab, 3)
    assert sum(f.sum() for f in folds) == len(lab)
    assert not np.any(np.sum(folds, 0) != 1)
    for f in folds:  # no episode split across folds
        assert set(lab[f]).isdisjoint(set(lab[~f]))


def test_gamma_test_recovers_noise_variance():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (6000, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.3, 6000)
    g = ro.gamma_test(X, y, k=10)
    assert abs(g["noise_var"] - 0.09) < 0.02
    assert 0.0 < g["ceiling_ev"] < 1.0


def test_fit_regressor_linear_and_mlp_fit_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, 3000)
    _, pv = ro.fit_regressor(X[:2000], y[:2000], X[2000:], width=0)
    assert ro.ev(y[2000:], pv) > 0.99
    _, pv = ro.fit_regressor(X[:2000], y[:2000], X[2000:], width=16, max_epochs=30, batch=64)
    assert ro.ev(y[2000:], pv) > 0.95


def test_episode_labels_from_logged_ranges_survive_wrap():
    # capacity 5; episode 0 writes slots 0-3, episode 1 writes 4,0,1 (wraps), so slots 2-3 keep episode 0
    lab = ro.episode_labels_from_ranges([0, 4], [4, 3], 5, 5)
    assert lab.tolist() == [1, 1, 0, 0, 1]
    assert ro.episode_labels_from_ranges([0, None], [4, 3], 5, 5) is None
    assert ro.episode_labels_from_ranges([2], [2], 5, 100).tolist() == [-1, -1, 0, 0, -1]  # unlogged rows -> -1


def test_gamma_lower_bound_key_and_old_name():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (3000, 2))
    y = np.sin(3 * X[:, 0]) + rng.normal(0, 0.3, 3000)
    g = ro.gamma_test(X, y, k=10)
    assert "ceiling_ev_upper_from_gamma1" not in g
    assert g["ceiling_ev_lower_from_gamma1"] <= g["ceiling_ev"]  # gamma(1) over-estimates the noise
    assert ro.ceiling_lower(g) == g["ceiling_ev_lower_from_gamma1"]
    assert ro.ceiling_lower({"ceiling_ev_upper_from_gamma1": 0.6}) == 0.6  # files written before the rename


def test_analyse_data_end_to_end_on_a_synthetic_buffer(tmp_path, monkeypatch):
    """Full (non-light) analysis on a tiny fake bandit buffer: both argmax targets, n-curve, width selection, md."""
    monkeypatch.setattr(ro, "WIDTHS", [0, 16])
    monkeypatch.setattr(ro, "FRACTIONS", [0.5, 1.0])
    rng = np.random.default_rng(3)
    n_ep, per, N, F = 3, 600, 3, 4
    T = n_ep * per
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    x[:, :, 0] = x[:, :1, 0]  # a state feature (same on every replica)
    x[:, :, 3] = np.clip(x[:, :, 3], -2.9, 2.9)  # the price feature
    ta = x[:, :, 1].argmax(1)
    a = np.where(rng.random(T) < 0.5, ta, rng.integers(0, N, T))
    xa = x[np.arange(T), a]
    r = (xa[:, 1] - 0.5 * xa[:, 2] + 0.3 * xa[:, 0] - xa[:, 3] + rng.normal(0, 0.3, T)).astype(np.float32)
    buf = tmp_path / "net_ep2.pt.buffer.npz"
    np.savez(buf, x=x, a=a, r=r, ta=ta, g=np.zeros(T, np.float32))
    design = tmp_path / "design.py"
    design.write_text(SRC)
    eps = [{"episode": f"e{k}", "learner": {"buffer_size_end": (k + 1) * per, "buffer_loaded_from_previous_episode":
                                            k * per}} for k in range(n_ep)]
    out = ro.analyse_data({"candidate": "fake/x", "buffer": str(buf), "fitness": None}, None, buf, eps, str(design),
                          log=lambda *_: None)
    assert out["status"] == "ok", out.get("status")
    ti = out["teacher_imitation"]
    assert set(ti["reward_model_argmax_eq_teacher_by_target"]) == {"r", "r_out"}
    assert ti["reward_model_argmax_eq_teacher"] == ti["reward_model_argmax_eq_teacher_by_target"]["r_out"]
    assert out["noise_within_episode"]["r_out"]["rows_per_episode_mean"] == per
    assert list(out["noise_n_curve"]["points"]) == [str(per)]  # the 10k/20k/40k points exceed T
    assert out["width_selection"]["val_ev_spread"] >= 0 and out["best_width"] in (0, 16)
    assert "ceiling_ev_lower_from_gamma1" in out["noise"]["r"]
    assert "note" in out["data_curve"]
    md = ro.to_markdown({"generated": "now", "candidates": {"fake/x": out}})
    assert "what the learner maximised" in md and "Pooled ceiling vs rows used" in md and "adding episodes" in md
