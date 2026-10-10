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
