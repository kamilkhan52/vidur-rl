"""Regression tests for the RL global scheduler defects fixed on 2026-09-21.

1. ``hidden_dims`` never reached the agent (every run trained 512,512,256,128).
2. ``inference_epsilon`` was accepted but ignored (inference was always greedy).
3. Replay transitions were all terminal (done=True) because the successor state
   was read from the global queue at completion time, which is always drained.
4. The transition CSV header omitted the next-state/action/reward/done columns.
5. ``set_seeds`` did not seed torch.
"""

import csv

import numpy as np
import pytest
import torch

from conftest import base_args, build_simulator, run_simulator
from vidur.scheduler.global_scheduler.rl_cache_aware_global_scheduler import (
    expected_state_dim,
    parse_hidden_dims,
)

RL_PREFIX = "r_l_cache_aware_global_scheduler_config_"


def rl_args(trace, out_dir, mode, **kw):
    overrides = {
        "global_scheduler_config_type": "rl_cache_aware",
        RL_PREFIX + "mode": mode,
        RL_PREFIX + "hidden_dims": "64,32",
        RL_PREFIX + "update_frequency": "5",
        RL_PREFIX + "batch_size": "4",
        RL_PREFIX + "epsilon_decay_steps": "10",
    }
    overrides.update({RL_PREFIX + k: v for k, v in kw.items()})
    return base_args(trace, out_dir, **overrides)


def test_parse_hidden_dims():
    assert parse_hidden_dims("1024,1024,512") == [1024, 1024, 512]
    assert parse_hidden_dims("64, 32") == [64, 32]
    assert parse_hidden_dims([256, 128]) == [256, 128]
    assert parse_hidden_dims(None) == [512, 512, 256, 128]
    with pytest.raises(ValueError):
        parse_hidden_dims("")
    with pytest.raises(ValueError):
        parse_hidden_dims("512,-1")


def test_expected_state_dim_matches_state_tracker():
    # Values documented in StateTracker for M=4, K=5.
    assert expected_state_dim(4, 5, True) == 72
    assert expected_state_dim(4, 5, False) == 52
    # K matters: the old hard-coded formula silently assumed K=5.
    assert expected_state_dim(4, 15, True) == 5 + 20 + 4 + 14 * 9 + 4 + 3


def test_set_seeds_seeds_torch():
    from vidur.utils.random import set_seeds

    set_seeds(123)
    a = torch.rand(3)
    set_seeds(123)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_hidden_dims_reach_agent_and_checkpoint(tiny_trace, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    sim = run_simulator(
        rl_args(tiny_trace, tmp_path / "train", "training", model_checkpoint_path=str(ckpt))
    )
    sched = sim._scheduler
    assert sched.hidden_dims == [64, 32]
    first_layer = sched._rl_agent._q_network.network[0]
    assert first_layer.out_features == 64

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved["hidden_dims"] == [64, 32]
    shapes = [tuple(v.shape) for k, v in saved["q_network_state_dict"].items() if k.endswith("weight")]
    assert shapes[0][0] == 64 and shapes[1] == (32, 64)

    # Loading with a different configured architecture must adopt the checkpoint's.
    sim2 = build_simulator(
        rl_args(
            tiny_trace,
            tmp_path / "infer",
            "inference",
            hidden_dims="512,512,256,128",
            model_checkpoint_path=str(ckpt),
        )
    )
    agent = sim2._scheduler._rl_agent
    assert sim2._scheduler.hidden_dims == [64, 32]
    assert agent.get_update_count() == saved["update_count"]
    for k, v in saved["q_network_state_dict"].items():
        assert torch.equal(agent._q_network.state_dict()[k], v)


def test_inference_epsilon_is_applied(tiny_trace, tmp_path):
    greedy = run_simulator(rl_args(tiny_trace, tmp_path / "g", "inference", inference_epsilon="0.0"))
    assert greedy._scheduler.total_random_decisions == 0
    assert greedy._scheduler.total_greedy_decisions == 24

    explore = run_simulator(rl_args(tiny_trace, tmp_path / "e", "inference", inference_epsilon="1.0"))
    assert explore._scheduler.total_random_decisions == 24
    assert explore._scheduler.total_greedy_decisions == 0


def test_transitions_are_chained_to_the_next_decision(tiny_trace, tmp_path):
    log = tmp_path / "transitions.csv"
    sim = run_simulator(
        rl_args(tiny_trace, tmp_path / "train", "training", transition_log_path=str(log))
    )
    sched = sim._scheduler
    assert not sched._awaiting_next_state and not sched._pending_transitions

    with open(log) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert rows, "no transitions logged"
    assert all(len(r) == len(header) for r in rows), "header/row column mismatch"
    assert header[-3:] == ["action", "reward", "done"]

    col = {name: i for i, name in enumerate(header)}
    first_next = next(i for i, h in enumerate(header) if h.startswith("next_"))
    state_cols = list(range(col["decision_idx"] + 1, first_next))
    next_cols = list(range(first_next, col["action"]))
    assert len(state_cols) == len(next_cols) == sched._state_tracker.get_state_dim()

    by_decision = {int(r[col["decision_idx"]]): r for r in rows}
    assert len(by_decision) == 24 == len(rows)
    last = max(by_decision)
    for d, r in by_decision.items():
        done = int(r[col["done"]])
        if d == last:
            assert done == 1
            assert all(float(r[i]) == 0.0 for i in next_cols)
        else:
            assert done == 0
            nxt = by_decision[d + 1]
            np.testing.assert_allclose(
                [float(r[i]) for i in next_cols],
                [float(nxt[i]) for i in state_cols],
                rtol=1e-6,
                err_msg=f"next_state of decision {d} != state of decision {d + 1}",
            )
    assert sum(int(r[col["done"]]) for r in rows) == 1
    assert len(sched.replay_buffer) == 24
