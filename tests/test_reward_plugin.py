"""The reward_plugin_path hook (EXP10's injection point) must override the built-in reward."""

import csv

from conftest import base_args, run_simulator

RL_PREFIX = "r_l_cache_aware_global_scheduler_config_"


def test_reward_plugin_overrides_builtin_reward(tiny_trace, tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        "def compute_reward(e2e_time, ttft, scheduling_delay):\n"
        "    # soft-deadline reward, bounded in [-1, 1]\n"
        "    return max(-1.0, min(1.0, 1.0 - ttft / 5.0))\n"
    )
    log = tmp_path / "transitions.csv"
    sim = run_simulator(
        base_args(
            tiny_trace,
            tmp_path / "train",
            global_scheduler_config_type="rl_cache_aware",
            **{
                RL_PREFIX + "mode": "training",
                RL_PREFIX + "hidden_dims": "32",
                RL_PREFIX + "reward_plugin_path": str(plugin),
                RL_PREFIX + "transition_log_path": str(log),
            },
        )
    )
    with open(log) as f:
        reader = csv.reader(f)
        header = next(reader)
        rewards = [float(r[header.index("reward")]) for r in reader]
    assert len(rewards) == 24
    # Built-in reward is always <= 0; the plugin gives +1 to fast requests.
    assert all(-1.0 <= r <= 1.0 for r in rewards)
    assert max(rewards) > 0.0
    # Reset the module-level plugin so later tests see the built-in reward.
    from vidur.scheduler.global_scheduler.reward_calculator import set_reward_plugin

    set_reward_plugin(None)
