"""
OpenEvolve initial program: RL formulation for Vidur request scheduler.

OpenEvolve will mutate this file to search for a better RL formulation.
Two functions must always be present with these exact signatures:
  - get_rl_config() -> Dict[str, Any]
  - compute_reward(e2e_time, ttft, scheduling_delay) -> float
"""

from typing import Any, Dict


def get_rl_config() -> Dict[str, Any]:
    """Return RL hyperparameters and architecture config."""
    return {
        # Q-network architecture (comma-separated hidden dims, at least 2 layers)
        "hidden_dims": "1024,1024,512",
        # Training hyperparameters
        "learning_rate": 0.0003,
        "batch_size": 64,
        "update_frequency": 100,          # Train every N completed requests
        "target_network_update_frequency": 1000,
        "replay_buffer_size": 10000,
        # Exploration schedule (epsilon-greedy)
        # ~230 training steps per iteration (23k requests / 100 update_freq)
        # ~1150 total steps across 5 iterations
        # decay_steps=300 → epsilon near minimum by iteration 2
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 300,
        # State features
        "lookahead_k": 5,                 # Lookahead queue depth (1-15)
        # Reward normalization
        "reward_normalization_factor": 100.0,
    }


def compute_reward(e2e_time: float, ttft: float, scheduling_delay: float) -> float:
    """
    Custom reward function for RL training.

    Args:
        e2e_time: End-to-end request latency in seconds
        ttft: Time-to-first-token in seconds (prefill_completed_at - arrived_at)
        scheduling_delay: Time spent waiting in the scheduler queue

    Returns:
        Reward signal (float). Negative = penalize bad outcomes.
        Must be clamped to prevent policy poisoning from extreme latency spikes.
    """
    reward = -ttft / 100.0
    return max(-10.0, reward)
