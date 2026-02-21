"""
Expert RL Formulation for Vidur Request Scheduler.
Optimized for the 'tension zone' where TTFT and E2E latency trade-offs are critical.
Focuses on preventing policy collapse while providing a stronger gradient signal.
"""

from typing import Any, Dict


def get_rl_config() -> Dict[str, Any]:
    """
    Returns an optimized RL configuration.
    
    - Architecture: Deep and wide (1024x3) to capture complex replica states.
    - Learning Rate: Slightly higher to escape the poor initial heuristic quickly.
    - Update Frequency: Increased (40 vs 100) to ensure the model sees more gradient steps
      within the limited 2000-request window.
    - Lookahead: Increased to 10 to better anticipate incoming burst patterns.
    """
    return {
        # High-capacity architecture for complex state interactions
        "hidden_dims": "1024,1024,1024",
        # Training hyperparameters
        "learning_rate": 0.0004,
        "batch_size": 128,
        "update_frequency": 40,           # More frequent updates for faster convergence
        "target_network_update_frequency": 400,
        "replay_buffer_size": 15000,
        # Exploration: Decay over the first 40-50% of the run
        "epsilon_start": 1.0,
        "epsilon_end": 0.02,
        "epsilon_decay_steps": 1000,
        # State features
        "lookahead_k": 10,                # Deeper lookahead for better prefill/decode balancing
        "reward_normalization_factor": 1.0, # Handled explicitly in compute_reward
    }


def compute_reward(e2e_time: float, ttft: float, scheduling_delay: float) -> float:
    """
    Optimized reward function focusing on Time-To-First-Token (TTFT).
    
    The previous reward (-ttft / 100.0) provided a very weak signal (e.g., -0.01 for 1s TTFT).
    This version scales the reward to use more of the available range (-10.0 to 0.0),
    providing a sharper gradient for the agent to distinguish between routing choices.
    
    Args:
        e2e_time: End-to-end request latency.
        ttft: Time-to-first-token (crucial for perceived responsiveness).
        scheduling_delay: Time spent in the global queue.
    """
    # Weight TTFT heavily as it is the primary differentiator in scheduling quality.
    # Penalize scheduling delay specifically to discourage queue stagnation.
    # Scaling factor 4.0 ensures a 2s TTFT (bad) is -8.0, while 0.2s (good) is -0.8.
    reward = -(ttft * 4.0 + scheduling_delay * 1.0)
    
    # Strict clamping to prevent 'policy poisoning' from extreme outliers (tail latency).
    return max(-10.0, reward)