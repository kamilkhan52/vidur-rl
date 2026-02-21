"""
Optimized RL formulation for Vidur request scheduler.
Focuses on TTFT-centric rewards and high-capacity network architecture 
to improve scheduling efficiency while maintaining training stability.
"""

from typing import Any, Dict


def get_rl_config() -> Dict[str, Any]:
    """
    Return RL hyperparameters and architecture config.
    
    Optimizations:
    - Increased capacity to 1024,1024,1024 to capture complex cross-replica state.
    - Reduced learning rate and increased batch size for stable convergence in noisy environments.
    - Higher lookahead_k (12) to allow the agent to anticipate request bursts.
    - Adjusted update frequencies to balance exploration and policy refinement.
    """
    return {
        # High-capacity architecture for complex state mapping
        "hidden_dims": "1024,1024,1024",
        # Training stability hyperparameters
        "learning_rate": 0.0001,           # Lower LR for smoother policy updates
        "batch_size": 128,                 # Larger batches to reduce gradient noise
        "update_frequency": 50,            # Train more frequently to utilize online data
        "target_network_update_frequency": 500, # Frequent target updates for tracking
        "replay_buffer_size": 20000,
        # Exploration schedule
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 800,        # Reach exploitation phase mid-run (approx 2000 requests total)
        # Features
        "lookahead_k": 12,                 # Deep queue visibility
        "reward_normalization_factor": 1.0, # Handled explicitly in compute_reward
    }


def compute_reward(e2e_time: float, ttft: float, scheduling_delay: float) -> float:
    """
    Custom reward function optimized for LLM scheduling.

    Args:
        e2e_time: End-to-end request latency in seconds
        ttft: Time-to-first-token in seconds
        scheduling_delay: Time spent waiting in the scheduler queue

    Returns:
        A TTFT-heavy reward signal. TTFT is the primary bottleneck in 
        high-load scenarios; penalizing it directly encourages the agent
        to balance prefill workloads across replicas.
    """
    # Weight TTFT heavily as it is the most sensitive metric for user experience.
    # We also include scheduling_delay as it provides a direct signal of 
    # backpressure in the replica's queue.
    # Scale: A TTFT of 1.0s results in a reward of -3.0.
    # A TTFT of 3.3s+ hits the -10.0 floor.
    reward = -(ttft * 3.0 + scheduling_delay * 1.0)
    
    # Strict clamping to prevent "policy poisoning" where a single extreme 
    # latency spike (e.g., 60s) would otherwise create a massive gradient 
    # that collapses the Q-network.
    return max(-10.0, reward)