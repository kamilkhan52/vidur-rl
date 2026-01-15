"""Reward computation for RL-based request routing."""

from vidur.entities import Request


class RewardCalculator:
    """
    Compute reward signals from completed requests.

    Primary objective: Minimize end-to-end latency.
    Reward = -latency / normalization_factor

    This maps typical latencies (0.1s - 10s) to rewards in range roughly [-1, -0.01] with factor=100.
    """

    def __init__(self, normalization_factor: float = 100.0):
        """
        Initialize reward calculator.

        Args:
            normalization_factor: Factor to normalize latency into reasonable reward range
        """
        self._normalization_factor = normalization_factor

    def compute_reward(self, request: Request) -> float:
        """
        Compute reward for a completed request.

        Args:
            request: Completed request with e2e_time available

        Returns:
            Reward (negative normalized latency)
        """
        # Negative because we want to minimize latency
        # e2e_time is in seconds
        latency_reward = -request.e2e_time / self._normalization_factor

        # NOTE: Clipping removed to preserve learning signal at all latency ranges.
        # Gradient clipping in the optimizer handles training stability instead.
        # Natural scaling: 1s → -0.01, 10s → -0.1, 100s → -1.0, 1000s → -10.0 (with factor=100)

        return latency_reward
