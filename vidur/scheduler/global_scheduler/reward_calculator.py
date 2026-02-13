"""Reward computation for RL-based request routing."""

from vidur.entities import Request
from vidur.types import RewardType


class RewardCalculator:
    """
    Compute reward signals from completed requests.

    Supports multiple reward types:
    - E2E_TIME: End-to-end latency (completed_at - arrived_at)
    - TTFT: Time to first token (prefill_completed_at - arrived_at)

    Reward = -latency / normalization_factor

    This maps typical latencies (0.1s - 10s) to rewards in range roughly [-1, -0.01] with factor=100.
    """

    def __init__(
        self,
        normalization_factor: float = 100.0,
        reward_type: str = RewardType.E2E_TIME,
    ):
        """
        Initialize reward calculator.

        Args:
            normalization_factor: Factor to normalize latency into reasonable reward range
            reward_type: Type of reward signal ('e2e_time' or 'ttft')
        """
        self._normalization_factor = normalization_factor
        self._reward_type = reward_type

    def compute_reward(self, request: Request) -> float:
        """
        Compute reward for a completed request.

        Args:
            request: Completed request with timing data available

        Returns:
            Reward (negative normalized latency)
        """
        # Select latency metric based on reward type
        if self._reward_type == RewardType.TTFT:
            # Time to first token: prefill_completed_at - arrived_at
            latency = request.prefill_completed_at - request.arrived_at
        else:
            # Default: End-to-end time
            latency = request.e2e_time

        # Negative because we want to minimize latency
        latency_reward = -latency / self._normalization_factor

        # APPLY REWARD CLAMPING (Updated 2026-02-06):
        # Prevent "Policy Poisoning" from extreme latencies (Death Spirals)
        # 10.0 with factor=100 represents a 1000s latency limit
        MIN_REWARD = -10.0
        if latency_reward < MIN_REWARD:
            latency_reward = MIN_REWARD

        return latency_reward

    @property
    def reward_type(self) -> str:
        """Return the configured reward type."""
        return self._reward_type
