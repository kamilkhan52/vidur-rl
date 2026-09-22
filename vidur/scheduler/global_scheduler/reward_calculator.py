"""Reward computation for RL-based request routing."""

from vidur.entities import Request
from vidur.types import RewardType

_reward_plugin_fn = None


def set_reward_plugin(fn):
    """Set a custom reward function to override the default computation."""
    global _reward_plugin_fn
    _reward_plugin_fn = fn


def normalize_reward_type(reward_type) -> RewardType:
    """
    Coerce a config value ('ttft', 'e2e_time', RewardType.X, or an int) to RewardType.

    The CLI delivers `reward_type` as a plain string; comparing that string
    against the IntEnum is always False, which (before 2026-09-22) silently
    turned every 'ttft' run into an E2E-reward run.
    """
    if isinstance(reward_type, RewardType):
        return reward_type
    if isinstance(reward_type, int):
        return RewardType(reward_type)
    try:
        return RewardType.from_str(str(reward_type))
    except KeyError:
        raise ValueError(
            f"Unknown reward_type {reward_type!r}; expected one of "
            f"{[str(t) for t in RewardType]}"
        )


class RewardCalculator:
    """
    Compute reward signals from completed requests.

    Supports multiple reward types:
    - E2E_TIME: End-to-end latency (completed_at - arrived_at)
    - TTFT: Time to first token (prefill_completed_at - arrived_at)

    Reward = max(-10, -latency / normalization_factor)

    This maps typical latencies (0.1s - 10s) to rewards in range roughly [-1, -0.01] with factor=100.
    """

    MIN_REWARD = -10.0  # with factor=100 this clamps at a 1000 s latency

    def __init__(
        self,
        normalization_factor: float = 100.0,
        reward_type=RewardType.E2E_TIME,
    ):
        """
        Initialize reward calculator.

        Args:
            normalization_factor: Factor to normalize latency into reasonable reward range
            reward_type: Reward signal type ('e2e_time' or 'ttft', string or RewardType)
        """
        self._normalization_factor = normalization_factor
        self._reward_type = normalize_reward_type(reward_type)

    def latency(self, request: Request) -> float:
        """The per-request latency this calculator is configured to penalise."""
        if self._reward_type == RewardType.TTFT:
            return request.prefill_completed_at - request.arrived_at
        return request.e2e_time

    def compute_reward(self, request: Request) -> float:
        """
        Compute reward for a completed request.

        Args:
            request: Completed request with timing data available

        Returns:
            Reward (negative normalized latency, clamped at MIN_REWARD)
        """
        if _reward_plugin_fn is not None:
            return _reward_plugin_fn(
                e2e_time=request.e2e_time,
                ttft=request.prefill_completed_at - request.arrived_at,
                scheduling_delay=request.scheduling_delay,
            )

        # Negative because we want to minimize latency.
        # Clamped (since 2026-02-06) to stop extreme latencies from dominating
        # the replay buffer ("policy poisoning").
        latency_reward = -self.latency(request) / self._normalization_factor
        return max(self.MIN_REWARD, latency_reward)

    @property
    def reward_type(self) -> RewardType:
        """Return the configured reward type."""
        return self._reward_type
