from vidur.types.base_int_enum import BaseIntEnum


class RewardType(BaseIntEnum):
    """Reward signal types for RL-based scheduling."""

    E2E_TIME = 1  # End-to-end time (completed_at - arrived_at)
    TTFT = 2  # Time to first token (prefill_completed_at - arrived_at)
