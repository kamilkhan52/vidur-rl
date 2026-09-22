"""The configured reward type must actually select the latency it names.

Before 2026-09-22 the CLI string 'ttft' was compared with the RewardType
IntEnum, which is never equal, so every 'ttft' run silently trained on E2E.
"""

import pytest

from vidur.entities import Request
from vidur.scheduler.global_scheduler.reward_calculator import (
    RewardCalculator,
    normalize_reward_type,
)
from vidur.types import RewardType


def completed_request(arrived=0.0, ttft=2.0, e2e=10.0):
    r = Request(arrived_at=arrived, num_prefill_tokens=10, num_decode_tokens=3,
                block_hash_ids=None, block_size=None, session_id=0)
    r.on_batch_schedule(arrived + 1.0)
    r.on_batch_end(arrived + ttft, 10)  # prefill done -> first token at `ttft`
    r.on_batch_end(arrived + ttft + 1.0, 1)
    r.on_batch_end(arrived + e2e, 1)     # last token -> completed at `e2e`
    assert r.completed
    return r


def test_normalize_reward_type():
    assert normalize_reward_type("ttft") == RewardType.TTFT
    assert normalize_reward_type("TTFT") == RewardType.TTFT
    assert normalize_reward_type("e2e_time") == RewardType.E2E_TIME
    assert normalize_reward_type(RewardType.TTFT) == RewardType.TTFT
    assert normalize_reward_type(int(RewardType.TTFT)) == RewardType.TTFT
    with pytest.raises(ValueError):
        normalize_reward_type("latency")


def test_string_ttft_selects_ttft_latency():
    r = completed_request(ttft=2.0, e2e=10.0)
    assert RewardCalculator(reward_type="ttft").compute_reward(r) == pytest.approx(-0.02)
    assert RewardCalculator(reward_type="e2e_time").compute_reward(r) == pytest.approx(-0.10)
    assert RewardCalculator(reward_type=RewardType.TTFT).latency(r) == pytest.approx(2.0)


def test_reward_is_clamped():
    r = completed_request(ttft=5000.0, e2e=6000.0)
    assert RewardCalculator(reward_type="ttft").compute_reward(r) == RewardCalculator.MIN_REWARD
    assert RewardCalculator(reward_type="e2e_time").compute_reward(r) == RewardCalculator.MIN_REWARD
