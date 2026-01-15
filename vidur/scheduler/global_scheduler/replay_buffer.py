"""Experience replay buffer for off-policy reinforcement learning."""

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores (state, action, reward) tuples for training the RL agent.
    Uses circular buffer with fixed capacity.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self._capacity = capacity
        self._buffer = []
        self._position = 0

    def add(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: State representation (numpy array)
            action: Action taken (replica ID)
            reward: Reward received (negative latency)
        """
        if len(self._buffer) < self._capacity:
            self._buffer.append(None)

        self._buffer[self._position] = {
            'state': state,
            'action': action,
            'reward': reward
        }
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with 'states', 'actions', 'rewards' tensors
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions, "
                f"buffer only contains {len(self._buffer)}"
            )

        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        # Stack numpy arrays first, then convert to tensor (much faster)
        states = np.array([t['state'] for t in batch])
        actions = np.array([t['action'] for t in batch])
        rewards = np.array([t['reward'] for t in batch])

        return {
            'states': torch.from_numpy(states).float(),
            'actions': torch.from_numpy(actions).long(),
            'rewards': torch.from_numpy(rewards).float(),
        }

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self._buffer)
