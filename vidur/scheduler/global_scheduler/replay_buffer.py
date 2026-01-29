"""Experience replay buffer for off-policy reinforcement learning."""

import csv
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores (state, action, reward) tuples for training the RL agent.
    Uses circular buffer with fixed capacity.

    Optionally logs all transitions to a CSV file for analysis.
    """

    def __init__(
        self,
        capacity: int = 10000,
        log_file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            log_file_path: Optional path to CSV file for logging transitions
            feature_names: Optional list of feature names for CSV headers
        """
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._transition_count = 0

        # CSV logging setup
        self._log_file_path = log_file_path
        self._feature_names = feature_names
        self._csv_file = None
        self._csv_writer = None

        if log_file_path:
            self._init_csv_logging()

    def _init_csv_logging(self) -> None:
        """Initialize CSV file for logging transitions."""
        # Create directory if needed
        log_dir = os.path.dirname(self._log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Open file and write headers
        self._csv_file = open(self._log_file_path, "w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)

        # Build header row
        headers = ["transition_idx", "timestamp"]
        if self._feature_names:
            headers.extend(self._feature_names)
        else:
            # Generic feature names if not provided
            headers.append("state_vector")
        headers.extend(["action", "reward"])

        self._csv_writer.writerow(headers)
        print(f"[ReplayBuffer] Logging transitions to: {self._log_file_path}")

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
            "state": state,
            "action": action,
            "reward": reward,
        }
        self._position = (self._position + 1) % self._capacity

        # Log to CSV if enabled
        if self._csv_writer:
            self._log_transition(state, action, reward)

        self._transition_count += 1

    def _log_transition(
        self, state: np.ndarray, action: int, reward: float
    ) -> None:
        """Write a transition to the CSV log file."""
        row = [self._transition_count, time.time()]

        if self._feature_names:
            # Individual feature columns
            row.extend(state.tolist())
        else:
            # Single column with stringified state
            row.append(state.tolist())

        row.extend([action, reward])
        self._csv_writer.writerow(row)

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
        states = np.array([t["state"] for t in batch])
        actions = np.array([t["action"] for t in batch])
        rewards = np.array([t["reward"] for t in batch])

        return {
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(actions).long(),
            "rewards": torch.from_numpy(rewards).float(),
        }

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self._buffer)

    def close(self) -> None:
        """Close the CSV log file if open."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __del__(self):
        """Ensure CSV file is closed on deletion."""
        self.close()
