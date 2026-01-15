"""Deep Q-Network agent for replica selection."""

import copy
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.

    Architecture:
    - Input: State vector (state_dim)
    - Hidden 1: 256 units + ReLU
    - Hidden 2: 128 units + ReLU
    - Hidden 3: 64 units + ReLU
    - Output: Q-values (num_actions)
    """

    def __init__(self, state_dim: int, num_actions: int):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimensionality of state vector
            num_actions: Number of actions (replicas)
        """
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values [batch_size, num_actions]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class RLAgent:
    """
    Deep Q-Network (DQN) agent for routing requests to replicas.

    Uses Double DQN to reduce overestimation bias:
    - Online network: Updated every training step
    - Target network: Updated periodically for stable targets
    """

    def __init__(self, state_dim: int, num_actions: int, config):
        """
        Initialize RL agent.

        Args:
            state_dim: Dimensionality of state vector
            num_actions: Number of actions (replicas)
            config: Configuration object with hyperparameters
        """
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._config = config

        # Build Q-networks
        self._q_network = QNetwork(state_dim, num_actions)
        self._target_network = copy.deepcopy(self._q_network)

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self._q_network.parameters(), lr=config.learning_rate
        )

        # Training counters
        self._update_count = 0

        # Load checkpoint if provided
        if config.model_checkpoint_path and os.path.exists(
            config.model_checkpoint_path
        ):
            self._load_checkpoint(config.model_checkpoint_path)

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action (replica) given state using Q-network.

        Uses greedy action selection: argmax_a Q(s, a)

        Args:
            state: State vector (numpy array)

        Returns:
            Action (replica ID as integer index)
        """
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, state_dim]

            # Forward pass
            q_values = self._q_network(state_tensor)  # [1, num_actions]

            # Select action with highest Q-value
            action = q_values.argmax(dim=1).item()

        return action

    def update(self, batch: Dict[str, torch.Tensor], gamma: float = 0.99) -> float:
        """
        Update Q-network using Double DQN.

        Double DQN update rule:
        - Action selection: a* = argmax_a' Q(s', a'; θ)
        - Value estimation: y = r + γ * Q(s', a*; θ^-)

        Args:
            batch: Dictionary with 'states', 'actions', 'rewards' tensors
            gamma: Discount factor

        Returns:
            Loss value
        """
        states = batch["states"]  # [batch_size, state_dim]
        actions = batch["actions"]  # [batch_size]
        rewards = batch["rewards"]  # [batch_size]

        # Current Q-values: Q(s, a; θ)
        current_q = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values for terminal transitions (no next state)
        # Since each routing decision is terminal in our formulation,
        # the target is just the reward
        target_q = rewards

        # Compute loss (MSE)
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), max_norm=1.0)

        self._optimizer.step()

        # Track updates
        self._update_count += 1

        return loss.item()

    def update_target_network(self):
        """
        Update target network weights from online network.

        Copies all parameters from online Q-network to target network.
        """
        self._target_network.load_state_dict(self._q_network.state_dict())

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Saves:
        - Q-network state dict
        - Target network state dict
        - Optimizer state dict
        - Update counter

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "q_network_state_dict": self._q_network.state_dict(),
            "target_network_state_dict": self._target_network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "update_count": self._update_count,
            "state_dim": self._state_dim,
            "num_actions": self._num_actions,
        }

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def _load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Restores:
        - Q-network parameters
        - Target network parameters
        - Optimizer state
        - Update counter

        Args:
            path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location="cpu")

            # Validate checkpoint
            if checkpoint.get("state_dim") != self._state_dim:
                print(
                    f"Warning: State dim mismatch. "
                    f"Checkpoint: {checkpoint.get('state_dim')}, "
                    f"Current: {self._state_dim}"
                )
                return

            if checkpoint.get("num_actions") != self._num_actions:
                print(
                    f"Warning: Num actions mismatch. "
                    f"Checkpoint: {checkpoint.get('num_actions')}, "
                    f"Current: {self._num_actions}"
                )
                return

            # Load state dicts
            self._q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self._target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._update_count = checkpoint.get("update_count", 0)

            print(f"Checkpoint loaded from {path}")
            print(f"Resuming from update {self._update_count}")

        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            print("Starting with randomly initialized weights")

    def get_update_count(self) -> int:
        """Return number of training updates performed."""
        return self._update_count
