"""Deep Q-Network agent for replica selection."""

import copy
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dims: List[int]):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimensionality of state vector
            num_actions: Number of actions (replicas)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        layers = []
        input_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_actions))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values [batch_size, num_actions]
        """
        return self.network(state)


class RLAgent:
    """
    Deep Q-Network (DQN) agent for routing requests to replicas.

    Uses Double DQN to reduce overestimation bias:
    - Online network: Updated every training step
    - Target network: Updated periodically for stable targets
    """

    def __init__(self, state_dim: int, num_actions: int, config, hidden_dims: List[int] = [512, 512, 256, 128]):
        """
        Initialize RL agent.

        Args:
            state_dim: Dimensionality of state vector
            num_actions: Number of actions (replicas)
            config: Configuration object with hyperparameters
            hidden_dims: List of hidden layer dimensions
        """
        self._state_dim = state_dim
        self._num_actions = num_actions
        self._config = config
        self._hidden_dims = hidden_dims

        # Build Q-networks
        self._q_network = QNetwork(state_dim, num_actions, hidden_dims)
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
        action, _ = self.select_action_with_diagnostics(state)
        return action

    def select_action_with_diagnostics(self, state: np.ndarray) -> tuple[int, Dict[str, float]]:
        """
        Select greedy action and return lightweight Q-value diagnostics.

        Args:
            state: State vector (numpy array)

        Returns:
            Tuple of (greedy action index, diagnostics dict)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self._q_network(state_tensor).squeeze(0)
            action = q_values.argmax().item()
            q_np = q_values.cpu().numpy()
            q_sorted = np.sort(q_np)
            q_gap = float(q_sorted[-1] - q_sorted[-2]) if len(q_sorted) >= 2 else 0.0
            diagnostics = {
                "q_mean": float(np.mean(q_np)),
                "q_std": float(np.std(q_np)),
                "q_max": float(np.max(q_np)),
                "q_min": float(np.min(q_np)),
                "q_gap": q_gap,
            }
        return action, diagnostics

    def update(self, batch: Dict[str, torch.Tensor], gamma: float = 0.99) -> Dict[str, float]:
        """
        Update Q-network using Double DQN.

        Args:
            batch: Dictionary with 'states', 'actions', 'rewards', 'next_states', 'dones' tensors
            gamma: Discount factor

        Returns:
            Diagnostic dictionary including loss and Q/TD statistics
        """
        states = batch["states"]  # [batch_size, state_dim]
        actions = batch["actions"]  # [batch_size]
        rewards = batch["rewards"]  # [batch_size]
        next_states = batch["next_states"]  # [batch_size, state_dim]
        dones = batch["dones"]  # [batch_size]

        # Current Q-values: Q(s, a; θ)
        current_q = self._q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self._q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_target = self._target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + gamma * (1.0 - dones) * next_q_target

        # Compute loss (MSE)
        loss = F.mse_loss(current_q, target_q)
        td_error = current_q - target_q

        # Optimize
        self._optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._q_network.parameters(), max_norm=1.0)

        self._optimizer.step()

        # Track updates
        self._update_count += 1

        return {
            "loss": float(loss.item()),
            "current_q_mean": float(current_q.mean().item()),
            "target_q_mean": float(target_q.mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "td_abs_max": float(td_error.abs().max().item()),
        }

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
            "hidden_dims": self._hidden_dims,
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
            self._hidden_dims = checkpoint.get("hidden_dims", [256, 128, 64])

            print(f"Checkpoint loaded from {path}")
            print(f"Resuming from update {self._update_count}")
            print(f"Hidden dims: {self._hidden_dims}")

        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            print("Starting with randomly initialized weights")

    def get_update_count(self) -> int:
        """Return number of training updates performed."""
        return self._update_count
