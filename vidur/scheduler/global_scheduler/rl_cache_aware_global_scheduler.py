"""
Cache-aware RL-based global scheduler for LLM request routing.

This scheduler uses reinforcement learning to route requests across replicas,
learning to balance load, minimize prefill-decode interference, and exploit
KV cache affinity.
"""

import math
import os
from typing import Dict, List, Tuple

import torch

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.entities.replica import Replica
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.replay_buffer import ReplayBuffer
from vidur.scheduler.global_scheduler.reward_calculator import RewardCalculator
from vidur.scheduler.global_scheduler.rl_agent import RLAgent
from vidur.scheduler.global_scheduler.state_tracker import StateTracker
from vidur.types.replica_id import ReplicaId


class RLCacheAwareGlobalScheduler(BaseGlobalScheduler):
    """
    RL-based global scheduler with cache-aware state representation.

    Key responsibilities:
    1. Extract state features for each request-routing decision
    2. Query RL policy network for action selection
    3. Track pending transitions and collect rewards
    4. Update policy network (if in training mode)
    5. Maintain shadow replica state for accurate state extraction
    """

    def __init__(
        self,
        config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
    ):
        """
        Initialize RL cache-aware global scheduler.

        Args:
            config: Simulation configuration
            replicas: Map of replica ID to Replica object
        """
        super().__init__(config, replicas)

        # RL Configuration
        rl_config = config.cluster_config.global_scheduler_config
        self._mode = rl_config.mode

        # Log mode and exploration settings
        if self._mode == "inference":
            print(f"[RL Scheduler] Mode: INFERENCE - Epsilon-greedy DISABLED (always exploit)")
        else:
            print(f"[RL Scheduler] Mode: TRAINING - Epsilon-greedy ENABLED (exploration active)")

        # Check if we're loading a pre-trained model and inspect its state_dim
        enable_prefix_caching = config.cluster_config.cache_config.enable_prefix_caching if hasattr(config, 'cluster_config') else False
        checkpoint_state_dim = None
        
        if rl_config.model_checkpoint_path:


            if os.path.exists(rl_config.model_checkpoint_path):
                try:
                    checkpoint = torch.load(rl_config.model_checkpoint_path, map_location='cpu')
                    checkpoint_state_dim = checkpoint.get('state_dim')
                    if checkpoint_state_dim:
                        print(f"[RL Scheduler] Found checkpoint with state_dim={checkpoint_state_dim}")
                except Exception as e:
                    print(f"[RL Scheduler] Could not read checkpoint state_dim: {e}")
        
        # Determine final enable_prefix_caching based on checkpoint or config
        if checkpoint_state_dim is not None:
            # Infer from checkpoint dimension (72 = with cache, 52 = without cache for 4 replicas)
            # Includes session affinity features (+M)
            M = len(replicas)
            expected_dim_with_cache = 5 + M * 5 + M + 4 * (5 + M) + M + 3
            expected_dim_without_cache = 5 + M * 5 + 4 * 5 + M + 3
            
            if checkpoint_state_dim == expected_dim_with_cache:
                enable_prefix_caching = True
                if not (config.cluster_config.cache_config.enable_prefix_caching if hasattr(config, 'cluster_config') else False):
                    print(f"[RL Scheduler] ⚠️  WARNING: Checkpoint uses cache-aware state (dim={checkpoint_state_dim}), but prefix caching is DISABLED in config")
                    print(f"[RL Scheduler] Overriding to use cache-aware state to match checkpoint")
            elif checkpoint_state_dim == expected_dim_without_cache:
                enable_prefix_caching = False
                if config.cluster_config.cache_config.enable_prefix_caching if hasattr(config, 'cluster_config') else False:
                    print(f"[RL Scheduler] ⚠️  WARNING: Checkpoint uses simplified state (dim={checkpoint_state_dim}), but prefix caching is ENABLED in config")
                    print(f"[RL Scheduler] Overriding to use simplified state to match checkpoint")
            else:
                print(f"[RL Scheduler] ⚠️  WARNING: Checkpoint state_dim={checkpoint_state_dim} doesn't match expected dimensions")
                print(f"[RL Scheduler]    Expected: {expected_dim_with_cache} (with cache) or {expected_dim_without_cache} (without cache)")
        
        if enable_prefix_caching:
            print(f"[RL Scheduler] Using cache-aware state (with prefix caching features)")
        else:
            print(f"[RL Scheduler] Using simplified state (without prefix caching features)")

        # State tracker - maintains shadow state of pending work
        self._state_tracker = StateTracker(
            num_replicas=len(replicas),
            replica_schedulers=self._replica_schedulers,
            enable_prefix_caching=enable_prefix_caching,
        )

        # RL Agent (Policy Network)
        self._rl_agent = RLAgent(
            state_dim=self._state_tracker.get_state_dim(),
            num_actions=len(replicas),
            config=rl_config,
        )

        # Initialize epsilon parameters BEFORE loading checkpoint (so we have defaults)
        if self._mode == "training":
            self._num_training_steps = 0
            self._epsilon_start = rl_config.epsilon_start
            self._epsilon_end = rl_config.epsilon_end
            self._epsilon_decay_steps = rl_config.epsilon_decay_steps

        # Log model loading status
        if rl_config.model_checkpoint_path:

            if os.path.exists(rl_config.model_checkpoint_path):
                print(f"[RL Scheduler] Loaded trained model from: {rl_config.model_checkpoint_path}")

                # Try to load epsilon state if continuing training
                if self._mode == "training":
                    if rl_config.reset_epsilon_on_load:
                        # User requested to reset epsilon state - use config values and start from step 0
                        current_epsilon = self._get_epsilon()
                        print(f"[RL Scheduler] Reset epsilon state (reset_epsilon_on_load=True)")
                        print(f"[RL Scheduler] Using config values: start={self._epsilon_start}, end={self._epsilon_end}, decay_steps={self._epsilon_decay_steps}")
                        print(f"[RL Scheduler] Starting from step 0, initial epsilon={current_epsilon:.4f}")
                    else:
                        # Normal behavior: load epsilon state from checkpoint
                        epsilon_path = rl_config.model_checkpoint_path.replace('.pt', '_epsilon.pt')
                        if os.path.exists(epsilon_path):

                            epsilon_state = torch.load(epsilon_path, map_location='cpu')
                            self._num_training_steps = epsilon_state.get('num_training_steps', 0)
                            self._epsilon_start = epsilon_state.get('epsilon_start', self._epsilon_start)
                            self._epsilon_end = epsilon_state.get('epsilon_end', self._epsilon_end)
                            self._epsilon_decay_steps = epsilon_state.get('epsilon_decay_steps', self._epsilon_decay_steps)
                            current_epsilon = self._get_epsilon()
                            print(f"[RL Scheduler] Restored epsilon state: step {self._num_training_steps}, epsilon {current_epsilon:.4f}")
                        else:
                            print(f"[RL Scheduler] No epsilon state found, starting epsilon decay from beginning")
            else:
                print(f"[RL Scheduler] Model checkpoint path provided but file not found: {rl_config.model_checkpoint_path}")
                if self._mode == "inference":
                    print(f"[RL Scheduler] WARNING: Running inference with randomly initialized model!")

        # Reward calculator
        self._reward_calculator = RewardCalculator(
            normalization_factor=rl_config.reward_normalization_factor,
            reward_type=rl_config.reward_type,
        )
        print(f"[RL Scheduler] Reward type: {rl_config.reward_type}")

        # Pending transitions (awaiting rewards)
        # Maps request.id -> {'state': state, 'action': replica_id}
        self._pending_transitions = {}

        # Training infrastructure
        if self._mode == "training":
            # Set up transition logging if path provided
            transition_log_path = rl_config.transition_log_path if rl_config.transition_log_path else None
            feature_names = self._state_tracker.get_feature_names() if transition_log_path else None

            self._replay_buffer = ReplayBuffer(
                capacity=rl_config.replay_buffer_size,
                log_file_path=transition_log_path,
                feature_names=feature_names,
            )
            self._update_counter = 0
            self._update_frequency = rl_config.update_frequency
            self._target_update_frequency = (
                rl_config.target_network_update_frequency
            )

            # Training metrics
            self._training_losses = []
            self._num_requests_seen = 0
            # Note: _num_training_steps and epsilon params initialized earlier (before loading checkpoint)

            # Checkpoint saving
            self._checkpoint_frequency = 10000  # Save every 10k requests
            self._last_checkpoint_request = 0

            # Training log file

            os.makedirs("/tmp/rl_training_logs", exist_ok=True)
            self._training_log_file = open("/tmp/rl_training_logs/training.log", "w", buffering=1)
            self._training_log_file.write("Training started\n")

        # Metrics tracking
        self._episode_rewards = []
        self._recent_rewards = []  # Track last 100 rewards for moving average

    def _get_epsilon(self) -> float:
        """
        Get current exploration epsilon with exponential decay.

        Uses formula: epsilon = end + (start - end) * exp(-decay_constant * step)
        where decay_constant is chosen so epsilon reaches ~1% of initial range
        at epsilon_decay_steps.

        Returns:
            Epsilon value for ε-greedy exploration
        """
        if self._mode != "training":
            return 0.0  # No exploration during inference

        # Exponential decay: reaches ~1% of (start - end) at epsilon_decay_steps
        # decay_constant = -ln(0.01) / decay_steps ≈ 4.605 / decay_steps
        decay_constant = 4.605 / self._epsilon_decay_steps
        epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(
            -decay_constant * self._num_training_steps
        )
        return epsilon

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        """
        Main scheduling logic - called on request arrivals.

        Process:
        1. Sort requests by arrival time (FIFO)
        2. For each request:
           a. Extract state features
           b. Select action (replica) via RL agent (ε-greedy if training)
           c. Store pending transition
           d. Update shadow state
        3. Return request-replica mapping

        Returns:
            List of (replica_id, request) tuples
        """
        self.sort_requests()
        request_mapping = []

        # Convert replica IDs to list for indexing
        replica_ids = list(self._replica_schedulers.keys())

        while self._request_queue:
            request = self._request_queue.pop(0)

            # 1. Extract state
            state = self._state_tracker.extract_state(request, self._request_queue)

            # 2. Select action (replica)
            if self._mode == "training":
                epsilon = self._get_epsilon()
                if self._random_number_generator.random() < epsilon:
                    # Explore: random action
                    action_idx = self._random_number_generator.randint(
                        0, len(replica_ids) - 1
                    )
                else:
                    # Exploit: use RL agent
                    action_idx = self._rl_agent.select_action(state)
            else:
                # Inference mode: always exploit
                action_idx = self._rl_agent.select_action(state)

            replica_id = replica_ids[action_idx]

            # 3. Store pending transition (awaiting reward)
            self._pending_transitions[request.id] = {
                'state': state,
                'action': action_idx,  # Store index, not ReplicaId
            }

            # 4. Update shadow state
            self._state_tracker.on_request_assigned(request, replica_id)

            request_mapping.append((replica_id, request))

            # Track number of requests for epsilon decay
            if self._mode == "training":
                self._num_requests_seen += 1

        return request_mapping

    def on_request_end(self, request: Request):
        """
        Called when request completes - collect reward and train.

        Args:
            request: Completed request
        """
        if request.id not in self._pending_transitions:
            return

        transition = self._pending_transitions.pop(request.id)

        # Compute reward from completed request
        reward = self._reward_calculator.compute_reward(request)

        # Track rewards for logging
        self._episode_rewards.append(reward)
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 100:
            self._recent_rewards.pop(0)

        # Add to replay buffer and train (if in training mode)
        if self._mode == "training":
            self._replay_buffer.add(
                state=transition['state'],
                action=transition['action'],
                reward=reward,
            )

            # Debug: Log first time we get enough samples
            if len(self._replay_buffer) == self._config.cluster_config.global_scheduler_config.batch_size:
                msg = f"[Training] Replay buffer ready with {len(self._replay_buffer)} samples\n"
                print(msg, flush=True)
                self._training_log_file.write(msg)

            # Periodic training updates
            self._update_counter += 1
            if (
                self._update_counter % self._update_frequency == 0
                and len(self._replay_buffer) >= self._config.cluster_config.global_scheduler_config.batch_size
            ):
                self._train_step()
                msg = f"[Training] Completed update at request {self._update_counter}\n"
                print(msg, flush=True)
                self._training_log_file.write(msg)

            # Periodic checkpoint saving
            if (
                self._config.cluster_config.global_scheduler_config.model_checkpoint_path
                and self._num_requests_seen - self._last_checkpoint_request >= self._checkpoint_frequency
            ):
                self._save_checkpoint()
                self._last_checkpoint_request = self._num_requests_seen

        # Update shadow state
        self._state_tracker.on_request_completed(request)

    def _train_step(self):
        """
        Perform one training step using a batch from replay buffer.
        """
        batch_size = self._config.cluster_config.global_scheduler_config.batch_size
        batch = self._replay_buffer.sample(batch_size)
        


        loss = self._rl_agent.update(batch)

        # Track loss for logging
        self._training_losses.append(loss)
        self._num_training_steps += 1

        # Periodic target network update
        if self._rl_agent.get_update_count() % self._target_update_frequency == 0:
            self._rl_agent.update_target_network()
            msg = f"[Training] Target network updated at step {self._num_training_steps}\n"
            print(msg, flush=True)
            self._training_log_file.write(msg)

        # Log training progress periodically
        if self._num_training_steps % 100 == 0:
            avg_loss = sum(self._training_losses[-100:]) / min(100, len(self._training_losses))
            avg_reward = sum(self._recent_rewards) / len(self._recent_rewards) if self._recent_rewards else 0.0
            epsilon = self._get_epsilon()

            # DEBUG: Log reward statistics
            if self._recent_rewards:
                min_reward = min(self._recent_rewards)
                max_reward = max(self._recent_rewards)
                print(f"[DEBUG] Rewards - count: {len(self._recent_rewards)}, min: {min_reward:.4f}, max: {max_reward:.4f}, avg: {avg_reward:.4f}", flush=True)
                print(f"[DEBUG] First 5 recent_rewards: {self._recent_rewards[:5]}", flush=True)

            msg = (
                f"[Training] Steps: {self._num_training_steps} | "
                f"Requests: {self._num_requests_seen} | "
                f"Loss: {avg_loss:.4f} | "
                f"Reward: {avg_reward:.4f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Buffer: {len(self._replay_buffer)}\n"
            )
            print(msg, flush=True)
            self._training_log_file.write(msg)

    def _save_checkpoint(self):
        """Save model checkpoint during training."""
        original_checkpoint_path = self._config.cluster_config.global_scheduler_config.model_checkpoint_path

        # Add training step number to filename
        base_path = original_checkpoint_path.replace('.pt', '')
        numbered_checkpoint_path = f"{base_path}_req{self._num_requests_seen}.pt"

        # Save RL agent checkpoint with request number
        self._rl_agent.save_checkpoint(numbered_checkpoint_path)

        # Save epsilon state separately (so it persists across restarts)
        epsilon_state = {
            'num_training_steps': self._num_training_steps,
            'epsilon_start': self._epsilon_start,
            'epsilon_end': self._epsilon_end,
            'epsilon_decay_steps': self._epsilon_decay_steps,
        }
        numbered_epsilon_path = numbered_checkpoint_path.replace('.pt', '_epsilon.pt')

        torch.save(epsilon_state, numbered_epsilon_path)

        # Also save latest checkpoint (using original path)
        self._rl_agent.save_checkpoint(original_checkpoint_path)
        
        # Save latest epsilon state (using original path)
        latest_epsilon_path = original_checkpoint_path.replace('.pt', '_epsilon.pt')
        torch.save(epsilon_state, latest_epsilon_path)

    def save_final_checkpoint(self):
        """
        Public method to save final checkpoint when simulation completes.
        Called explicitly from main.py after simulator.run() finishes.
        """
        if self._mode != "training":
            return
        
        checkpoint_path = self._config.cluster_config.global_scheduler_config.model_checkpoint_path
        if not checkpoint_path:
            return
        
        print(f"\n[RL Scheduler] Saving final checkpoint to {checkpoint_path}", flush=True)
        
        # Save the trained model with full trace coverage
        self._save_checkpoint()
        
        print(f"[RL Scheduler] Final checkpoint saved after {self._num_requests_seen} requests", flush=True)
