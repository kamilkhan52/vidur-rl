"""
Cache-aware RL-based global scheduler for LLM request routing.

This scheduler uses reinforcement learning to route requests across replicas,
learning to balance load, minimize prefill-decode interference, and exploit
KV cache affinity.

Transition semantics (fixed 2026-09-21):
    Each routing decision i observes state s_i and picks replica a_i. The
    reward r_i is only known when request i completes. The successor state
    s_{i+1} is the state observed at the *next routing decision*, which is
    recorded at decision time. A transition (s_i, a_i, r_i, s_{i+1}, done=False)
    enters the replay buffer once both r_i and s_{i+1} are known. Only the very
    last decision of a simulation is terminal (done=True).

    Before this fix the successor state was taken from the global queue at
    completion time, which is empty almost always (schedule() drains it), so
    every transition was marked done=True and the Double-DQN bootstrap never
    contributed: training was effectively a contextual bandit.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def parse_hidden_dims(spec) -> List[int]:
    """Parse a hidden-dims spec ("512,512,256,128" or a list) into a list of ints."""
    if spec is None:
        return [512, 512, 256, 128]
    if isinstance(spec, (list, tuple)):
        dims = [int(d) for d in spec]
    else:
        dims = [int(d) for d in str(spec).replace(" ", "").split(",") if d]
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"Invalid hidden_dims spec: {spec!r}")
    return dims


def expected_state_dim(num_replicas: int, lookahead_k: int, with_cache: bool) -> int:
    """State dimensionality produced by StateTracker for the given settings."""
    M, K = num_replicas, lookahead_k
    if with_cache:
        return 5 + M * 5 + M + (K - 1) * (5 + M) + M + 3
    return 5 + M * 5 + (K - 1) * 5 + M + 3


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
        super().__init__(config, replicas)
        # Its StateTracker features include true output lengths (req_decode_*, lookahead*_decode_*,
        # replica*_pending_decode_*): an oracle no real router has (review V41_r1, 2026-10-10). Refuse unless the
        # caller acknowledges it; results from this scheduler are not deployable-router results.
        if os.environ.get("VIDUR_ALLOW_ORACLE_RL") != "1":
            raise RuntimeError(
                "rl_cache_aware uses true output lengths in its state (oracle features). Set VIDUR_ALLOW_ORACLE_RL=1 "
                "to run it anyway; never report its results as a deployable router. Use the 'learned' scheduler.")

        rl_config = config.cluster_config.global_scheduler_config
        self._rl_config = rl_config
        self._mode = rl_config.mode
        if self._mode not in ("training", "inference"):
            raise ValueError(f"Unknown RL scheduler mode: {self._mode!r}")

        self._inference_epsilon = float(getattr(rl_config, "inference_epsilon", 0.0))
        if self._mode == "inference":
            print(
                f"[RL Scheduler] Mode: INFERENCE - epsilon-greedy with fixed "
                f"epsilon={self._inference_epsilon:.3f} "
                f"({'pure greedy' if self._inference_epsilon == 0.0 else 'exploration active'})"
            )
        else:
            print("[RL Scheduler] Mode: TRAINING - Epsilon-greedy ENABLED (exploration active)")

        # Inspect an existing checkpoint (if any) so the state layout and the
        # network architecture can be matched to it.
        enable_prefix_caching = config.cluster_config.cache_config.enable_prefix_caching
        checkpoint = None
        checkpoint_state_dim = None
        if rl_config.model_checkpoint_path and os.path.exists(rl_config.model_checkpoint_path):
            try:
                checkpoint = torch.load(
                    rl_config.model_checkpoint_path, map_location="cpu", weights_only=False
                )
                checkpoint_state_dim = checkpoint.get("state_dim")
                if checkpoint_state_dim:
                    print(f"[RL Scheduler] Found checkpoint with state_dim={checkpoint_state_dim}")
            except Exception as e:
                print(f"[RL Scheduler] Could not read checkpoint: {e}")

        M = len(replicas)
        K = rl_config.lookahead_k
        if checkpoint_state_dim is not None:
            dim_with_cache = expected_state_dim(M, K, True)
            dim_without_cache = expected_state_dim(M, K, False)
            if checkpoint_state_dim == dim_with_cache:
                if not enable_prefix_caching:
                    print(
                        f"[RL Scheduler] WARNING: checkpoint uses cache-aware state "
                        f"(dim={checkpoint_state_dim}) but prefix caching is DISABLED in config; "
                        f"overriding to cache-aware state to match checkpoint"
                    )
                enable_prefix_caching = True
            elif checkpoint_state_dim == dim_without_cache:
                if enable_prefix_caching:
                    print(
                        f"[RL Scheduler] WARNING: checkpoint uses simplified state "
                        f"(dim={checkpoint_state_dim}) but prefix caching is ENABLED in config; "
                        f"overriding to simplified state to match checkpoint"
                    )
                enable_prefix_caching = False
            else:
                print(
                    f"[RL Scheduler] WARNING: checkpoint state_dim={checkpoint_state_dim} matches "
                    f"neither {dim_with_cache} (with cache) nor {dim_without_cache} (without cache) "
                    f"for M={M}, K={K}; the checkpoint will not load"
                )

        print(
            f"[RL Scheduler] Using {'cache-aware' if enable_prefix_caching else 'simplified'} state"
        )

        # State tracker - maintains shadow state of pending work
        self._state_tracker = StateTracker(
            num_replicas=M,
            replica_schedulers=self._replica_schedulers,
            enable_prefix_caching=enable_prefix_caching,
            lookahead_k=K,
        )

        # Network architecture: from config, unless a checkpoint dictates otherwise.
        hidden_dims = parse_hidden_dims(rl_config.hidden_dims)
        if checkpoint is not None and checkpoint.get("hidden_dims"):
            ckpt_dims = [int(d) for d in checkpoint["hidden_dims"]]
            if ckpt_dims != hidden_dims:
                print(
                    f"[RL Scheduler] WARNING: config hidden_dims={hidden_dims} but checkpoint was "
                    f"trained with {ckpt_dims}; using the checkpoint architecture"
                )
                hidden_dims = ckpt_dims
        self._hidden_dims = hidden_dims
        print(f"[RL Scheduler] Q-network hidden dims: {hidden_dims}")

        self._rl_agent = RLAgent(
            state_dim=self._state_tracker.get_state_dim(),
            num_actions=M,
            config=rl_config,
            hidden_dims=hidden_dims,
        )
        self._gamma = float(getattr(rl_config, "gamma", 0.99))

        # Epsilon schedule state (training only)
        self._num_training_steps = 0
        self._num_requests_seen = 0
        self._epsilon_start = rl_config.epsilon_start
        self._epsilon_end = rl_config.epsilon_end
        self._epsilon_decay_steps = rl_config.epsilon_decay_steps

        if rl_config.model_checkpoint_path:
            if os.path.exists(rl_config.model_checkpoint_path):
                print(f"[RL Scheduler] Loaded trained model from: {rl_config.model_checkpoint_path}")
                if self._mode == "training":
                    if rl_config.reset_epsilon_on_load:
                        print("[RL Scheduler] Reset epsilon state (reset_epsilon_on_load=True)")
                        print(
                            f"[RL Scheduler] Using config values: start={self._epsilon_start}, "
                            f"end={self._epsilon_end}, decay_steps={self._epsilon_decay_steps}; "
                            f"initial epsilon={self._get_epsilon():.4f}"
                        )
                    else:
                        epsilon_path = rl_config.model_checkpoint_path.replace(".pt", "_epsilon.pt")
                        if os.path.exists(epsilon_path):
                            epsilon_state = torch.load(
                                epsilon_path, map_location="cpu", weights_only=False
                            )
                            self._num_training_steps = epsilon_state.get("num_training_steps", 0)
                            self._num_requests_seen = epsilon_state.get(
                                "num_requests_seen", self._num_training_steps
                            )
                            self._epsilon_start = epsilon_state.get("epsilon_start", self._epsilon_start)
                            self._epsilon_end = epsilon_state.get("epsilon_end", self._epsilon_end)
                            self._epsilon_decay_steps = epsilon_state.get(
                                "epsilon_decay_steps", self._epsilon_decay_steps
                            )
                            print(
                                f"[RL Scheduler] Restored epsilon state: requests {self._num_requests_seen}, "
                                f"updates {self._num_training_steps}, epsilon {self._get_epsilon():.4f}"
                            )
                        else:
                            print("[RL Scheduler] No epsilon state found, starting epsilon decay from beginning")
            else:
                print(
                    f"[RL Scheduler] Model checkpoint path provided but file not found: "
                    f"{rl_config.model_checkpoint_path}"
                )
                if self._mode == "inference":
                    print("[RL Scheduler] WARNING: Running inference with randomly initialized model!")

        # Reward calculator
        self._reward_calculator = RewardCalculator(
            normalization_factor=rl_config.reward_normalization_factor,
            reward_type=rl_config.reward_type,
        )
        print(f"[RL Scheduler] Reward type: {rl_config.reward_type}")

        # Load reward plugin if specified (overrides reward_type)
        if rl_config.reward_plugin_path:
            import importlib.util

            spec = importlib.util.spec_from_file_location("rl_formulation", rl_config.reward_plugin_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "compute_reward"):
                from vidur.scheduler.global_scheduler.reward_calculator import set_reward_plugin

                set_reward_plugin(mod.compute_reward)
                print(f"[RL Scheduler] Loaded reward plugin from: {rl_config.reward_plugin_path}")

        # Decision bookkeeping
        # request.id -> {'state', 'action', 'decision_idx', optionally 'next_state'}
        self._pending_transitions: Dict[int, dict] = {}
        # request.id -> completed transition still waiting for its successor state
        self._awaiting_next_state: Dict[int, dict] = {}
        self._last_decision_request_id: Optional[int] = None
        self._decision_idx = 0

        # Training infrastructure
        self._replay_buffer = None
        self._training_log_file = None
        if self._mode == "training":
            transition_log_path = rl_config.transition_log_path or None
            feature_names = self._state_tracker.get_feature_names() if transition_log_path else None

            self._replay_buffer = ReplayBuffer(
                capacity=rl_config.replay_buffer_size,
                log_file_path=transition_log_path,
                feature_names=feature_names,
            )
            self._update_counter = 0
            self._update_frequency = rl_config.update_frequency
            self._target_update_frequency = rl_config.target_network_update_frequency

            self._training_losses = []
            self._training_current_q_means = []
            self._training_target_q_means = []
            self._training_td_abs_means = []
            self._training_td_abs_maxes = []

            self._checkpoint_frequency = 10000  # requests between periodic checkpoints
            self._last_checkpoint_request = 0

            # Training log lives next to the simulation output (never a shared /tmp path:
            # concurrent array tasks on one node used to clobber each other's log).
            log_dir = config.metrics_config.output_dir
            os.makedirs(log_dir, exist_ok=True)
            self._training_log_file = open(os.path.join(log_dir, "rl_training.log"), "w", buffering=1)
            self._training_log_file.write("Training started\n")

        # Metrics tracking
        self._episode_rewards = []
        self._recent_rewards = []  # last 100 rewards for moving average
        self._decision_action_counts = [0 for _ in replicas]
        self._decision_random_count = 0
        self._decision_greedy_count = 0
        self._decision_q_mean_sum = 0.0
        self._decision_q_std_sum = 0.0
        self._decision_q_gap_sum = 0.0
        self._decision_q_gap_max = 0.0
        self._decision_count_since_log = 0
        self._total_random_decisions = 0
        self._total_greedy_decisions = 0

    # ------------------------------------------------------------------ helpers

    @property
    def hidden_dims(self) -> List[int]:
        return list(self._hidden_dims)

    @property
    def total_random_decisions(self) -> int:
        return self._total_random_decisions

    @property
    def total_greedy_decisions(self) -> int:
        return self._total_greedy_decisions

    @property
    def replay_buffer(self) -> Optional[ReplayBuffer]:
        return self._replay_buffer

    def _log(self, msg: str) -> None:
        print(msg, flush=True)
        if self._training_log_file is not None:
            self._training_log_file.write(msg if msg.endswith("\n") else msg + "\n")

    def _reset_decision_diagnostics(self):
        self._decision_action_counts = [0 for _ in self._decision_action_counts]
        self._decision_random_count = 0
        self._decision_greedy_count = 0
        self._decision_q_mean_sum = 0.0
        self._decision_q_std_sum = 0.0
        self._decision_q_gap_sum = 0.0
        self._decision_q_gap_max = 0.0
        self._decision_count_since_log = 0

    def _get_epsilon(self) -> float:
        """
        Current exploration epsilon.

        Training: exponential decay on *requests seen*,
            epsilon = end + (start - end) * exp(-decay_constant * requests),
        with decay_constant chosen so epsilon is within 1% of `end` after
        `epsilon_decay_steps` requests.

        Inference: the fixed `inference_epsilon` from config (default 0 = greedy).
        """
        if self._mode != "training":
            return self._inference_epsilon
        decay_constant = 4.605 / self._epsilon_decay_steps
        return self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(
            -decay_constant * self._num_requests_seen
        )

    # --------------------------------------------------------------- scheduling

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        """
        Route every queued request (FIFO) through the policy.

        For each request: extract state, pick a replica (epsilon-greedy), record
        the decision, link it as the successor state of the previous decision,
        and update the shadow state.
        """
        self.sort_requests()
        request_mapping = []
        replica_ids = list(self._replica_schedulers.keys())

        while self._request_queue:
            request = self._request_queue.pop(0)

            state = self._state_tracker.extract_state(request, self._request_queue)

            epsilon = self._get_epsilon()
            greedy_action_idx, q_diag = self._rl_agent.select_action_with_diagnostics(state)
            if epsilon > 0.0 and self._random_number_generator.random() < epsilon:
                action_idx = self._random_number_generator.randint(0, len(replica_ids) - 1)
                self._decision_random_count += 1
                self._total_random_decisions += 1
            else:
                action_idx = greedy_action_idx
                self._decision_greedy_count += 1
                self._total_greedy_decisions += 1

            self._decision_action_counts[action_idx] += 1
            self._decision_q_mean_sum += q_diag["q_mean"]
            self._decision_q_std_sum += q_diag["q_std"]
            self._decision_q_gap_sum += q_diag["q_gap"]
            self._decision_q_gap_max = max(self._decision_q_gap_max, q_diag["q_gap"])
            self._decision_count_since_log += 1

            replica_id = replica_ids[action_idx]

            # The state of this decision is the successor state of the previous one.
            self._link_successor_state(state)

            self._pending_transitions[request.id] = {
                "state": state,
                "action": action_idx,  # index into replica_ids, not ReplicaId
                "decision_idx": self._decision_idx,
            }
            self._last_decision_request_id = request.id
            self._decision_idx += 1

            self._state_tracker.on_request_assigned(request, replica_id)
            request_mapping.append((replica_id, request))

            if self._mode == "training":
                self._num_requests_seen += 1

        return request_mapping

    def _link_successor_state(self, next_state: np.ndarray) -> None:
        """Attach `next_state` to the previous decision (pending or completed)."""
        prev_id = self._last_decision_request_id
        if prev_id is None:
            return
        if prev_id in self._pending_transitions:
            self._pending_transitions[prev_id]["next_state"] = next_state
        elif prev_id in self._awaiting_next_state:
            t = self._awaiting_next_state.pop(prev_id)
            self._add_transition(t, next_state=next_state, done=False)

    # ---------------------------------------------------------------- learning

    def on_request_end(self, request: Request):
        """Collect the reward for a completed request; store its transition."""
        if request.id not in self._pending_transitions:
            return

        transition = self._pending_transitions.pop(request.id)
        reward = self._reward_calculator.compute_reward(request)
        transition["reward"] = reward
        transition["request_id"] = request.id

        self._episode_rewards.append(reward)
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > 100:
            self._recent_rewards.pop(0)

        # Update shadow state before any later decision state is constructed.
        self._state_tracker.on_request_completed(request)

        if self._mode != "training":
            return

        if "next_state" in transition:
            self._add_transition(transition, next_state=transition["next_state"], done=False)
        else:
            # This was the most recent decision; its successor state is not
            # known until the next routing decision (or simulation end).
            self._awaiting_next_state[request.id] = transition

    def _add_transition(self, t: dict, next_state: np.ndarray, done: bool) -> None:
        if self._replay_buffer is None:
            return
        self._replay_buffer.add(
            state=t["state"],
            action=t["action"],
            reward=t["reward"],
            next_state=next_state,
            done=done,
            meta={"request_id": t.get("request_id", -1), "decision_idx": t.get("decision_idx", -1)},
        )

        if len(self._replay_buffer) == self._rl_config.batch_size:
            self._log(f"[Training] Replay buffer ready with {len(self._replay_buffer)} samples\n")

        self._update_counter += 1
        if (
            self._update_counter % self._update_frequency == 0
            and len(self._replay_buffer) >= self._rl_config.batch_size
        ):
            self._train_step()
            self._log(f"[Training] Completed update at transition {self._update_counter}\n")

        if (
            self._rl_config.model_checkpoint_path
            and self._num_requests_seen - self._last_checkpoint_request >= self._checkpoint_frequency
        ):
            self._save_checkpoint()
            self._last_checkpoint_request = self._num_requests_seen

    def finalize_transitions(self) -> int:
        """
        Flush transitions still waiting for a successor state as terminal.
        Returns the number flushed. Requests that never completed (e.g. stuck
        behind an unschedulable request) are dropped: they carry no reward.
        """
        if self._mode != "training":
            return 0
        flushed = 0
        for req_id in list(self._awaiting_next_state.keys()):
            t = self._awaiting_next_state.pop(req_id)
            self._add_transition(t, next_state=np.zeros_like(t["state"]), done=True)
            flushed += 1
        if self._pending_transitions:
            self._log(
                f"[Training] WARNING: {len(self._pending_transitions)} routed requests never "
                f"completed; their transitions carry no reward and were dropped\n"
            )
        return flushed

    def _train_step(self):
        """One gradient step on a replay batch."""
        batch = self._replay_buffer.sample(self._rl_config.batch_size)
        update_stats = self._rl_agent.update(batch, gamma=self._gamma)
        loss = update_stats["loss"]

        self._training_losses.append(loss)
        self._training_current_q_means.append(update_stats["current_q_mean"])
        self._training_target_q_means.append(update_stats["target_q_mean"])
        self._training_td_abs_means.append(update_stats["td_abs_mean"])
        self._training_td_abs_maxes.append(update_stats["td_abs_max"])
        self._num_training_steps += 1

        if self._rl_agent.get_update_count() % self._target_update_frequency == 0:
            self._rl_agent.update_target_network()
            self._log(f"[Training] Target network updated at step {self._num_training_steps}\n")

        if self._num_training_steps % 100 == 0:
            self._log_training_progress()

    def _log_training_progress(self):
        n = min(100, len(self._training_losses))
        avg_loss = sum(self._training_losses[-100:]) / n
        avg_reward = sum(self._recent_rewards) / len(self._recent_rewards) if self._recent_rewards else 0.0
        avg_current_q = sum(self._training_current_q_means[-100:]) / n
        avg_target_q = sum(self._training_target_q_means[-100:]) / n
        avg_td_abs = sum(self._training_td_abs_means[-100:]) / n
        max_td_abs = max(self._training_td_abs_maxes[-100:]) if self._training_td_abs_maxes else 0.0

        if self._decision_count_since_log > 0:
            c = self._decision_count_since_log
            rand_frac = self._decision_random_count / c
            greedy_frac = self._decision_greedy_count / c
            avg_q_mean = self._decision_q_mean_sum / c
            avg_q_std = self._decision_q_std_sum / c
            avg_q_gap = self._decision_q_gap_sum / c
            action_dist = ",".join(
                f"{i}:{count / c:.2f}" for i, count in enumerate(self._decision_action_counts)
            )
        else:
            rand_frac = greedy_frac = avg_q_mean = avg_q_std = avg_q_gap = 0.0
            action_dist = ""

        if self._recent_rewards:
            self._log(
                f"[DEBUG] Rewards - count: {len(self._recent_rewards)}, min: {min(self._recent_rewards):.4f}, "
                f"max: {max(self._recent_rewards):.4f}, avg: {avg_reward:.4f}"
            )

        self._log(
            f"[Training] Steps: {self._num_training_steps} | "
            f"Requests: {self._num_requests_seen} | "
            f"Loss: {avg_loss:.4f} | "
            f"Reward: {avg_reward:.4f} | "
            f"Epsilon: {self._get_epsilon():.3f} | "
            f"Buffer: {len(self._replay_buffer)} | "
            f"CurrQ: {avg_current_q:.4f} | "
            f"TgtQ: {avg_target_q:.4f} | "
            f"TDAbs: {avg_td_abs:.4f} | "
            f"TDMax: {max_td_abs:.4f} | "
            f"RandFrac: {rand_frac:.3f} | "
            f"GreedyFrac: {greedy_frac:.3f} | "
            f"QMean: {avg_q_mean:.4f} | "
            f"QStd: {avg_q_std:.4f} | "
            f"QGap: {avg_q_gap:.4f} | "
            f"QGapMax: {self._decision_q_gap_max:.4f} | "
            f"ActionDist: {action_dist}\n"
        )
        self._reset_decision_diagnostics()

    # ------------------------------------------------------------- checkpoints

    def _save_checkpoint(self):
        """Save a numbered checkpoint plus the 'latest' copy (and epsilon state)."""
        original_checkpoint_path = self._rl_config.model_checkpoint_path
        base_path = original_checkpoint_path.replace(".pt", "")
        numbered_checkpoint_path = f"{base_path}_req{self._num_requests_seen}.pt"

        self._rl_agent.save_checkpoint(numbered_checkpoint_path)

        epsilon_state = {
            "num_training_steps": self._num_training_steps,
            "num_requests_seen": self._num_requests_seen,
            "epsilon_start": self._epsilon_start,
            "epsilon_end": self._epsilon_end,
            "epsilon_decay_steps": self._epsilon_decay_steps,
        }
        torch.save(epsilon_state, numbered_checkpoint_path.replace(".pt", "_epsilon.pt"))

        self._rl_agent.save_checkpoint(original_checkpoint_path)
        torch.save(epsilon_state, original_checkpoint_path.replace(".pt", "_epsilon.pt"))

    def save_final_checkpoint(self):
        """
        End-of-simulation hook, called from main.py after simulator.run():
        flush outstanding transitions, then save the final checkpoint.
        """
        if self._mode != "training":
            return

        flushed = self.finalize_transitions()
        if flushed:
            self._log(f"[Training] Flushed {flushed} terminal transition(s) at simulation end\n")

        checkpoint_path = self._rl_config.model_checkpoint_path
        if not checkpoint_path:
            return

        print(f"\n[RL Scheduler] Saving final checkpoint to {checkpoint_path}", flush=True)
        self._save_checkpoint()
        print(f"[RL Scheduler] Final checkpoint saved after {self._num_requests_seen} requests", flush=True)
