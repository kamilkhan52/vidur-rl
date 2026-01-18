"""State extraction and tracking for RL-based request routing."""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from vidur.entities import Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.types.replica_id import ReplicaId


class StateTracker:
    """
    Maintains shadow state of replica workload for accurate feature extraction.

    Tracks:
    - Pending prefill/decode tokens per replica (properly decremented on completion)
    - In-flight block hashes for shadow cache (visibility into queued requests)
    - Session-to-replica mapping for session affinity signals
    - Request-to-replica mapping with stored token counts
    """

    def __init__(
        self,
        num_replicas: int,
        replica_schedulers: Dict[ReplicaId, BaseReplicaScheduler],
        enable_prefix_caching: bool = False,
    ):
        """
        Initialize state tracker.

        Args:
            num_replicas: Number of replicas in the system
            replica_schedulers: Map of replica ID to replica scheduler
            enable_prefix_caching: Whether prefix caching is enabled (affects state dimension)
        """
        self._num_replicas = num_replicas
        self._replica_schedulers = replica_schedulers
        self._enable_prefix_caching = enable_prefix_caching

        # Shadow state (updated on assignment and completion)
        self._pending_prefills: Dict[ReplicaId, int] = {
            rid: 0 for rid in replica_schedulers.keys()
        }
        self._pending_decodes: Dict[ReplicaId, int] = {
            rid: 0 for rid in replica_schedulers.keys()
        }
        self._pending_requests: Dict[ReplicaId, int] = {
            rid: 0 for rid in replica_schedulers.keys()
        }

        # Request tracking - stores replica_id, prefills_added, decodes_added, block_hashes
        self._request_to_replica: Dict[int, Dict[str, Any]] = {}

        # In-flight block hashes per replica (for shadow cache)
        self._in_flight_blocks: Dict[ReplicaId, Set[int]] = {
            rid: set() for rid in replica_schedulers.keys()
        }

        # Session-to-replica mapping (for session affinity features)
        self._session_to_replica: Dict[Optional[int], ReplicaId] = {}

        # Compute state dimension (depends on prefix caching)
        # With caching (4 replicas): 72 dims = 5 + 20 + 4 + 36 + 4 + 3
        # Without caching (4 replicas): 52 dims = 5 + 20 + 20 + 4 + 3
        self._state_dim = self._compute_state_dim()

    def _compute_state_dim(self) -> int:
        """
        Compute state vector dimensionality.

        With prefix caching enabled (M replicas, K=5 lookahead):
        - Current request: 5
        - Per-replica features: M * 5
        - Cache hits: M
        - Lookahead: (K-1) * (5 + M)  # K-1 requests with cache features
        - Session affinity: M  # One affinity signal per replica
        - Global: 3
        Total: 5 + M*5 + M + 4*(5+M) + M + 3 = 72 for M=4

        Without prefix caching (M replicas, K=5 lookahead):
        - Current request: 5
        - Per-replica features: M * 5
        - Lookahead: (K-1) * 5  # K-1 requests without cache features
        - Session affinity: M  # One affinity signal per replica
        - Global: 3
        Total: 5 + M*5 + 4*5 + M + 3
        """
        M = self._num_replicas
        K = 5  # Lookahead window

        if self._enable_prefix_caching:
            # Include cache hit features + session affinity
            return 5 + M * 5 + M + (K - 1) * (5 + M) + M + 3
        else:
            # Exclude cache hit features, include session affinity
            return 5 + M * 5 + (K - 1) * 5 + M + 3

    def get_state_dim(self) -> int:
        """Return state vector dimensionality."""
        return self._state_dim

    def get_feature_names(self) -> List[str]:
        """
        Generate human-readable feature names for the state vector.

        Returns:
            List of feature names matching the state vector structure.
        """
        names = []
        M = self._num_replicas
        K = 5  # Lookahead window

        # 1. Current request features (5)
        names.extend([
            "req_prefill_norm",
            "req_decode_norm",
            "req_prefill_log",
            "req_decode_log",
            "req_prefill_decode_ratio",
        ])

        # 2. Per-replica features (M × 5)
        for i in range(M):
            names.extend([
                f"replica{i}_pending_prefill_norm",
                f"replica{i}_pending_decode_norm",
                f"replica{i}_pending_reqs_norm",
                f"replica{i}_pending_prefill_log",
                f"replica{i}_pending_decode_log",
            ])

        # 3. Cache hits for current request (M) - only if prefix caching
        if self._enable_prefix_caching:
            for i in range(M):
                names.append(f"req_cache_hit_replica{i}")

        # 4. Lookahead features ((K-1) × (5 or 5+M))
        for k in range(K - 1):
            names.extend([
                f"lookahead{k}_prefill_norm",
                f"lookahead{k}_decode_norm",
                f"lookahead{k}_prefill_log",
                f"lookahead{k}_decode_log",
                f"lookahead{k}_prefill_decode_ratio",
            ])
            if self._enable_prefix_caching:
                for i in range(M):
                    names.append(f"lookahead{k}_cache_hit_replica{i}")

        # 5. Session affinity features (M)
        for i in range(M):
            names.append(f"session_affinity_replica{i}")

        # 6. Global features (3)
        names.extend([
            "global_total_prefill_norm",
            "global_total_reqs_norm",
            "global_load_imbalance",
        ])

        return names

    def extract_state(
        self, current_request: Request, queue: List[Request]
    ) -> np.ndarray:
        """
        Extract MDP state features for current routing decision.

        State structure depends on prefix caching:

        WITH prefix caching (M replicas, K=5 lookahead): 72 dims for M=4
        1. Current request features [5]
        2. Per-replica features [M × 5]
        3. Per-replica cache hits [M] (with shadow cache from in-flight requests)
        4. Lookahead queue features [(K-1) × (5 + M)]
        5. Session affinity features [M]
        6. Global features [3]

        WITHOUT prefix caching (M replicas, K=5 lookahead): 52 dims for M=4
        1. Current request features [5]
        2. Per-replica features [M × 5]
        3. Lookahead queue features [(K-1) × 5]
        4. Session affinity features [M]
        5. Global features [3]

        Args:
            current_request: Request being routed
            queue: Remaining requests in queue

        Returns:
            State vector (numpy array of shape [state_dim])
        """
        # 1. Extract current request features (5 dims)
        req_features = self._extract_request_features(current_request)

        # 2. Extract per-replica features (M × 5 dims)
        replica_features = []
        for replica_id in sorted(self._replica_schedulers.keys()):
            replica_feat = self._extract_replica_features(replica_id)
            replica_features.extend(replica_feat)

        # 3. Extract cache hits for current request (includes shadow cache)
        if self._enable_prefix_caching:
            cache_hits = self._extract_cache_hits_with_shadow(current_request)

        # 4. Extract lookahead features
        lookahead_features = self._extract_lookahead_features(queue, K=5)

        # 5. Extract session affinity features (M dims)
        session_affinity = self._extract_session_affinity(current_request)

        # 6. Extract global features (3 dims)
        global_features = self._extract_global_features()

        # Concatenate all features
        if self._enable_prefix_caching:
            state = np.concatenate(
                [
                    req_features,
                    np.array(replica_features),
                    np.array(cache_hits),
                    lookahead_features,
                    np.array(session_affinity),
                    global_features,
                ]
            )
        else:
            state = np.concatenate(
                [
                    req_features,
                    np.array(replica_features),
                    lookahead_features,
                    np.array(session_affinity),
                    global_features,
                ]
            )

        return state.astype(np.float32)

    def _extract_request_features(self, request: Request) -> np.ndarray:
        """
        Extract features for a single request.

        Returns 5-dimensional feature vector:
        - Normalized prefill tokens
        - Normalized decode tokens
        - Log-transformed prefill tokens
        - Log-transformed decode tokens
        - Prefill/decode ratio
        """
        return np.array(
            [
                request.num_prefill_tokens / 4096.0,
                request.num_decode_tokens / 512.0,
                np.log1p(request.num_prefill_tokens),
                np.log1p(request.num_decode_tokens),
                request.num_prefill_tokens / max(request.num_decode_tokens, 1),
            ]
        )

    def _extract_replica_features(self, replica_id: ReplicaId) -> np.ndarray:
        """
        Extract features for a single replica.

        Returns 5-dimensional feature vector:
        - Normalized pending prefills
        - Normalized pending decodes
        - Normalized pending requests
        - Log-transformed pending prefills
        - Log-transformed pending decodes
        """
        return np.array(
            [
                self._pending_prefills[replica_id] / 10000.0,
                self._pending_decodes[replica_id] / 10000.0,
                self._pending_requests[replica_id] / 50.0,
                np.log1p(self._pending_prefills[replica_id]),
                np.log1p(self._pending_decodes[replica_id]),
            ]
        )

    def _extract_lookahead_features(
        self, queue: List[Request], K: int = 5
    ) -> np.ndarray:
        """
        Extract features for next K-1 requests in queue.

        For each lookahead request, extracts:
        - Request features (5 dims)
        - Cache hit potential per replica (M dims) - only if prefix caching enabled

        Zero-pads if fewer than K-1 requests available.
        """
        features = []
        num_lookahead = K - 1  # K-1 because current request is separate

        for i in range(num_lookahead):
            if i < len(queue):
                # Request features
                req_feat = self._extract_request_features(queue[i])
                features.extend(req_feat)

                # Cache hit potential per replica (only if prefix caching enabled)
                if self._enable_prefix_caching:
                    for replica_id in sorted(self._replica_schedulers.keys()):
                        try:
                            cached_length = self._replica_schedulers[
                                replica_id
                            ].get_cached_prefill_length(queue[i])
                            cache_hit_ratio = cached_length / max(queue[i].num_prefill_tokens, 1)
                        except (AttributeError, Exception):
                            # Prefix caching not available
                            cache_hit_ratio = 0.0
                        features.append(cache_hit_ratio)
            else:
                # Zero-padding for missing requests
                if self._enable_prefix_caching:
                    # 5 request features + M cache hit ratios
                    padding_size = 5 + self._num_replicas
                else:
                    # 5 request features only
                    padding_size = 5
                features.extend([0.0] * padding_size)

        return np.array(features)

    def _extract_global_features(self) -> np.ndarray:
        """
        Extract global system features.

        Returns 3-dimensional feature vector:
        - Total pending prefills (normalized)
        - Total pending requests (normalized)
        - Load imbalance (max - min / total)
        """
        total_prefills = sum(self._pending_prefills.values())
        total_requests = sum(self._pending_requests.values())

        # Load imbalance
        prefill_loads = list(self._pending_prefills.values())
        if prefill_loads and sum(prefill_loads) > 0:
            load_imbalance = (max(prefill_loads) - min(prefill_loads)) / sum(
                prefill_loads
            )
        else:
            load_imbalance = 0.0

        return np.array(
            [
                total_prefills / 10000.0,
                total_requests / 200.0,
                load_imbalance,
            ]
        )

    def _extract_cache_hits_with_shadow(self, request: Request) -> List[float]:
        """
        Extract cache hit ratios including shadow cache from in-flight requests.

        Shadow cache tracks block hashes from queued requests that haven't
        completed yet, giving the RL agent visibility into potential cache hits
        from requests that are in-flight but not yet in the physical cache.

        Args:
            request: Request to check cache hits for

        Returns:
            List of cache hit ratios per replica
        """
        cache_hits = []
        request_blocks = (
            set(request.block_hash_ids)
            if request.block_hash_ids is not None
            else set()
        )

        for replica_id in sorted(self._replica_schedulers.keys()):
            try:
                # Get actual cached length from replica
                cached_length = self._replica_schedulers[
                    replica_id
                ].get_cached_prefill_length(request)
                actual_cache_ratio = cached_length / max(
                    request.num_prefill_tokens, 1
                )

                # Add shadow cache contribution from in-flight blocks
                if request_blocks and self._enable_prefix_caching:
                    in_flight = self._in_flight_blocks.get(replica_id, set())
                    # Count request blocks that are in-flight but not yet cached
                    # (already cached blocks are counted in actual_cache_ratio)
                    shadow_hits = len(request_blocks & in_flight)
                    shadow_ratio = shadow_hits / max(len(request_blocks), 1)
                    # Combine: actual + shadow (capped at 1.0)
                    cache_hit_ratio = min(1.0, actual_cache_ratio + shadow_ratio)
                else:
                    cache_hit_ratio = actual_cache_ratio

            except (AttributeError, Exception):
                # Prefix caching not available or error in cache lookup
                cache_hit_ratio = 0.0

            cache_hits.append(cache_hit_ratio)

        return cache_hits

    def _extract_session_affinity(self, request: Request) -> List[float]:
        """
        Extract session affinity features.

        One-hot encoding indicating which replica (if any) previously processed
        requests from this session. Helps the RL agent learn to route
        conversation turns to the same replica for cache locality.

        Args:
            request: Request to check session affinity for

        Returns:
            List of affinity signals per replica (1.0 for previous replica, else 0.0)
        """
        session_affinity = []
        session_id = request.session_id

        for replica_id in sorted(self._replica_schedulers.keys()):
            if session_id is not None:
                previous_replica = self._session_to_replica.get(session_id)
                affinity = 1.0 if previous_replica == replica_id else 0.0
            else:
                affinity = 0.0
            session_affinity.append(affinity)

        return session_affinity

    def on_request_assigned(self, request: Request, replica_id: ReplicaId) -> None:
        """
        Update shadow state when request is assigned to a replica.

        Args:
            request: Request being assigned
            replica_id: Target replica ID
        """
        # Calculate uncached prefills
        try:
            cached_length = self._replica_schedulers[
                replica_id
            ].get_cached_prefill_length(request)
        except (AttributeError, Exception):
            # Prefix caching not available
            cached_length = 0

        uncached_prefills = request.num_prefill_tokens - cached_length

        # Update shadow state counters
        self._pending_prefills[replica_id] += uncached_prefills
        self._pending_decodes[replica_id] += request.num_decode_tokens
        self._pending_requests[replica_id] += 1

        # Track in-flight block hashes for shadow cache
        block_hashes: Set[int] = set()
        if request.block_hash_ids is not None:
            block_hashes = set(request.block_hash_ids)
            self._in_flight_blocks[replica_id].update(block_hashes)

        # Store info needed for proper decrement on completion
        self._request_to_replica[request.id] = {
            "replica_id": replica_id,
            "prefills_added": uncached_prefills,
            "decodes_added": request.num_decode_tokens,
            "block_hashes": block_hashes,
        }

        # Update session-to-replica mapping
        if request.session_id is not None:
            self._session_to_replica[request.session_id] = replica_id


    def on_request_completed(self, request: Request) -> None:
        """
        Update shadow state when request completes.

        Args:
            request: Completed request
        """
        if request.id not in self._request_to_replica:
            return

        # Get stored info and remove from tracking
        info = self._request_to_replica.pop(request.id)
        replica_id = info["replica_id"]

        # Decrement all pending counters using stored values
        self._pending_requests[replica_id] = max(
            0, self._pending_requests[replica_id] - 1
        )
        self._pending_prefills[replica_id] = max(
            0, self._pending_prefills[replica_id] - info["prefills_added"]
        )
        self._pending_decodes[replica_id] = max(
            0, self._pending_decodes[replica_id] - info["decodes_added"]
        )

        # Remove block hashes from in-flight set
        block_hashes = info.get("block_hashes", set())
        if block_hashes:
            self._in_flight_blocks[replica_id] -= block_hashes
