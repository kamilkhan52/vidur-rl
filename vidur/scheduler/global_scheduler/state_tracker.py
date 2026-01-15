"""State extraction and tracking for RL-based request routing."""

from typing import Dict, List

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
    - Pending prefill tokens per replica
    - Pending decode tokens per replica
    - Pending request count per replica
    - Request-to-replica mapping

    TODO (Phase 2): Implement full state extraction with cache awareness.
    For Phase 1, this is a stub that returns dummy state vectors.
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
        self._pending_prefills = {rid: 0 for rid in replica_schedulers.keys()}
        self._pending_decodes = {rid: 0 for rid in replica_schedulers.keys()}
        self._pending_requests = {rid: 0 for rid in replica_schedulers.keys()}

        # Request tracking
        self._request_to_replica = {}


        # Compute state dimension (depends on prefix caching)
        # With caching (4 replicas): 68 dims = 5 + 20 + 4 + 36 + 3
        # Without caching (4 replicas): 48 dims = 5 + 20 + 20 + 3
        self._state_dim = self._compute_state_dim()

    def _compute_state_dim(self) -> int:
        """
        Compute state vector dimensionality.

        With prefix caching enabled (M replicas, K=5 lookahead):
        - Current request: 5
        - Per-replica features: M * 5
        - Cache hits: M
        - Lookahead: (K-1) * (5 + M)  # K-1 requests with cache features
        - Global: 3
        Total: 5 + M*5 + M + 4*(5+M) + 3

        Without prefix caching (M replicas, K=5 lookahead):
        - Current request: 5
        - Per-replica features: M * 5
        - Lookahead: (K-1) * 5  # K-1 requests without cache features
        - Global: 3
        Total: 5 + M*5 + 4*5 + 3
        """
        M = self._num_replicas
        K = 5  # Lookahead window
        
        if self._enable_prefix_caching:
            # Include cache hit features
            return 5 + M * 5 + M + (K - 1) * (5 + M) + 3
        else:
            # Exclude cache hit features
            return 5 + M * 5 + (K - 1) * 5 + 3

    def get_state_dim(self) -> int:
        """Return state vector dimensionality."""
        return self._state_dim

    def extract_state(
        self, current_request: Request, queue: List[Request]
    ) -> np.ndarray:
        """
        Extract MDP state features for current routing decision.

        State structure depends on prefix caching:

        WITH prefix caching (M replicas, K=5 lookahead): 68 dims for M=4
        1. Current request features [5]
        2. Per-replica features [M × 5]
        3. Per-replica cache hits [M]
        4. Lookahead queue features [(K-1) × (5 + M)]
        5. Global features [3]

        WITHOUT prefix caching (M replicas, K=5 lookahead): 48 dims for M=4
        1. Current request features [5]
        2. Per-replica features [M × 5]
        3. Lookahead queue features [(K-1) × 5]
        4. Global features [3]

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

        # 3. Extract cache hits for current request on each replica (M dims)
        # Only include if prefix caching is enabled
        if self._enable_prefix_caching:
            cache_hits = []
            for replica_id in sorted(self._replica_schedulers.keys()):
                try:
                    cached_length = self._replica_schedulers[
                        replica_id
                    ].get_cached_prefill_length(current_request)
                    cache_hit_ratio = cached_length / max(
                        current_request.num_prefill_tokens, 1
                    )
                except (AttributeError, Exception):
                    # Prefix caching not available or error in cache lookup
                    cache_hit_ratio = 0.0
                cache_hits.append(cache_hit_ratio)

        # 4. Extract lookahead features
        lookahead_features = self._extract_lookahead_features(queue, K=5)

        # 5. Extract global features (3 dims)
        global_features = self._extract_global_features()

        # Concatenate all features
        if self._enable_prefix_caching:
            state = np.concatenate(
                [
                    req_features,
                    np.array(replica_features),
                    np.array(cache_hits),
                    lookahead_features,
                    global_features,
                ]
            )
        else:
            state = np.concatenate(
                [
                    req_features,
                    np.array(replica_features),
                    lookahead_features,
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

    def on_request_assigned(self, request: Request, replica_id: ReplicaId) -> None:
        """
        Update shadow state when request is assigned to a replica.

        Args:
            request: Request being assigned
            replica_id: Target replica ID
        """
        # Update shadow state
        try:
            cached_length = self._replica_schedulers[replica_id].get_cached_prefill_length(
                request
            )
        except (AttributeError, Exception):
            # Prefix caching not available
            cached_length = 0

        uncached_prefills = request.num_prefill_tokens - cached_length

        self._pending_prefills[replica_id] += uncached_prefills
        self._pending_decodes[replica_id] += request.num_decode_tokens
        self._pending_requests[replica_id] += 1
        self._request_to_replica[request.id] = replica_id


    def on_request_completed(self, request: Request) -> None:
        """
        Update shadow state when request completes.

        Args:
            request: Completed request
        """
        if request.id not in self._request_to_replica:
            return

        replica_id = self._request_to_replica.pop(request.id)



        # Decrement pending count
        self._pending_requests[replica_id] = max(
            0, self._pending_requests[replica_id] - 1
        )
