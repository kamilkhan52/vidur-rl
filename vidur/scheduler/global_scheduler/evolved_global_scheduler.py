"""
Global scheduler whose routing decisions come from a small, swappable policy file.

This is the harness for automated router discovery (evolutionary / agentic search):
the simulator, the metrics and this wrapper stay fixed; only the policy file changes.

Policy contract (a Python module at ``policy_path``) — define ONE of:

    def route(burst, replicas, ctx) -> list[int]
        Called once per scheduling call with every request that arrived at this
        instant (Mooncake conv/tool traces arrive in bursts). Must return one
        replica index per request, in order.

    def choose(request, replicas, ctx) -> int
        Called once per request. The harness updates the replica views after each
        assignment, so later requests of the same burst see earlier ones.

Views are read-only snapshots (frozen dataclasses). Replica indices refer to the
order of ``replicas`` (0 .. N-1). ``ctx`` provides ``rng`` (the scheduler's seeded
``random.Random``; the only allowed randomness), ``now`` (simulated time: the
burst's arrival time in route/choose, the observed event's time in observe),
``num_replicas`` and ``state`` (a dict the policy may use to keep memory across
calls within one simulation).

With ``view_refresh_s`` = R > 0 the replica metrics in the views (waiting, running,
kv_usage) are those of a periodic poller: snapshots on the fixed grid k*R, the last
tick at or before the decision. The other fields stay exact.

Information a real router could not have is deliberately withheld: the output
length of a request, future arrivals, and all simulator objects. Block hashes are
those of the prompt's full blocks only (``prompt_block_hashes``, v4.1): the trace's
list also covers the output, so its length would reveal the output length. This
holds for ``prefix_hashes`` and for the router-side ledgers behind
``inflight_prefix_tokens`` / ``active_prefix_tokens``. Cached-prefix lengths are
computed with a side-effect-free probe of each replica's prefix cache.
"""

import importlib.util
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.entities.replica import Replica
from vidur.kv_cache.utils import BlockHashType
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.scheduler.global_scheduler.prompt_hashes import prompt_block_hashes
from vidur.types.replica_id import ReplicaId


class PolicyError(RuntimeError):
    """The policy returned an invalid decision or raised."""


@dataclass(frozen=True)
class ReplicaView:
    index: int
    outstanding_requests: int  # routed here by this router and not yet completed
    pending_prefill_tokens: int  # uncached prefill tokens routed here, not yet prefilled
    waiting: int  # replica's waiting queue length (before this burst is added)
    running: int  # replica's running set size
    kv_usage: float  # fraction of KV blocks in use, 0..1
    kv_capacity_tokens: int


@dataclass(frozen=True)
class RequestView:
    arrived_at: float
    num_prefill_tokens: int
    session_id: Optional[int]
    cached_prefix_tokens: Tuple[int, ...]  # per replica index
    session_last_replica: int  # replica index this session was last routed to, -1 if new
    position_in_burst: int
    burst_size: int
    # The hashes of the prompt's full blocks (num_prefill_tokens // block_tokens of them; never the output
    # positions) and the block size in tokens, so a router can keep its own route-time prefix index (as SGLang,
    # AIBrix and Mooncake do) instead of seeing only the cache.
    prefix_hashes: Tuple[int, ...] = ()
    block_tokens: int = 0
    # Per replica: leading prefix tokens covered by requests routed there and not yet prefilled
    # (in flight), i.e. what a router that updates its cache view on routing would also count.
    inflight_prefix_tokens: Tuple[int, ...] = ()
    # Per replica: leading prefix tokens covered by requests routed there and not yet finished (active
    # sequences, prefilled or not), i.e. Dynamo's active-sequence overlap.
    active_prefix_tokens: Tuple[int, ...] = ()


@dataclass(frozen=True)
class RequestOutcome:
    """Feedback a real router sees for a request it routed, passed to an optional ``observe(outcome, ctx)``.

    Sent twice per request: event "first_token" when its prefill completes (output length still hidden) and
    event "end" when it finishes. Times are simulated seconds. For online learners (Lodestar-style) and RL.
    """
    event: str
    replica: int
    arrived_at: float
    first_token_at: float
    completed_at: Optional[float]
    num_prefill_tokens: int
    num_cached_tokens: int
    num_decode_tokens: Optional[int]
    session_id: Optional[int]


class PolicyContext:
    def __init__(self, rng, num_replicas: int):
        self.rng = rng
        self.num_replicas = num_replicas
        self.now = 0.0
        self.state: Dict[str, Any] = {}


def load_policy(path: str, require_router: bool = True):
    spec = importlib.util.spec_from_file_location("evolved_policy", path)
    if spec is None or spec.loader is None:
        raise PolicyError(f"cannot load policy from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if require_router and not (hasattr(mod, "route") or hasattr(mod, "choose")):
        raise PolicyError(f"policy {path!r} defines neither route() nor choose()")
    return mod


class EvolvedGlobalScheduler(BaseGlobalScheduler):
    REQUIRES_ROUTER = True  # subclasses that learn the routing (LEARNED) load a design module instead

    def __init__(self, config: SimulationConfig, replicas: Dict[ReplicaId, Replica]):
        super().__init__(config, replicas)
        gs_config = config.cluster_config.global_scheduler_config
        if not gs_config.policy_path:
            raise PolicyError("evolved global scheduler requires --evolved_global_scheduler_config_policy_path")
        self._policy = load_policy(gs_config.policy_path, require_router=self.REQUIRES_ROUTER)
        self._replica_ids: List[ReplicaId] = list(self._replica_schedulers.keys())
        self._index_of = {rid: i for i, rid in enumerate(self._replica_ids)}
        self._outstanding = [0] * len(self._replica_ids)
        self._pending_prefill = [0] * len(self._replica_ids)
        # request.id -> (replica index, uncached prefill tokens counted at assignment)
        self._assigned: Dict[int, Tuple[int, int]] = {}
        self._session_last: Dict[int, int] = {}
        self._prefilled: set = set()
        # per replica: multiset of prompt block hashes of requests routed there, not yet prefilled
        self._inflight: List[Dict[int, int]] = [dict() for _ in self._replica_ids]
        self._inflight_req: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        # per replica: multiset of prompt block hashes of requests routed there, not yet finished
        self._active: List[Dict[int, int]] = [dict() for _ in self._replica_ids]
        self._active_req: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        self._ctx = PolicyContext(self._random_number_generator, len(self._replica_ids))
        # v4.1: polled replica metrics (waiting, running, kv_usage) as a periodic poller sees them. With
        # view_refresh_s = R > 0 the replicas are snapshotted on the fixed grid k*R (independent of when decisions
        # happen) and a decision at t sees the snapshot of tick floor(t/R)*R, i.e. data that is (t mod R) old.
        # R = 0: exact state at the decision (the v3 behaviour). Router-owned counters and the prefix probe are
        # always exact (the router keeps them itself).
        self._view_refresh_s = max(0.0, float(getattr(gs_config, "view_refresh_s", 0.0) or 0.0))
        self.needs_clock = self._view_refresh_s > 0  # Simulator calls on_clock(t) before every event
        self._polled: Optional[List[Tuple[int, int, float]]] = None
        self._polled_tick = -1  # grid index k of the snapshot in _polled
        # Routing calls only (per-decision cost); observe() calls are counted separately.
        self.policy_calls = 0
        self.policy_seconds = 0.0
        self.observe_calls = 0
        self.observe_seconds = 0.0

    # ------------------------------------------------------------ observation

    def _cached_prefix_tokens(self, request: Request) -> Tuple[int, ...]:
        """Side-effect-free prefix-cache probe on every replica (mirrors get_computed_blocks, prompt blocks only)."""
        out = []
        hashes = prompt_block_hashes(request, self._block_tokens())
        for rid in self._replica_ids:
            kv = getattr(self._replica_schedulers[rid], "_kv_cache_manager", None)
            if kv is None or not kv.enable_caching or not hashes:
                out.append(0)
                continue
            max_blocks = request.num_prefill_tokens // kv.block_size
            n = 0
            for h in hashes:
                if n >= max_blocks or kv.block_pool.get_cached_block(BlockHashType(h, tuple())) is None:
                    break
                n += 1
            out.append(n * kv.block_size)
        return tuple(out)

    def _block_tokens(self) -> int:
        for rid in self._replica_ids:
            kv = getattr(self._replica_schedulers[rid], "_kv_cache_manager", None)
            if kv is not None:
                return kv.block_size
        return 0

    def _inflight_prefix_tokens(self, request: Request, block_tokens: int, table=None) -> Tuple[int, ...]:
        hashes = prompt_block_hashes(request, block_tokens)
        out = []
        for counts in (self._inflight if table is None else table):
            n = 0
            for h in hashes:
                if counts.get(h, 0) <= 0:
                    break
                n += 1
            out.append(min(n * block_tokens, request.num_prefill_tokens))
        return tuple(out)

    def _inflight_drop(self, request_id: int, req=None, tables=None) -> None:
        info = (self._inflight_req if req is None else req).pop(request_id, None)
        if info is None:
            return
        counts = (self._inflight if tables is None else tables)[info[0]]
        for h in info[1]:
            c = counts.get(h, 0) - 1
            if c > 0:
                counts[h] = c
            else:
                counts.pop(h, None)

    def _poll(self) -> List[Tuple[int, int, float]]:
        out = []
        for rid in self._replica_ids:
            rs = self._replica_schedulers[rid]
            kv = getattr(rs, "_kv_cache_manager", None)
            out.append((len(rs._waiting_queue), len(getattr(rs, "_running", [])),
                        kv.usage if kv is not None else 0.0))
        return out

    def _tick_index(self, t: float) -> int:
        """Largest k with k*R <= t (guarding against float rounding in t/R)."""
        r = self._view_refresh_s
        k = math.floor(t / r)
        if k * r > t:
            k -= 1
        elif (k + 1) * r <= t:
            k += 1
        return k

    def on_clock(self, t: float) -> None:
        """Called by the Simulator before the event at time ``t`` is handled (only when view_refresh_s > 0).

        The replica state now is the state left by all events before ``t``; nothing changes in between, so it is
        also the state at every grid tick in (previous event time, t]. When ``t`` reaches a tick past the stored
        snapshot, the snapshot is retaken and labelled with the last tick <= t. A tick that falls exactly on an
        event time is polled before that event (and before any routing at that instant).
        """
        k = self._tick_index(t)
        if k > self._polled_tick:
            self._polled, self._polled_tick = self._poll(), k

    def _replica_views(self) -> List[ReplicaView]:
        if self._view_refresh_s > 0:
            if self._polled is None:  # no event seen yet (a policy called outside a simulation run)
                self.on_clock(self._ctx.now)
            polled = self._polled
        else:
            polled = self._poll()
        views = []
        for i, rid in enumerate(self._replica_ids):
            rs = self._replica_schedulers[rid]
            waiting, running, kv_usage = polled[i]
            views.append(
                ReplicaView(
                    index=i,
                    outstanding_requests=self._outstanding[i],
                    pending_prefill_tokens=self._pending_prefill[i],
                    waiting=waiting,
                    running=running,
                    kv_usage=kv_usage,
                    kv_capacity_tokens=getattr(rs, "max_request_total_tokens", 0),
                )
            )
        return views

    # --------------------------------------------------------------- routing

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()
        burst = self._request_queue
        self._request_queue = []
        if not burst:
            return []
        self._ctx.now = burst[0].arrived_at
        cached = [self._cached_prefix_tokens(r) for r in burst]
        block_tokens = self._block_tokens()
        req_views = [
            RequestView(
                arrived_at=r.arrived_at,
                num_prefill_tokens=r.num_prefill_tokens,
                session_id=r.session_id,
                cached_prefix_tokens=cached[k],
                session_last_replica=self._session_last.get(r.session_id, -1)
                if r.session_id is not None
                else -1,
                position_in_burst=k,
                burst_size=len(burst),
                prefix_hashes=prompt_block_hashes(r, block_tokens),
                block_tokens=block_tokens,
                inflight_prefix_tokens=self._inflight_prefix_tokens(r, block_tokens),
                active_prefix_tokens=self._inflight_prefix_tokens(r, block_tokens, self._active),
            )
            for k, r in enumerate(burst)
        ]

        mapping: List[Tuple[ReplicaId, Request]] = []
        if hasattr(self._policy, "route"):
            views = self._replica_views()
            choices = self._call(self._policy.route, req_views, views)
            if not isinstance(choices, (list, tuple)) or len(choices) != len(burst):
                raise PolicyError(
                    f"route() must return {len(burst)} replica indices, got {choices!r}"
                )
            for k, (r, c) in enumerate(zip(burst, choices)):
                mapping.append(self._assign(r, self._check(c), cached[k]))
        else:
            for k, r in enumerate(burst):
                rv = req_views[k]
                if r.session_id is not None:
                    rv = replace(rv, session_last_replica=self._session_last.get(r.session_id, -1))
                c = self._call(self._policy.choose, rv, self._replica_views())
                mapping.append(self._assign(r, self._check(c), cached[k]))
        return mapping

    def _call(self, fn, *args, observe: bool = False):
        t0 = time.perf_counter()
        try:
            result = fn(*args, self._ctx)
        except PolicyError:
            raise
        except Exception as e:  # surface policy bugs loudly, never silently default
            raise PolicyError(f"policy raised {type(e).__name__}: {e}") from e
        finally:
            if observe:
                self.observe_calls += 1
                self.observe_seconds += time.perf_counter() - t0
            else:
                self.policy_calls += 1
                self.policy_seconds += time.perf_counter() - t0
        return result

    def _check(self, choice) -> int:
        if isinstance(choice, bool) or not isinstance(choice, int):
            raise PolicyError(f"replica index must be an int, got {choice!r}")
        if not 0 <= choice < len(self._replica_ids):
            raise PolicyError(f"replica index {choice} out of range 0..{len(self._replica_ids) - 1}")
        return choice

    def _assign(self, request: Request, idx: int, cached: Tuple[int, ...]):
        uncached = request.num_prefill_tokens - cached[idx]
        self._outstanding[idx] += 1
        self._pending_prefill[idx] += uncached
        self._assigned[request.id] = (idx, uncached)
        hashes = prompt_block_hashes(request, self._block_tokens())
        if hashes:
            counts = self._inflight[idx]
            for h in hashes:
                counts[h] = counts.get(h, 0) + 1
            self._inflight_req[request.id] = (idx, hashes)
            counts = self._active[idx]
            for h in hashes:
                counts[h] = counts.get(h, 0) + 1
            self._active_req[request.id] = (idx, hashes)
        if request.session_id is not None:
            self._session_last[request.session_id] = idx
        return self._replica_ids[idx], request

    # ------------------------------------------------------------- feedback

    def _observe(self, request: Request, idx: int, event: str) -> None:
        """During observe, ctx.now is the time of the observed event (the prefill or request end); it is restored
        to the last routing time afterwards."""
        observe = getattr(self._policy, "observe", None)
        if observe is None:
            return
        end = event == "end"
        outcome = RequestOutcome(
            event=event, replica=idx, arrived_at=request.arrived_at,
            first_token_at=request.prefill_completed_at,
            completed_at=request.completed_at if end else None,
            num_prefill_tokens=request.original_num_prefill_tokens,
            num_cached_tokens=request.num_prefill_tokens_cached,
            num_decode_tokens=request.original_num_decode_tokens if end else None,
            session_id=request.session_id)
        saved_now = self._ctx.now
        self._ctx.now = outcome.completed_at if end else outcome.first_token_at
        try:
            self._call(observe, outcome, observe=True)
        finally:
            self._ctx.now = saved_now

    def on_prefill_end(self, request: Request) -> None:
        # Prefill completes again after a preemption restart; count it once.
        # (The built-in LOP_UNCACHED decrements on every prefill end.)
        info = self._assigned.get(request.id)
        if info is not None and request.id not in self._prefilled:
            idx, uncached = info
            self._pending_prefill[idx] -= uncached
            self._prefilled.add(request.id)
            self._inflight_drop(request.id)
            self._observe(request, idx, "first_token")

    def on_request_end(self, request: Request) -> None:
        info = self._assigned.pop(request.id, None)
        if info is not None:
            self._observe(request, info[0], "end")
            self._outstanding[info[0]] -= 1
            if request.id not in self._prefilled:  # defensive: prefill end not seen
                self._pending_prefill[info[0]] -= info[1]
            self._prefilled.discard(request.id)
        self._inflight_drop(request.id)
        self._inflight_drop(request.id, self._active_req, self._active)
