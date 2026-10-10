"""Snapshot and fork a running Vidur simulation (A11 spike, 2026-10-10).

A planner (lookahead, MCTS, expert iteration, a hindsight oracle) needs to stop a simulation at a routing decision,
copy it, try a different replica in the copy and run the copy forward. This module does that from outside
``vidur/``: it drives the event loop itself (``Simulator.run`` cannot pause), and forks with ``copy.deepcopy``
whose memo pre-seeds the heavy objects that never change during a run, so all forks share them:

- the execution-time predictor (a pure lookup table of ~10^5 entries),
- the loaded policy module (policies keep their state in ``ctx.state``, which is copied),
- the trace arrays of the request generator and every request's block-hash list.

Vidur numbers events, requests and batches with class-level counters. Each ``SimHandle`` keeps its own copy of
those counters and installs it while it runs, so a fork numbers its new objects exactly as the unforked run would.
"""

import atexit
import contextlib
import copy
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from vidur.config import SimulationConfig
from vidur.entities.base_entity import BaseEntity
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def _counter_classes() -> List[type]:
    """Every class that owns a class-level ``_id`` counter (BaseEvent, BaseEntity and subclasses that rebind it)."""
    out, todo = [], [BaseEvent, BaseEntity]
    while todo:
        c = todo.pop()
        if "_id" in c.__dict__:
            out.append(c)
        todo.extend(c.__subclasses__())
    return out


def _read_counters() -> Dict[type, int]:
    return {c: c.__dict__["_id"] for c in _counter_classes()}


def _write_counters(state: Dict[type, int]) -> None:
    for c, v in state.items():
        c._id = v


def _fresh_counters() -> None:
    """Counter state of a fresh process: no subclass owns a counter yet; the roots hold their initial values."""
    for c in _counter_classes():
        if c not in (BaseEvent, BaseEntity):
            delattr(c, "_id")
    BaseEvent._id = 0
    BaseEntity._id = -1


def build_simulator(argv: List[str]) -> "SimHandle":
    """Construct a Simulator from vidur.main CLI arguments (as ``evolve.evaluate.sim_command`` builds them), with
    the id counters of a fresh process, so it numbers events, requests and batches as ``vidur.main`` would."""
    saved = _read_counters()
    _fresh_counters()
    try:
        sim = _construct(argv)  # seeds the global RNGs (set_seeds) as vidur.main does
        return SimHandle(sim, _read_counters())
    finally:
        _fresh_counters()
        _write_counters(saved)


def _construct(argv: List[str]) -> Simulator:
    old = sys.argv
    try:
        sys.argv = ["vidur.main"] + list(argv)
        config = SimulationConfig.create_from_cli_args()
    finally:
        sys.argv = old
    set_seeds(config.seed)
    sim = Simulator(config)
    atexit.unregister(sim._write_output)  # outputs are written explicitly with SimHandle.write_output
    return sim


def _precopy_block_pool(pool, memo: dict) -> None:
    """Copy a vLLM-v1 BlockPool without recursion: deepcopy would follow the free list block by block.

    Registers the new pool, every new block, the free queue and the cached-block map in ``memo`` so that every other
    reference to them (request block tables, the KV manager) is redirected to the copies by the main deepcopy.
    Block hashes are immutable tuples and are shared.
    """
    from vidur.kv_cache.kv_cache_block import KVCacheBlock
    from vidur.kv_cache.kv_cache_block_queue import FreeKVCacheBlockQueue

    new_of = object.__new__
    old_blocks = pool.blocks
    new_blocks = []
    for b in old_blocks:
        nb = new_of(KVCacheBlock)
        d = b.__dict__.copy()
        nb.__dict__ = d
        new_blocks.append(nb)
    for nb in new_blocks:  # re-point the free list (block ids index the pool)
        d = nb.__dict__
        p_, n_ = d["prev_free_block"], d["next_free_block"]
        d["prev_free_block"] = None if p_ is None else new_blocks[p_.block_id]
        d["next_free_block"] = None if n_ is None else new_blocks[n_.block_id]
    for b, nb in zip(old_blocks, new_blocks):
        memo[id(b)] = nb
    memo[id(old_blocks)] = new_blocks

    q = pool.free_block_queue
    nq = new_of(FreeKVCacheBlockQueue)
    nq.__dict__ = q.__dict__.copy()
    nq.free_list_head = None if q.free_list_head is None else new_blocks[q.free_list_head.block_id]
    nq.free_list_tail = None if q.free_list_tail is None else new_blocks[q.free_list_tail.block_id]
    memo[id(q)] = nq

    old_map = pool.cached_block_hash_to_block
    new_map = type(old_map)(dict) if hasattr(old_map, "default_factory") else type(old_map)()
    for h, by_id in old_map.items():
        new_map[h] = {i: new_blocks[i] for i in by_id}
        memo[id(by_id)] = new_map[h]
    memo[id(old_map)] = new_map

    np_ = new_of(type(pool))
    np_.__dict__ = pool.__dict__.copy()
    np_.blocks, np_.free_block_queue, np_.cached_block_hash_to_block = new_blocks, nq, new_map
    memo[id(pool)] = np_
    memo[id(pool.__dict__)] = np_.__dict__


class SimHandle:
    """A simulator plus its private id counters, stepped from outside ``vidur``."""

    def __init__(self, sim: Simulator, counters: Optional[Dict[type, int]] = None, rng_state=None):
        self.sim = sim
        self.counters = counters if counters is not None else _read_counters()
        # Global RNGs: only the RL replay buffer draws from np.random; heuristic and evolved runs use per-object RNGs,
        # which deepcopy copies. Kept per handle so interleaved forks of a learned run stay reproducible.
        self.rng_state = rng_state if rng_state is not None else (random.getstate(), np.random.get_state())
        self.events = 0
        self.decisions = 0

    @contextlib.contextmanager
    def active(self):
        saved, saved_rng = _read_counters(), (random.getstate(), np.random.get_state())
        _write_counters(self.counters)
        random.setstate(self.rng_state[0])
        np.random.set_state(self.rng_state[1])
        try:
            yield
        finally:
            self.counters = _read_counters()
            self.rng_state = (random.getstate(), np.random.get_state())
            _write_counters(saved)
            random.setstate(saved_rng[0])
            np.random.set_state(saved_rng[1])

    # ------------------------------------------------------------ stepping

    def _next_is_arrival(self) -> bool:
        s = self.sim
        t = s._request_generator.get_next_request_arrival_time()
        return t is not None and (not s._event_queue or t <= s._event_queue[0]._time)

    def done(self) -> bool:
        s = self.sim
        return s._time_limit_reached or not (
            s._event_queue or s._request_generator.get_next_request_arrival_time() is not None)

    def at_decision(self) -> bool:
        """True when the next event routes a non-empty burst (all same-instant arrivals are already queued)."""
        s = self.sim
        return (not self._next_is_arrival() and bool(s._event_queue)
                and isinstance(s._event_queue[0], GlobalScheduleEvent) and bool(s._scheduler._request_queue))

    def _step(self) -> None:
        """One iteration of ``Simulator.run``'s loop body (minus traces, which our runs disable)."""
        import heapq
        s = self.sim
        if self._next_is_arrival():
            r = s._request_generator.get_next_request()
            s._add_event(RequestArrivalEvent(r.arrived_at, r))
            return
        event = heapq.heappop(s._event_queue)
        if isinstance(event, GlobalScheduleEvent) and s._scheduler._request_queue:
            self.decisions += 1
        s._set_time(event._time)
        s._add_events(event.handle_event(s._scheduler, s._cluster_metric_store))
        self.events += 1

    def run(self, until: Optional[Callable[["SimHandle"], bool]] = None) -> bool:
        """Step until ``until(self)`` holds before an event (returns True) or the run ends (returns False)."""
        with self.active():
            while not self.done():
                if until is not None and until(self):
                    return True
                self._step()
        return False

    def run_to_decision(self, k: int) -> bool:
        """Advance to the point just before the k-th routing call (k counts from 1)."""
        return self.run(lambda h: h.at_decision() and h.decisions == k - 1)

    def next_decision(self) -> bool:
        n = self.decisions
        return self.run(lambda h: h.at_decision() and h.decisions == n)

    # ------------------------------------------------------------ forking

    def _shared_memo(self) -> dict:
        s = self.sim
        memo = {}

        def share(o):
            if o is not None:
                memo[id(o)] = o

        gs = s._scheduler
        share(getattr(gs, "_execution_time_predictor", None))
        for rs in gs._replica_schedulers.values():
            share(getattr(rs, "_execution_time_predictor", None))
        share(getattr(gs, "_policy", None))  # a module: deepcopy cannot copy it, and it is stateless by contract
        g = s._request_generator
        for name in ("_arrived_at", "_num_prefill_tokens", "_num_decode_tokens", "_block_hash_ids", "_block_size",
                     "_session_id"):
            share(getattr(g, name, None))
        for h in getattr(g, "_block_hash_ids", None) or ():
            share(h)
        for rs in gs._replica_schedulers.values():
            kv = getattr(rs, "_kv_cache_manager", None)
            if kv is not None:
                _precopy_block_pool(kv.block_pool, memo)
        return memo

    def fork(self, output_dir: Optional[str] = None) -> "SimHandle":
        memo = self._shared_memo()
        sim2 = copy.deepcopy(self.sim, memo)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            sim2._config.metrics_config.output_dir = output_dir
            sim2._cluster_metric_store._config.output_dir = output_dir
            for st in sim2._cluster_metric_store._replica_metric_stores.values():
                st._config.output_dir = output_dir
        h = SimHandle(sim2, dict(self.counters), copy.deepcopy(self.rng_state))
        h.events, h.decisions = self.events, self.decisions
        return h

    # ------------------------------------------------------------ outputs

    def write_output(self) -> None:
        with self.active():
            self.sim._write_output()

    @property
    def now(self) -> float:
        return self.sim._time


def override_next_decision(h: SimHandle, choice_for_first: int) -> None:
    """Make the next routing call send the burst's first request to replica ``choice_for_first``.

    Patches the evolved scheduler's ``_assign`` for one assignment, so its bookkeeping (outstanding counts,
    pending prefill, in-flight prefixes, session map) follows the forced decision. The policy is still called,
    so its rng draws and ``ctx.state`` updates are those of the unforced decision.
    """
    gs = h.sim._scheduler
    orig = gs._assign

    def once(request, idx, cached):
        del gs._assign  # back to the class method for every later assignment
        return orig(request, choice_for_first, cached)

    gs._assign = once


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
