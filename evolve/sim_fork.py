"""Snapshot and fork a running Vidur simulation (A11 spike, 2026-10-10; review E r1 fixes the same day).

A planner (lookahead, MCTS, expert iteration, a hindsight oracle) needs to stop a simulation at a routing decision,
copy it, try a different replica in the copy and run the copy forward. This module does that from outside
``vidur/``: it drives the event loop itself (``Simulator.run`` cannot pause), and forks with ``copy.deepcopy``
whose memo pre-seeds the heavy objects that never change during a run, so all forks share them:

- the execution-time predictor (a lookup table of ~10^5 entries; its only mutable field, the ``clamp_hits``
  provenance counter, is kept per handle like the id counters below),
- the loaded policy module (evolved policies keep their state in ``ctx.state``, which is copied; a module with
  mutable globals is refused by ``fork``),
- the trace arrays of the request generator and every request's block-hash list.

Process-global state is kept per ``SimHandle`` and installed while the handle runs: the class-level id counters
of every ``BaseEvent``/``BaseEntity`` subclass (including classes such as ``Batch`` that only get a counter on
first use, recorded as absent), the global ``random``/``np.random``/torch RNG states and the predictor's
``clamp_hits``. A fork therefore numbers its new objects, draws its random numbers and counts its clamps exactly
as the unforked run would. ``fork`` may be called from inside a ``run(until=...)`` callback; it then copies the
live values.

Learned (RL) schedulers are not forkable (``fork`` raises ``NotImplementedError``): see ``_unforkable_reason``.

Forced decisions: see ``override_next_decision``.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # outputs plot through matplotlib; never open a GUI backend

import atexit  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

from vidur.config import SimulationConfig  # noqa: E402
from vidur.entities.base_entity import BaseEntity  # noqa: E402
from vidur.events import BaseEvent, RequestArrivalEvent  # noqa: E402
from vidur.events.global_schedule_event import GlobalScheduleEvent  # noqa: E402
from vidur.scheduler.global_scheduler.evolved_global_scheduler import EvolvedGlobalScheduler  # noqa: E402
from vidur.scheduler.global_scheduler.learned_global_scheduler import LearnedGlobalScheduler  # noqa: E402
from vidur.simulator import Simulator  # noqa: E402
from vidur.utils.random import set_seeds  # noqa: E402

_ROOTS = (BaseEvent, BaseEntity)
_ROOT_INIT = {BaseEvent: 0, BaseEntity: -1}


class _Missing:
    """Marks a class that does not own an ``_id`` counter yet (it inherits its root's until first use)."""

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


def _counter_classes() -> List[type]:
    """Every subclass of BaseEvent and BaseEntity (each may own a class-level ``_id`` counter)."""
    out, todo, seen = [], list(_ROOTS), set()
    while todo:
        c = todo.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        todo.extend(c.__subclasses__())
    return out


def _read_counters() -> Dict[type, object]:
    return {c: c.__dict__.get("_id", MISSING) for c in _counter_classes()}


def _write_counters(state: Dict[type, object]) -> None:
    """Install ``state``. A class absent from ``state`` (imported later) or marked MISSING loses its counter."""
    for c in _counter_classes():
        v = state.get(c, MISSING)
        if v is MISSING:
            if c in _ROOTS:
                c._id = _ROOT_INIT[c]
            elif "_id" in c.__dict__:
                delattr(c, "_id")
        else:
            c._id = v


def _fresh_counters() -> None:
    """Counter state of a fresh process: no subclass owns a counter yet; the roots hold their initial values."""
    _write_counters({})


def _global_rng():
    torch = sys.modules.get("torch")
    return (random.getstate(), np.random.get_state(), torch.get_rng_state() if torch is not None else None)


def _set_global_rng(state) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch = sys.modules.get("torch")
    if torch is not None and state[2] is not None:
        torch.set_rng_state(state[2])


def _copy_rng(state):
    return (state[0], copy.deepcopy(state[1]), None if state[2] is None else state[2].clone())


def build_simulator(argv: List[str]) -> "SimHandle":
    """Construct a Simulator from vidur.main CLI arguments (as ``evolve.evaluate.sim_command`` builds them), with
    the id counters of a fresh process, so it numbers events, requests and batches as ``vidur.main`` would."""
    saved, saved_rng = _read_counters(), _global_rng()
    _fresh_counters()
    try:
        sim = _construct(argv)  # seeds the global RNGs (set_seeds) as vidur.main does
        return SimHandle(sim, _read_counters(), _global_rng())
    finally:
        _write_counters(saved)
        _set_global_rng(saved_rng)


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


_MUTABLE = (dict, list, set, deque, bytearray)


def _unforkable_reason(sim: Simulator) -> Optional[str]:
    gs = sim._scheduler
    if isinstance(gs, LearnedGlobalScheduler):
        # Making these exact needs, per handle: the torch RNG (handled), a private copy of the design module's
        # mutable globals (rl_loto_*: _PEND/_ORDER/_BASE feed the rewards), the network, optimiser and replay
        # buffer (shared or copied consistently), and the learner recording the forced action in its transition.
        # The last needs a hook in learned_global_scheduler.py (hashed).
        return ("learned (RL) schedulers are not forkable: the design module keeps mutable module-level state "
                "shared by all forks, and under a forced decision the learner would record its own action")
    pol = getattr(gs, "_policy", None)
    if isinstance(pol, types.ModuleType):
        bad = sorted(k for k, v in vars(pol).items() if not k.startswith("__") and isinstance(v, _MUTABLE))
        if bad:
            return (f"policy module {getattr(pol, '__file__', pol)!r} has mutable module globals {bad}; the module "
                    "is shared by all forks, so they would overwrite each other's state (keep state in ctx.state)")
    return None


_LIVE: List["SimHandle"] = []  # handles whose state is installed, innermost last


class SimHandle:
    """A simulator plus its private process-global state (id counters, global RNGs, predictor clamp counts)."""

    def __init__(self, sim: Simulator, counters: Optional[Dict[type, object]] = None, rng_state=None,
                 clamp_hits: Optional[dict] = None, owns_output_dir: bool = True):
        self.sim = sim
        self.counters = counters if counters is not None else _read_counters()
        self.rng_state = rng_state if rng_state is not None else _global_rng()
        p = self._predictor()
        self.clamp_hits = clamp_hits if clamp_hits is not None else dict(getattr(p, "clamp_hits", None) or {})
        self.owns_output_dir = owns_output_dir
        self.events = 0
        self.decisions = 0

    def _predictor(self):
        p = getattr(self.sim._scheduler, "_execution_time_predictor", None)
        return p if p is not None and hasattr(p, "clamp_hits") else None

    def _is_live(self) -> bool:
        if _LIVE and _LIVE[-1] is self:
            return True
        if any(h is self for h in _LIVE):
            raise RuntimeError("this handle is suspended under another active handle (a different handle is "
                               "running inside its callback); use it only after that run returns")
        return False

    @contextlib.contextmanager
    def active(self):
        """Install this handle's process-global state for the duration (re-entrant for the same handle)."""
        if self._is_live():
            yield
            return
        p = self._predictor()
        saved, saved_rng = _read_counters(), _global_rng()
        saved_ch = p.clamp_hits if p is not None else None
        _write_counters(self.counters)
        _set_global_rng(self.rng_state)
        if p is not None:
            p.clamp_hits = self.clamp_hits
        _LIVE.append(self)
        try:
            yield
        finally:
            _LIVE.pop()
            self.counters = _read_counters()
            self.rng_state = _global_rng()
            _write_counters(saved)
            _set_global_rng(saved_rng)
            if p is not None:
                self.clamp_hits = p.clamp_hits
                p.clamp_hits = saved_ch

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
        """Step until ``until(self)`` holds before an event (returns True) or the run ends (returns False).

        At the end, requests left in the scheduler are recorded in ``sim.stuck_requests`` as ``Simulator.run``
        does (``write_output`` then writes ``stuck_requests.json``)."""
        with self.active():
            while not self.done():
                if until is not None and until(self):
                    return True
                self._step()
            s = self.sim
            if not (s._scheduler.is_empty() or s._time_limit_reached):
                s.stuck_requests = s._collect_stuck_requests()
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
        share(getattr(gs, "_policy", None))  # a module without mutable globals (checked by _unforkable_reason)
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
        """An independent copy of this simulation at its current point (callable inside a ``run`` callback).

        A pending ``override_next_decision`` is part of the state and is copied (bound to the copy's scheduler).
        Without ``output_dir`` the fork cannot ``write_output`` (it would overwrite the parent's files)."""
        reason = _unforkable_reason(self.sim)
        if reason is not None:
            raise NotImplementedError(reason)
        if self._is_live():  # inside a run() callback: the cached fields are those from when run() started
            counters, rng = _read_counters(), _global_rng()
        else:
            counters, rng = self.counters, self.rng_state
        memo = self._shared_memo()
        sim2 = copy.deepcopy(self.sim, memo)
        sim2._output_written = False
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            sim2._config.metrics_config.output_dir = output_dir
            sim2._cluster_metric_store._config.output_dir = output_dir
            for st in sim2._cluster_metric_store._replica_metric_stores.values():
                st._config.output_dir = output_dir
        h = SimHandle(sim2, dict(counters), _copy_rng(rng), dict(self.clamp_hits),
                      owns_output_dir=output_dir is not None)
        h.events, h.decisions = self.events, self.decisions
        return h

    # ------------------------------------------------------------ outputs

    def write_output(self) -> None:
        """``Simulator.write_output``: metrics, ``predictor_info.json`` (this handle's clamp counts) and
        ``stuck_requests.json`` when requests were left in the scheduler. Once per handle."""
        if not self.owns_output_dir:
            raise RuntimeError("this fork was made without output_dir and would overwrite its parent's outputs; "
                               "pass output_dir to fork()")
        with self.active():
            self.sim.write_output()

    @property
    def now(self) -> float:
        return self.sim._time


# ---------------------------------------------------------------- forced decisions

_FORCE_ATTR = "_sim_fork_force"
_FIFO_KEYS = {"attn_queues", "attn_sums"}


def override_next_decision(h: SimHandle, choice_for_first: int, policy_state: str = "auto") -> None:
    """Make the next routing call send the burst's first request to replica ``choice_for_first``.

    Semantics: a forced decision means the router executed action ``a = choice_for_first``. The policy is still
    called (its scoring and its ``ctx.rng`` draws happen as in the unforced call); its return value is replaced by
    ``a``, the scheduler's ledger (outstanding counts, pending prefill, in-flight/active prefixes, session map)
    books ``a``, and any state the policy keeps about its own decision is made to reflect ``a``:

    - forced == the policy's own choice, or ``ctx.state`` unchanged by the call: nothing to reconcile (exact);
    - the policy module defines ``on_forced(request, replicas, chosen, forced, ctx)``: called after ``choose``
      returned ``chosen``; it must leave ``ctx.state`` as if ``choose`` had returned ``forced``;
    - ``ctx.state`` holds only the attended-context FIFO (``attn_queues``/``attn_sums``; tail4_ni*, l3_* finalists,
      which trim every queue from the views and then append the prompt length at the pick): the change made by the
      call is verified to be exactly "trim, then append n at the pick", and the entry is moved to ``a`` (exact);
    - ``policy_state="independent"``: the caller asserts that the policy's state update does not depend on its
      pick (e.g. claude_r1 ``sessions``, completion_r2 ``session_prompt``/``scarcity_ewma``, t4_70b
      ``prompt_ewma`` by inspection), so the state is left as the call made it;
    - otherwise ``NotImplementedError`` (the scheduler is then mid-decision and the handle must be discarded).

    For ``route()`` policies only the first request is overridden; the rest of the burst keeps the policy's choices,
    which were made assuming its own first choice. The pending override is bound to ``h``'s scheduler and is
    copied by ``fork`` (each copy applies it to itself once).
    """
    gs = h.sim._scheduler
    if isinstance(gs, LearnedGlobalScheduler) or not isinstance(gs, EvolvedGlobalScheduler):
        raise NotImplementedError(f"forced decisions need an evolved (non-learned) global scheduler, got "
                                  f"{type(gs).__name__}")
    if _FORCE_ATTR in gs.__dict__:
        raise RuntimeError("an override is already pending on this handle")
    if policy_state not in ("auto", "independent"):
        raise ValueError(f"policy_state must be 'auto' or 'independent', got {policy_state!r}")
    gs._check(choice_for_first)  # PolicyError unless an int in 0..N-1 (a negative index would wrap)
    gs.__dict__[_FORCE_ATTR] = (choice_for_first, policy_state)
    gs._call = types.MethodType(_forcing_call, gs)  # deepcopy rebinds a bound method to the copied scheduler


def _forcing_call(self, fn, *args):
    pol = self._policy
    is_route = fn is getattr(pol, "route", None)
    if not is_route and fn is not getattr(pol, "choose", None):  # observe(): pass through
        return type(self)._call(self, fn, *args)
    forced, mode = self.__dict__.pop(_FORCE_ATTR)
    del self.__dict__["_call"]
    before = copy.deepcopy(self._ctx.state)
    out = type(self)._call(self, fn, *args)
    if is_route:
        if not isinstance(out, (list, tuple)) or not out:
            return out  # schedule() raises its PolicyError
        chosen = self._check(out[0])
    else:
        chosen = self._check(out)
    if chosen != forced and mode == "auto":
        request = args[0][0] if is_route else args[0]
        _reconcile(pol, before, request, args[1], chosen, forced, self._ctx)
    if is_route:
        return [forced] + list(out[1:])
    return forced


def _reconcile(pol, before: dict, request, replicas, chosen: int, forced: int, ctx) -> None:
    after = ctx.state
    hook = getattr(pol, "on_forced", None)
    if hook is not None:
        hook(request, replicas, chosen, forced, ctx)
        return
    if after == before:
        return
    if set(after) == _FIFO_KEYS and set(before) <= _FIFO_KEYS and _fifo_move(before, after, request, chosen, forced):
        return
    raise NotImplementedError(
        f"cannot force replica {forced} (policy chose {chosen}): ctx.state changed in a way sim_fork cannot "
        f"reconcile (keys {sorted(after)}). Define on_forced() in the policy, or pass policy_state='independent' "
        "if its state update does not depend on its pick. The handle is mid-decision; discard it.")


def _fifo_move(before: dict, after: dict, request, chosen: int, forced: int) -> bool:
    """Verify that the call trimmed each queue (a suffix remains) and appended n at ``chosen``; move it to
    ``forced``. Returns False (state untouched) when the change has any other shape."""
    q1, s1 = after["attn_queues"], after["attn_sums"]
    q0 = before.get("attn_queues")
    num = len(q1)
    if q0 is not None and len(q0) != num:
        return False
    if len(s1) != num or not (0 <= forced < num):
        return False
    n = request.num_prefill_tokens
    for i in range(num):
        cur = list(q1[i])
        if i == chosen:
            if not cur or cur[-1] != n:
                return False
            cur = cur[:-1]
        old = list(q0[i]) if q0 is not None else []
        if len(cur) > len(old) or old[len(old) - len(cur):] != cur or s1[i] != sum(q1[i]):
            return False
    q1[chosen].pop()
    s1[chosen] -= n
    q1[forced].append(n)
    s1[forced] += n
    return True


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
