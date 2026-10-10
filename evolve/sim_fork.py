"""Snapshot and fork a running Vidur simulation (A11 spike, 2026-10-10; reviews E r1/r2 fixes).

A planner (lookahead, MCTS, expert iteration, a hindsight oracle) needs to stop a simulation at a routing decision,
copy it, try a different replica in the copy and run the copy forward. This module does that from outside
``vidur/``: it drives the event loop itself (``Simulator.run`` cannot pause), and forks with ``copy.deepcopy``
whose memo pre-seeds the heavy objects that never change during a run, so all forks share them:

- the execution-time predictor (a lookup table of ~10^5 entries; its only mutable field, the ``clamp_hits``
  provenance counter, is kept per handle like the id counters below),
- the loaded policy module (evolved policies keep their state in ``ctx.state``, which is copied; ``fork``
  refuses a module that may keep state elsewhere: a static scan of its source plus a runtime fingerprint),
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

import ast  # noqa: E402
import atexit  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import pickle  # noqa: E402
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


# ---------------------------------------------------------------- policy-module state
#
# The policy module is shared by every fork (deepcopy cannot copy a module), so any state it keeps outside
# ``ctx.state`` would be shared too. Two conservative checks, both of which may refuse a harmless module:
# - a static scan of the module's source (``_module_state_issues``): no ``global``/``nonlocal``; module-level and
#   class-level assignments only of constant expressions (literal numbers/strings/tuples, arithmetic on them,
#   names/attributes, float()/int()/math.* calls; dict/list/set literals and other calls are refused); no
#   decorators other than staticmethod/classmethod/property/dataclass (this refuses lru_cache and other caches);
#   constant default arguments only; no stores into an attribute or item of a module-level name (function
#   attributes, class attributes, other modules); no setattr/delattr/globals/vars/exec/eval;
# - a runtime fingerprint of the module's non-code state (``_module_fingerprint``: globals, functions' defaults,
#   closure cells, attributes and cache_info, module-defined classes' data attributes, ndarray bytes, RNG states)
#   taken when the handle is built and compared at every ``fork``.
# ``fork(trust_module_state=True)`` skips both, for callers that know the module is stateless or that isolate each
# rollout in its own process (``os.fork`` makes module state per process).

_SAFE_DECORATORS = {"staticmethod", "classmethod", "property", "dataclass", "dataclasses.dataclass"}
_CONST_CALLS = {"float", "int", "str", "bool", "complex", "frozenset", "tuple", "bytes", "range", "abs", "min", "max"}
_UNSAFE_CALLS = {"setattr", "delattr", "globals", "vars", "exec", "eval", "__import__", "locals"}
_AST_CACHE: Dict[tuple, List[str]] = {}


def _const_expr(e) -> bool:
    if e is None or isinstance(e, ast.Constant):
        return True
    if isinstance(e, ast.Tuple):
        return all(_const_expr(x) for x in e.elts)
    if isinstance(e, ast.UnaryOp):
        return _const_expr(e.operand)
    if isinstance(e, ast.BinOp):
        return _const_expr(e.left) and _const_expr(e.right)
    if isinstance(e, ast.BoolOp):
        return all(_const_expr(x) for x in e.values)
    if isinstance(e, ast.Compare):
        return _const_expr(e.left) and all(_const_expr(x) for x in e.comparators)
    if isinstance(e, ast.IfExp):
        return _const_expr(e.test) and _const_expr(e.body) and _const_expr(e.orelse)
    if isinstance(e, (ast.Name, ast.Attribute)):
        return isinstance(e.ctx, ast.Load)
    if isinstance(e, ast.Call):
        f = e.func
        ok = (isinstance(f, ast.Name) and f.id in _CONST_CALLS) or (
            isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "math")
        return ok and all(_const_expr(a) for a in e.args) and all(_const_expr(k.value) for k in e.keywords)
    return False


def _root_name(e) -> Optional[str]:
    while isinstance(e, (ast.Attribute, ast.Subscript, ast.Starred)):
        e = e.value
    return e.id if isinstance(e, ast.Name) else None


def _decorator_issue(d, where: str) -> Optional[str]:
    name = ast.unparse(d.func if isinstance(d, ast.Call) else d)
    return None if name in _SAFE_DECORATORS else f"{where}: decorator @{name}"


def _function_issues(fn, module_names: set) -> List[str]:
    out = []
    loc = fn.name if hasattr(fn, "name") else "lambda"
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            if not all(_const_expr(d) for d in list(a.defaults) + list(a.kw_defaults)):
                out.append(f"{loc}: mutable or computed default argument")
            for d in getattr(node, "decorator_list", ()):
                issue = _decorator_issue(d, loc)
                if issue and node is not fn:
                    out.append(issue)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.append(f"{loc}: '{type(node).__name__.lower()} {', '.join(node.names)}'")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _UNSAFE_CALLS:
            out.append(f"{loc}: call to {node.func.id}()")
    params = {x.arg for x in ast.walk(fn.args) if isinstance(x, ast.arg)}
    local = params | {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store,
                                                                                                    ast.Del))}
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for t in targets:
            for x in ast.walk(t):
                if isinstance(x, (ast.Attribute, ast.Subscript)) and isinstance(x.ctx, (ast.Store, ast.Del)):
                    r = _root_name(x)
                    if r in module_names and r not in local:
                        out.append(f"{loc}: stores into module-level '{r}' ({ast.unparse(x)})")
    return out


def _body_issues(body, module_names: set, where: str) -> List[str]:
    out = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list:
                issue = _decorator_issue(d, f"{where}{node.name}")
                if issue:
                    out.append(issue)
            out += _function_issues(node, module_names)
        elif isinstance(node, ast.ClassDef):
            for d in node.decorator_list:
                issue = _decorator_issue(d, f"{where}{node.name}")
                if issue:
                    out.append(issue)
            out += _body_issues(node.body, module_names, f"{where}{node.name}.")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if not _const_expr(node.value):
                out.append(f"{where}{ast.unparse(node)[:60]!r}: non-constant value")
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not all(isinstance(t, ast.Name) or (isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name)
                                                                                      for x in t.elts)) for t in tg):
                out.append(f"{where}{ast.unparse(node)[:60]!r}: store into an attribute or item")
        elif isinstance(node, ast.If):
            out += _body_issues(node.body + node.orelse, module_names, where)
        elif isinstance(node, ast.Try):
            out += _body_issues(node.body + node.orelse + node.finalbody
                                + [s for h in node.handlers for s in h.body], module_names, where)
        else:
            out.append(f"{where}statement {ast.unparse(node)[:60]!r}")
    return out


def _module_names(tree) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node in tree.body:
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    for node in tree.body:
        for t in (node.targets if isinstance(node, ast.Assign) else
                  [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []):
            names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
    return names


def _module_source(mod) -> Optional[str]:
    path = getattr(mod, "__file__", None)
    if not path:
        return None
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _module_state_issues(mod) -> List[str]:
    """Static scan of a policy module's source for state kept outside ``ctx.state`` (see the section comment)."""
    src = _module_source(mod)
    if src is None:
        return [f"source of {mod!r} is not available"]
    key = (getattr(mod, "__file__", None), hashlib.sha256(src.encode()).hexdigest())
    if key not in _AST_CACHE:
        tree = ast.parse(src)
        _AST_CACHE[key] = _body_issues(tree.body, _module_names(tree), "")
    return list(_AST_CACHE[key])


def _fp_value(x) -> str:
    if isinstance(x, np.ndarray):
        return f"nd{x.dtype}{x.shape}" + hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    try:
        return hashlib.sha256(pickle.dumps(x, protocol=4)).hexdigest()
    except Exception:
        return repr(x)


def _module_fingerprint(mod) -> Optional[str]:
    """Digest of the module's non-code state (identity-sensitive where content cannot be pickled)."""
    if not isinstance(mod, types.ModuleType):
        return None
    name = mod.__name__
    parts = []
    for k, v in sorted(vars(mod).items()):
        if k.startswith("__") or isinstance(v, types.ModuleType):
            continue
        if callable(v) and getattr(v, "__module__", None) != name and not hasattr(v, "cache_info"):
            parts.append((k, id(v)))  # imported function or class: only its binding
        elif isinstance(v, type):
            parts.append((k, [(a, _fp_value(b)) for a, b in sorted(vars(v).items())
                              if not a.startswith("__") and not callable(b)
                              and not isinstance(b, (staticmethod, classmethod, property))]))
        elif callable(v):
            cells = [c.cell_contents for c in (getattr(v, "__closure__", None) or ())
                     if not _is_empty_cell(c)]
            ci = v.cache_info() if hasattr(v, "cache_info") else None
            parts.append((k, _fp_value(getattr(v, "__defaults__", None)),
                          _fp_value(getattr(v, "__kwdefaults__", None)),
                          [_fp_value(c) for c in cells], _fp_value(dict(getattr(v, "__dict__", {}) or {})),
                          repr(ci)))
        else:
            parts.append((k, _fp_value(v)))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def _is_empty_cell(c) -> bool:
    try:
        c.cell_contents
        return False
    except ValueError:
        return True


def _policy_module(sim: Simulator):
    pol = getattr(sim._scheduler, "_policy", None)
    return pol if isinstance(pol, types.ModuleType) else None


def _unforkable_reason(h: "SimHandle", trust_module_state: bool = False) -> Optional[str]:
    gs = h.sim._scheduler
    if isinstance(gs, LearnedGlobalScheduler):
        # An exact RL fork would need: a forced-action hook in LearnedGlobalScheduler.schedule (it does not go through
        # _call) that records the forced action in the transition, _routed, the action stats and the who-acted flag;
        # per-handle copies of the design module's mutable globals (rl_loto_*: _PEND/_ORDER/_BASE feed the rewards);
        # in training mode, per-fork copies of the network, target network, optimiser and replay buffer, and
        # per-fork checkpoint/log paths. learned_global_scheduler.py is in evaluate._VERSION_EXCLUDE, so such a hook
        # would not change code_version.
        return ("learned (RL) schedulers are not forkable: the design module keeps mutable module-level state "
                "shared by all forks, and under a forced decision the learner would record its own action")
    pol = _policy_module(h.sim)
    if pol is None or trust_module_state:
        return None
    issues = _module_state_issues(pol)
    if issues:
        return (f"policy module {getattr(pol, '__file__', pol)!r} may keep state outside ctx.state, which all forks "
                f"would share: {'; '.join(issues[:5])}. Keep state in ctx.state, or pass trust_module_state=True "
                "if each rollout runs in its own process")
    if h.policy_fp is not None and _module_fingerprint(pol) != h.policy_fp:
        return (f"policy module {getattr(pol, '__file__', pol)!r} changed its module-level state since the handle "
                "was built; all forks would share it (pass trust_module_state=True if each rollout runs in its own "
                "process)")
    return None


_LIVE: List["SimHandle"] = []  # handles whose state is installed, innermost last


class SimHandle:
    """A simulator plus its private process-global state (id counters, global RNGs, predictor clamp counts)."""

    def __init__(self, sim: Simulator, counters: Optional[Dict[type, object]] = None, rng_state=None,
                 clamp_hits: Optional[dict] = None, owns_output_dir: bool = True, policy_fp=MISSING):
        self.sim = sim
        # fingerprint of the policy module's state when the simulation was built (compared at every fork)
        self.policy_fp = _module_fingerprint(_policy_module(sim)) if policy_fp is MISSING else policy_fp
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
        share(getattr(gs, "_policy", None))  # a module; its state is checked by _unforkable_reason
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

    def fork(self, output_dir: Optional[str] = None, trust_module_state: bool = False) -> "SimHandle":
        """An independent copy of this simulation at its current point (callable inside a ``run`` callback).

        A pending ``override_next_decision`` is part of the state and is copied (bound to the copy's scheduler).
        Without ``output_dir`` the fork cannot ``write_output`` (it would overwrite the parent's files).
        Refuses learned schedulers and policy modules that may keep state outside ``ctx.state`` (see the
        policy-module state section); ``trust_module_state=True`` skips the module checks."""
        reason = _unforkable_reason(self, trust_module_state)
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
                      owns_output_dir=output_dir is not None, policy_fp=self.policy_fp)
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
    def last_override(self) -> Optional[dict]:
        """``{"policy_choice", "executed"}`` of the last decision taken under ``override_next_decision``."""
        return self.sim._scheduler.__dict__.get(_LAST_ATTR)

    @property
    def now(self) -> float:
        return self.sim._time


# ---------------------------------------------------------------- forced decisions

_FORCE_ATTR = "_sim_fork_force"
_LAST_ATTR = "_sim_fork_last"
_FIFO_KEYS = {"attn_queues", "attn_sums"}
_FIFO_HELPER = '''
def _attended_context(replicas, state):
    queues = state.get("attn_queues")
    if queues is None or len(queues) != len(replicas):
        queues = [deque() for _ in replicas]
        state["attn_queues"] = queues
        state["attn_sums"] = [0] * len(replicas)
    sums = state["attn_sums"]
    for i, r in enumerate(replicas):
        q = queues[i]
        while len(q) > r.outstanding_requests:
            sums[i] -= q.popleft()
    return queues, sums
'''
_FIFO_TAIL = ["queues[pick].append(n)", "attended[pick] += n", "return pick"]
_FIFO_CALL = ast.unparse(ast.parse("queues, attended = _attended_context(replicas, ctx.state)").body[0])


def override_next_decision(h: SimHandle, choice_for_first: Optional[int], policy_state: str = "auto",
                           record: Optional[dict] = None, route_first_only: bool = False) -> None:
    """Force the next routing call's first request onto replica ``choice_for_first`` (``None``: record only).

    Semantics: a forced decision means the router executed action ``a = choice_for_first``. The policy is still
    called (its scoring and its ``ctx.rng`` draws happen as in the unforced call); its return value is replaced by
    ``a``, the scheduler's ledger (outstanding counts, pending prefill, in-flight/active prefixes, session map)
    books ``a``, and the state the policy keeps about its own decision must end as if it had chosen ``a``. When the
    policy's own choice is ``a`` nothing changes. Otherwise exactness is guaranteed only by:

    - the policy module's ``on_forced(request, replicas, chosen, forced, ctx)`` hook, called after ``choose``
      returned ``chosen``; it must leave ``ctx.state`` (and anything else it keeps) as if ``choose`` had returned
      ``forced``;
    - the attended-context FIFO commit, for policies recognised by their source (tail4_ni*, l3_*: the canonical
      ``_attended_context`` trim, ``n = request.num_prefill_tokens`` and a choose() that ends with
      ``queues[pick].append(n); attended[pick] += n; return pick``, with ``ctx.state`` and the queues touched
      nowhere else) or declaring ``FORCED_COMMIT = "attn_fifo"``; the call's change is then verified against the
      views (each queue trimmed to ``min(len, outstanding)``, then exactly ``n`` appended at the pick) and the entry
      is moved to ``a``;
    - ``policy_state="independent"``: the caller asserts that the policy's state update does not depend on its pick
      (claude_r1 ``sessions``, completion_r2 ``session_prompt``/``scarcity_ewma``, t4_70b ``prompt_ewma``: checked
      by reading the code in review E r2). Checked: refused when the state holds anything per replica (a sequence
      of num_replicas entries, or a dict keyed by replica indices), which is where a pick-dependent update lives;
    - a policy with an empty ``ctx.state`` before and after the call and a module that passes the static
      module-state scan.

    Anything else raises ``NotImplementedError`` (e.g. SGLang/AIBrix, whose prefix index may be unchanged at their
    own pick but would change at ``a``). A raise happens inside the routing call; the handle is then mid-decision
    and must be discarded. ``route()`` policies: a burst of more than one request with ``a`` different from the
    policy's first choice raises, since the rest of the burst was routed for the policy's own first choice, unless
    ``route_first_only=True`` accepts that mixed action.

    ``record``: a dict that receives ``policy_choice`` (the policy's own choice for the first request) and
    ``executed`` (the replica booked) when the decision happens; ``None`` as the choice only records. The same
    values are kept on the handle (``last_override``). The pending override is stored on ``h``'s scheduler, so it
    works unchanged in an ``os.fork`` child and is copied by ``SimHandle.fork`` (each copy applies it to itself
    once and fills its own copy of ``record``).
    """
    gs = h.sim._scheduler
    if isinstance(gs, LearnedGlobalScheduler) or not isinstance(gs, EvolvedGlobalScheduler):
        raise NotImplementedError(f"forced decisions need an evolved (non-learned) global scheduler, got "
                                  f"{type(gs).__name__}")
    if _FORCE_ATTR in gs.__dict__:
        raise RuntimeError("an override is already pending on this handle")
    if policy_state not in ("auto", "independent"):
        raise ValueError(f"policy_state must be 'auto' or 'independent', got {policy_state!r}")
    if choice_for_first is not None:
        gs._check(choice_for_first)  # PolicyError unless an int in 0..N-1 (a negative index would wrap)
        if hasattr(gs._policy, "route") and len(gs._request_queue) > 1 and not route_first_only:
            raise NotImplementedError("route() policy with a pending burst of more than one request: forcing only "
                                      "the first request is a mixed action (pass route_first_only=True to accept)")
    gs.__dict__[_FORCE_ATTR] = (choice_for_first, policy_state, record, route_first_only)
    gs.__dict__.pop(_LAST_ATTR, None)
    gs._call = types.MethodType(_forcing_call, gs)  # deepcopy rebinds a bound method to the copied scheduler


def _forcing_call(self, fn, *args):
    pol = self._policy
    is_route = fn is getattr(pol, "route", None)
    if not is_route and fn is not getattr(pol, "choose", None):  # observe(): pass through
        return type(self)._call(self, fn, *args)
    forced, mode, record, route_first_only = self.__dict__.pop(_FORCE_ATTR)
    del self.__dict__["_call"]
    before = copy.deepcopy(self._ctx.state)
    out = type(self)._call(self, fn, *args)
    if is_route:
        if not isinstance(out, (list, tuple)) or not out:
            return out  # schedule() raises its PolicyError
        chosen = self._check(out[0])
    else:
        chosen = self._check(out)
    executed = chosen if forced is None else forced
    if executed != chosen:
        if is_route and len(args[0]) > 1 and not route_first_only:
            raise NotImplementedError("route() burst of more than one request: forcing only the first request is a "
                                      "mixed action (pass route_first_only=True to accept). Discard the handle.")
        request = args[0][0] if is_route else args[0]
        _reconcile(pol, before, request, args[1], chosen, executed, self._ctx, mode)
    last = {"policy_choice": chosen, "executed": executed}
    self.__dict__[_LAST_ATTR] = last
    if record is not None:
        record.update(last)
    if is_route:
        return [executed] + list(out[1:])
    return executed


def _per_replica(v, num: int) -> bool:
    if isinstance(v, (list, tuple, deque, np.ndarray)):
        return len(v) == num
    if isinstance(v, dict):
        return bool(v) and all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < num for k in v)
    return False


def _reconcile(pol, before: dict, request, replicas, chosen: int, forced: int, ctx, mode: str) -> None:
    after = ctx.state
    hook = getattr(pol, "on_forced", None)
    if hook is not None:
        hook(request, replicas, chosen, forced, ctx)
        return
    issues = _module_state_issues(pol)
    where = f"cannot force replica {forced} (policy chose {chosen}) exactly"
    if issues:
        raise NotImplementedError(f"{where}: the policy module may keep state outside ctx.state "
                                  f"({'; '.join(issues[:3])}); define on_forced() in the policy. Discard the handle.")
    if mode == "independent":
        num = ctx.num_replicas
        per = sorted(k for st in (before, after) for k, v in st.items() if _per_replica(v, num))
        if per:
            raise NotImplementedError(f"{where}: policy_state='independent' refused, ctx.state keeps per-replica "
                                      f"entries {sorted(set(per))} (where a pick-dependent update lives); define "
                                      "on_forced() in the policy. Discard the handle.")
        return
    if not before and not after:
        return
    if set(after) == _FIFO_KEYS and set(before) <= _FIFO_KEYS and _fifo_policy(pol) \
            and _fifo_move(before, after, request, replicas, chosen, forced):
        return
    raise NotImplementedError(
        f"{where}: ctx.state (keys {sorted(after)}) is not one sim_fork can reconcile. Define on_forced() in the "
        "policy, or pass policy_state='independent' if its state update does not depend on its pick. The handle "
        "is mid-decision; discard it.")


_FIFO_OK: Dict[tuple, bool] = {}


def _fifo_policy(pol) -> bool:
    """True when the policy declares the attended-context FIFO or its source has exactly the finalists' shape."""
    if getattr(pol, "FORCED_COMMIT", None) == "attn_fifo":
        return True
    src = _module_source(pol)
    if src is None:
        return False
    key = (getattr(pol, "__file__", None), hashlib.sha256(src.encode()).hexdigest())
    if key not in _FIFO_OK:
        try:
            _FIFO_OK[key] = _fifo_source_ok(ast.parse(src))
        except SyntaxError:
            _FIFO_OK[key] = False
    return _FIFO_OK[key]


def _fifo_source_ok(tree) -> bool:
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    helper, choose = fns.get("_attended_context"), fns.get("choose")
    if helper is None or choose is None:
        return False
    body = helper.body[1:] if (helper.body and isinstance(helper.body[0], ast.Expr)
                               and isinstance(helper.body[0].value, ast.Constant)) else helper.body
    canon = ast.parse(_FIFO_HELPER).body[0]
    if ast.dump(ast.Module(body=body, type_ignores=[])) != ast.dump(ast.Module(body=canon.body, type_ignores=[])) \
            or ast.dump(helper.args) != ast.dump(canon.args):
        return False
    if [ast.unparse(s) for s in choose.body[-3:]] != _FIFO_TAIL:
        return False
    top = [ast.unparse(s) for s in choose.body]
    if top.count("n = request.num_prefill_tokens") != 1 \
            or top.count(_FIFO_CALL) != 1:
        return False
    if [ast.unparse(s) for s in choose.body].index(_FIFO_CALL) \
            >= len(choose.body) - 3:
        return False
    # n, queues and attended are bound once in choose; nothing else stores into them, mutates them or touches
    # ctx.state (outside the helper call)
    stores = [x for x in ast.walk(choose) if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del))]
    if sum(x.id == "n" for x in stores) != 1 or sum(x.id == "queues" for x in stores) != 1 \
            or sum(x.id == "attended" for x in stores) != 1:
        return False
    tail = choose.body[-3:-1]
    for node in ast.walk(choose):
        if any(node is t or _within(t, node) for t in tail):
            continue
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)) \
                and _root_name(node) in ("queues", "attended"):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and _root_name(node.func.value) in ("queues", "attended"):
            return False
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn is helper:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr == "state" \
                    and not (fn is choose and _in_helper_call(choose, node)):
                return False
            if isinstance(node, ast.Name) and node.id == "_attended_context" and fn is not choose:
                return False
    return True


def _within(parent, node) -> bool:
    return any(x is node for x in ast.walk(parent))


def _in_helper_call(choose, node) -> bool:
    for s in choose.body:
        if ast.unparse(s) == _FIFO_CALL and _within(s, node):
            return True
    return False


def _fifo_move(before: dict, after: dict, request, replicas, chosen: int, forced: int) -> bool:
    """Verify that the call trimmed each queue to ``min(len, outstanding)`` of the views it was given (a suffix
    remains) and appended exactly ``n`` at ``chosen``; then move that entry to ``forced``. Returns False (state
    untouched) when the change has any other shape."""
    try:
        q1, s1 = after["attn_queues"], after["attn_sums"]
        q0 = before.get("attn_queues")
        num = len(q1)
        if (q0 is not None and len(q0) != num) or len(s1) != num or len(replicas) != num or not 0 <= forced < num:
            return False
        n = request.num_prefill_tokens
        for i in range(num):
            cur = list(q1[i])
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in cur):
                return False
            old = list(q0[i]) if q0 is not None else []
            kept = min(len(old), replicas[i].outstanding_requests)
            if i == chosen:
                if len(cur) != kept + 1 or cur[-1] != n:
                    return False
                cur = cur[:-1]
            if len(cur) != kept or old[len(old) - kept:] != cur or s1[i] != sum(q1[i]):
                return False
    except (TypeError, KeyError, AttributeError, IndexError):
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
