"""Router subclass that runs our routing policies inside LLMServingSim 2.0.

It mirrors ``vidur/scheduler/global_scheduler/evolved_global_scheduler.py`` (the policy contract and the
router-owned counters) on top of LLMServingSim's ``serving.core.router.Router``:

- The policy is a Python file defining ``choose(request, replicas, ctx) -> int`` (or ``route(burst, replicas, ctx)``)
  and optionally ``observe(outcome, ctx)``; or one of the built-in baselines in ``BUILTINS``.
- Views are Vidur's frozen dataclasses (``ReplicaView``, ``RequestView``, ``RequestOutcome``, ``PolicyContext``),
  imported from the vidur fork when it imports, else exact copies (``tests/test_lss2_adapter.py`` checks the copies).
- Router-owned counters, as Vidur defines them: ``outstanding_requests`` (+1 at routing, -1 at the request's end),
  ``pending_prefill_tokens`` (+ uncached prompt tokens at routing, - the same amount at the first token, once),
  the session's last replica, and the in-flight / active prompt-block ledgers behind ``inflight_prefix_tokens`` and
  ``active_prefix_tokens``.
- Polled replica state comes from the live LLMServingSim schedulers: ``waiting`` = len(sched.waiting), ``running`` =
  len(sched.running), ``kv_usage`` = NPU pool usage (pinned blocks / blocks, vLLM's definition, as Vidur's),
  ``kv_capacity_tokens`` = blocks * block size. ``cached_prefix_tokens`` is a side-effect-free probe of each replica's
  prefix cache (``sched.kv.get_computed_blocks``) made before any request of the burst is placed.
- Bursts: requests with the same arrival time routed in one call, as Vidur's GlobalScheduleEvent does. As in Vidur,
  the whole burst is decided before any of it is added to a replica (``waiting`` is the queue before the burst).
- Feedback: ``observe`` gets ``first_token`` when a request's prefill completes and ``end`` when it finishes, at
  the batch completion that produced them (each scheduler's ``add_done`` is wrapped on the instance).

Timing difference from Vidur (inherent to LLMServingSim's main loop): routing happens at the first simulator event at
or after a request's arrival, not at the arrival instant. ``ctx.now`` is still the arrival time (Vidur's definition);
the lag is recorded in ``route_lag_*``.
"""
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evolve.lss2.convert import BLOCK, expand_tokens

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ views: the vidur fork's, else exact copies

class _PolicyErrorCopy(RuntimeError):
    """The policy returned an invalid decision or raised."""


@dataclass(frozen=True)
class _ReplicaViewCopy:
    index: int
    outstanding_requests: int  # routed here by this router and not yet completed
    pending_prefill_tokens: int  # uncached prefill tokens routed here, not yet prefilled
    waiting: int  # replica's waiting queue length (before this burst is added)
    running: int  # replica's running set size
    kv_usage: float  # fraction of KV blocks in use, 0..1
    kv_capacity_tokens: int


@dataclass(frozen=True)
class _RequestViewCopy:
    arrived_at: float
    num_prefill_tokens: int
    session_id: Optional[int]
    cached_prefix_tokens: Tuple[int, ...]  # per replica index
    session_last_replica: int  # replica index this session was last routed to, -1 if new
    position_in_burst: int
    burst_size: int
    prefix_hashes: Tuple[int, ...] = ()
    block_tokens: int = 0
    inflight_prefix_tokens: Tuple[int, ...] = ()
    active_prefix_tokens: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _RequestOutcomeCopy:
    event: str
    replica: int
    arrived_at: float
    first_token_at: float
    completed_at: Optional[float]
    num_prefill_tokens: int
    num_cached_tokens: int
    num_decode_tokens: Optional[int]
    session_id: Optional[int]


class _PolicyContextCopy:
    def __init__(self, rng, num_replicas: int):
        self.rng = rng
        self.num_replicas = num_replicas
        self.now = 0.0
        self.state: Dict[str, Any] = {}


try:  # pragma: no cover - depends on the environment
    from vidur.scheduler.global_scheduler.evolved_global_scheduler import (  # noqa: F401
        PolicyContext, PolicyError, ReplicaView, RequestOutcome, RequestView)
    VIEWS_SOURCE = "vidur"
except Exception:  # vidur's import chain needs scikit-learn etc.; the lss2 env does not have them
    VIEWS_SOURCE = "copy"
    PolicyError, ReplicaView, RequestView = _PolicyErrorCopy, _ReplicaViewCopy, _RequestViewCopy
    RequestOutcome, PolicyContext = _RequestOutcomeCopy, _PolicyContextCopy


# ------------------------------------------------------------------ built-in baselines (= evolve/policies/*.py)

def _rr(request, replicas, ctx):
    i = ctx.state.get("rr_next", 0)
    ctx.state["rr_next"] = i + 1
    return i % len(replicas)


def _lor(request, replicas, ctx):
    least = min(r.outstanding_requests for r in replicas)
    return ctx.rng.choice([r.index for r in replicas if r.outstanding_requests == least])


def _sticky_lor(request, replicas, ctx):
    if request.session_last_replica >= 0:
        return request.session_last_replica
    return _lor(request, replicas, ctx)


def _lop_uncached(request, replicas, ctx):
    pending = [r.pending_prefill_tokens for r in replicas]
    imbalance = []
    for j in range(len(replicas)):
        extra = request.num_prefill_tokens - request.cached_prefix_tokens[j]
        loads = [p + (extra if i == j else 0) for i, p in enumerate(pending)]
        imbalance.append(max(loads) - min(loads))
    best = min(imbalance)
    return ctx.rng.choice([j for j in range(len(replicas)) if imbalance[j] == best])


BUILTINS = {"rr": _rr, "lor": _lor, "sticky_lor": _sticky_lor, "lop_uncached": _lop_uncached}


# ------------------------------------------------------------------ policy loading

def _install_es_stubs() -> None:
    """Make ``evolve.rl_plan.rl_es`` importable without the Vidur simulator stack.

    rl_es imports evolve.evaluate, evolve.rl_plan.{c0, headroom, rl_residual} and evolve.sim_fork for its training
    loop; they pull in vidur, scipy and scikit-learn. The policy itself (``build_policy``) needs only
    ``c0.BASE`` (the 4873 file), ``c0_policy.load_base`` and numpy. The stubs carry exactly the module-level
    constants rl_es reads at import time; the policy code that runs is rl_es's own, unmodified.
    """
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__lss2_stub__ = True
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], child, mod)
        return mod

    def _unavailable(*a, **k):
        raise RuntimeError("stubbed for the LLMServingSim adapter (simulator stack not importable)")

    import evolve  # noqa: F401
    import evolve.rl_plan  # noqa: F401
    stub("evolve.evaluate", REPO=REPO)
    stub("evolve.sim_fork", build_simulator=_unavailable, after_os_fork=_unavailable,
         override_next_decision=_unavailable)
    stub("evolve.rl_plan.headroom")
    stub("evolve.rl_plan.c0", BASE=str((REPO / "evolve/finalists/v4/v4_487363424749.py").resolve()))
    # evolve.rl_plan.rl_residual constants (TRACES, N_REQ, TRAIN_DRAWS, EVAL_DRAWS), unused by the policy itself
    stub("evolve.rl_plan.rl_residual", TRACES=("Tool", "Conv"), N_REQ={"Tool": 23608, "Conv": 12031},
         TRAIN_DRAWS=tuple(range(22, 40)), EVAL_DRAWS=(40, 41, 42, 43))
    sys.modules.pop("evolve.rl_plan.rl_es", None)


IMPORT_MODE = {"mode": "n/a"}


def _exec_policy(path: str):
    spec = importlib.util.spec_from_file_location("evolved_policy", path)
    if spec is None or spec.loader is None:
        raise PolicyError(f"cannot load policy from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_policy(spec: str):
    """A built-in name or a policy file. Policy files that import the Vidur stack fall back to ``_install_es_stubs``
    when the full import chain is not available (recorded in ``IMPORT_MODE``)."""
    if spec in BUILTINS:
        IMPORT_MODE["mode"] = "builtin"
        return types.SimpleNamespace(choose=BUILTINS[spec], __name__=spec)
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    try:
        mod = _exec_policy(spec)
        IMPORT_MODE["mode"] = "native"
    except ImportError as e:
        IMPORT_MODE["mode"] = f"stubs (native import failed: {type(e).__name__}: {e})"
        _install_es_stubs()
        mod = _exec_policy(spec)
    if not (hasattr(mod, "route") or hasattr(mod, "choose")):
        raise PolicyError(f"policy {spec!r} defines neither route() nor choose()")
    return mod


# ------------------------------------------------------------------ the router core (independent of LLMServingSim)

class PolicyRouterCore:
    """Policy calls, router-owned counters and feedback. Needs ``self.prefill_schedulers`` (LLMServingSim
    ``Scheduler``s or fakes with ``waiting``, ``running``, ``inflight``, ``kv``, ``enable_prefix_caching``,
    ``model``, ``instance_id``, ``add_request``, ``add_done``) and ``self._pending_requests``/``_pending_idx``."""

    POLICY = None  # set by the launcher before LLMServingSim constructs the router
    SEED = 1
    REQUEST_CLS = None  # LLMServingSim's Request (used for the prefix probe)

    def _core_init(self):
        if self.POLICY is None:
            raise PolicyError("PolicyRouterCore.POLICY is not set")
        self._policy = self.POLICY
        scheds = self.prefill_schedulers
        R = len(scheds)
        self._R = R
        self._outstanding = [0] * R
        self._pending_prefill = [0] * R
        self._assigned: Dict[int, Tuple[int, int]] = {}
        self._session_last: Dict[int, int] = {}
        self._prefilled: set = set()
        self._inflight: List[Dict[int, int]] = [dict() for _ in range(R)]
        self._inflight_req: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        self._active: List[Dict[int, int]] = [dict() for _ in range(R)]
        self._active_req: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        self._ctx = PolicyContext(random.Random(self.SEED), R)
        self._meta: Dict[int, dict] = {}  # request id -> request data (arrival_s, session, hashes, ...)
        self.records: Dict[int, dict] = {}  # request id -> per-request outcome
        self.policy_calls = 0
        self.policy_seconds = 0.0
        self.observe_calls = 0
        self.observe_seconds = 0.0
        self.route_seconds = 0.0  # all adapter work at routing (probes, views, policy)
        self.route_lag_sum = 0.0
        self.route_lag_max = 0.0
        self.n_routed = 0
        self.n_bursts = 0
        self.max_burst = 0
        self._block_tokens = BLOCK
        for i, s in enumerate(scheds):
            s.add_done = self._wrap_add_done(i, s, s.add_done)

    # -------------------------------------------------------------- loading

    def _row_to_req_data(self, row: dict, req_id: int, enable_prefix_caching: bool) -> dict:
        n = int(row["input_toks"])
        rd = {
            "index": req_id,
            "input_toks": n,
            "output_toks": int(n + row["output_toks"]),
            "arrival_time_ns": int(row["arrival_time_ns"]),
            "arrival_s": float(row.get("arrival_s", row["arrival_time_ns"] / 1e9)),
            "session": row.get("session_id"),
            "row": row.get("row", req_id),
            "phash": tuple(row.get("prompt_hashes") or ()),
        }
        if enable_prefix_caching:
            if "input_tok_ids" in row:
                rd["input_hash_ids"] = row["input_tok_ids"]
            elif "tok_block_ids" in row:
                rd["input_hash_ids"] = expand_tokens(row["tok_block_ids"], n)
            else:
                rd["input_hash_ids"] = []
            rd["output_hash_ids"] = []
        return rd

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Flat JSONL rows (``evolve.lss2.convert``); absolute paths allowed. Agentic rows are not supported."""
        if not os.path.isabs(path):
            path = f"../{path}"  # LLMServingSim runs from astra-sim/
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        with open(path) as f:
            for line in f:
                if self.req_num > 0 and len(self._pending_requests) >= self.req_num:
                    break
                row = json.loads(line)
                if "sub_requests" in row:
                    raise ValueError("agentic sessions are not supported by the policy router")
                rid = self._next_request_id
                self._next_request_id += 1
                self._pending_requests.append(self._row_to_req_data(row, rid, enable_prefix_caching))
        self._pending_requests.sort(key=lambda r: r["arrival_time_ns"])  # stable: file order within a tick

    # -------------------------------------------------------------- views

    def _replica_views(self) -> List[ReplicaView]:
        views = []
        for i, s in enumerate(self.prefill_schedulers):
            pool = s.kv.npu_pool
            views.append(ReplicaView(
                index=i,
                outstanding_requests=self._outstanding[i],
                pending_prefill_tokens=self._pending_prefill[i],
                waiting=len(s.waiting),
                running=len(s.running),
                kv_usage=pool.usage(),
                kv_capacity_tokens=pool.num_blocks * pool.block_size,
            ))
        return views

    def _probe_request(self, rd: dict):
        return self.REQUEST_CLS(rd["index"], None, rd["input_toks"], rd["output_toks"], rd["arrival_time_ns"], 0,
                                rd.get("input_hash_ids", []), [])

    def _cached_prefix_tokens(self, probe) -> Tuple[int, ...]:
        out = []
        for s in self.prefill_schedulers:
            if not s.enable_prefix_caching:
                out.append(0)
                continue
            out.append(int(s.kv.get_computed_blocks(probe)[1]))
        return tuple(out)

    def _ledger_tokens(self, hashes: Tuple[int, ...], n: int, table) -> Tuple[int, ...]:
        out = []
        bt = self._block_tokens
        for counts in table:
            k = 0
            for h in hashes:
                if counts.get(h, 0) <= 0:
                    break
                k += 1
            out.append(min(k * bt, n))
        return tuple(out)

    def _ledger_drop(self, rid: int, req_tab, tables) -> None:
        info = req_tab.pop(rid, None)
        if info is None:
            return
        counts = tables[info[0]]
        for h in info[1]:
            c = counts.get(h, 0) - 1
            if c > 0:
                counts[h] = c
            else:
                counts.pop(h, None)

    # -------------------------------------------------------------- routing

    def route_arrived_requests(self, current_time_ns):
        pend = self._pending_requests
        routed = 0
        while self._pending_idx < len(pend) and pend[self._pending_idx]["arrival_time_ns"] <= current_time_ns:
            i = self._pending_idx
            t = pend[i]["arrival_time_ns"]
            j = i + 1
            while j < len(pend) and pend[j]["arrival_time_ns"] == t:
                j += 1
            self._route_burst(pend[i:j], current_time_ns)
            self._pending_idx = j
            routed += j - i
        return routed

    def _call(self, fn, *args, observe: bool = False):
        t0 = time.perf_counter()
        try:
            return fn(*args, self._ctx)
        except PolicyError:
            raise
        except Exception as e:
            raise PolicyError(f"policy raised {type(e).__name__}: {e}") from e
        finally:
            if observe:
                self.observe_calls += 1
                self.observe_seconds += time.perf_counter() - t0
            else:
                self.policy_calls += 1
                self.policy_seconds += time.perf_counter() - t0

    def _check(self, c) -> int:
        if isinstance(c, bool) or not isinstance(c, int):
            raise PolicyError(f"replica index must be an int, got {c!r}")
        if not 0 <= c < self._R:
            raise PolicyError(f"replica index {c} out of range 0..{self._R - 1}")
        return c

    def _route_burst(self, burst: List[dict], current_ns: int) -> None:
        t0 = time.perf_counter()
        self._ctx.now = burst[0]["arrival_s"]
        probes = [self._probe_request(rd) for rd in burst]
        cached = [self._cached_prefix_tokens(p) for p in probes]
        bt = self._block_tokens
        views = [
            RequestView(
                arrived_at=rd["arrival_s"],
                num_prefill_tokens=rd["input_toks"],
                session_id=rd["session"],
                cached_prefix_tokens=cached[k],
                session_last_replica=self._session_last.get(rd["session"], -1) if rd["session"] is not None else -1,
                position_in_burst=k,
                burst_size=len(burst),
                prefix_hashes=rd["phash"],
                block_tokens=bt,
                inflight_prefix_tokens=self._ledger_tokens(rd["phash"], rd["input_toks"], self._inflight),
                active_prefix_tokens=self._ledger_tokens(rd["phash"], rd["input_toks"], self._active),
            )
            for k, rd in enumerate(burst)
        ]
        picks = []
        if hasattr(self._policy, "route"):
            choices = self._call(self._policy.route, views, self._replica_views())
            if not isinstance(choices, (list, tuple)) or len(choices) != len(burst):
                raise PolicyError(f"route() must return {len(burst)} replica indices, got {choices!r}")
            for k, (rd, c) in enumerate(zip(burst, choices)):
                c = self._check(c)
                self._assign(rd, c, cached[k])
                picks.append(c)
        else:
            for k, rd in enumerate(burst):
                rv = views[k]
                if rd["session"] is not None:
                    rv = replace(rv, session_last_replica=self._session_last.get(rd["session"], -1))
                c = self._check(self._call(self._policy.choose, rv, self._replica_views()))
                self._assign(rd, c, cached[k])
                picks.append(c)
        for rd, c, probe in zip(burst, picks, probes):
            self._add_to_replica(rd, c, probe)
        lag = max(0.0, (current_ns - burst[0]["arrival_time_ns"]) / 1e9)
        self.route_lag_sum += lag * len(burst)
        self.route_lag_max = max(self.route_lag_max, lag)
        self.n_routed += len(burst)
        self.n_bursts += 1
        self.max_burst = max(self.max_burst, len(burst))
        self.route_seconds += time.perf_counter() - t0

    def _assign(self, rd: dict, idx: int, cached: Tuple[int, ...]) -> None:
        rid = rd["index"]
        uncached = rd["input_toks"] - cached[idx]
        self._outstanding[idx] += 1
        self._pending_prefill[idx] += uncached
        self._assigned[rid] = (idx, uncached)
        hashes = rd["phash"]
        if hashes:
            for table, req_tab in ((self._inflight, self._inflight_req), (self._active, self._active_req)):
                counts = table[idx]
                for h in hashes:
                    counts[h] = counts.get(h, 0) + 1
                req_tab[rid] = (idx, hashes)
        if rd["session"] is not None:
            self._session_last[rd["session"]] = idx
        self._meta[rid] = rd
        self.records[rid] = {"id": rid, "row": rd["row"], "replica": idx, "arrival_s": rd["arrival_s"],
                             "arrival_ns": rd["arrival_time_ns"], "num_prefill_tokens": rd["input_toks"],
                             "num_decode_tokens": rd["output_toks"] - rd["input_toks"],
                             "session_id": rd["session"], "probe_cached": cached[idx]}

    def _add_to_replica(self, rd: dict, idx: int, probe) -> None:
        s = self.prefill_schedulers[idx]
        if s.enable_prefix_caching:
            s.add_request([rd["index"], s.model, rd["input_toks"], rd["output_toks"], rd["arrival_time_ns"],
                           s.instance_id, rd.get("input_hash_ids", []), rd.get("output_hash_ids", [])],
                          is_init=self._is_init)
            # share the probe's chained block hashes (they do not depend on the replica) instead of rehashing
            hashes = getattr(probe, "block_hashes", None)
            if hashes is not None:
                for r in reversed(s.waiting):
                    if r.id == rd["index"]:
                        r.block_hashes = hashes
                        break
        else:
            s.add_request([rd["index"], s.model, rd["input_toks"], rd["output_toks"], rd["arrival_time_ns"],
                           s.instance_id], is_init=self._is_init)

    # -------------------------------------------------------------- feedback

    def _wrap_add_done(self, i: int, sched, orig):
        def add_done(id, sys_, finish):
            batch = None
            for b in sched.inflight:
                if b.batch_id == id - 1:
                    batch = b
                    break
            reqs = list(batch.requests) if batch is not None else ()
            out = orig(id, sys_, finish)
            for r in reqs:
                if r.id in self._assigned and r.id not in self._prefilled and r.ttft >= 0:
                    self._on_first_token(r, finish)
            for r in out[2]:
                self._on_end(r, finish)
            return out
        return add_done

    def _observe(self, rid: int, idx: int, event: str, req) -> None:
        observe = getattr(self._policy, "observe", None)
        if observe is None:
            return
        rec = self.records[rid]
        end = event == "end"
        outcome = RequestOutcome(
            event=event, replica=idx, arrived_at=rec["arrival_s"], first_token_at=rec["first_token_s"],
            completed_at=rec["completed_s"] if end else None, num_prefill_tokens=rec["num_prefill_tokens"],
            num_cached_tokens=rec["num_cached_tokens"], num_decode_tokens=rec["num_decode_tokens"] if end else None,
            session_id=rec["session_id"])
        saved = self._ctx.now
        self._ctx.now = outcome.completed_at if end else outcome.first_token_at
        try:
            self._call(observe, outcome, observe=True)
        finally:
            self._ctx.now = saved

    def _on_first_token(self, req, finish_ns: int) -> None:
        rid = req.id
        idx, uncached = self._assigned[rid]
        self._pending_prefill[idx] -= uncached
        self._prefilled.add(rid)
        self._ledger_drop(rid, self._inflight_req, self._inflight)
        rec = self.records[rid]
        rec["first_token_s"] = finish_ns / 1e9
        rec["num_cached_tokens"] = int(getattr(req, "npu_cache_hit", 0))
        self._observe(rid, idx, "first_token", req)

    def _on_end(self, req, finish_ns: int) -> None:
        rid = req.id
        info = self._assigned.pop(rid, None)
        rec = self.records.get(rid)
        if rec is not None:
            rec["completed_s"] = finish_ns / 1e9
            rec["ttft_s"] = req.ttft / 1e9
            rec["e2e_s"] = req.latency / 1e9
            rec["itl_ns"] = list(req.itl)
            rec["preemptions"] = int(getattr(req, "num_preemptions", 0))
            rec.setdefault("first_token_s", (req.arrival + req.ttft) / 1e9)
            rec.setdefault("num_cached_tokens", int(getattr(req, "npu_cache_hit", 0)))
        if info is not None:
            self._observe(rid, info[0], "end", req)
            self._outstanding[info[0]] -= 1
            if rid not in self._prefilled:  # defensive: first token not seen
                self._pending_prefill[info[0]] -= info[1]
            self._prefilled.discard(rid)
        self._ledger_drop(rid, self._inflight_req, self._inflight)
        self._ledger_drop(rid, self._active_req, self._active)

    def stats(self) -> dict:
        return {"policy_calls": self.policy_calls, "policy_seconds": self.policy_seconds,
                "observe_calls": self.observe_calls, "observe_seconds": self.observe_seconds,
                "route_seconds": self.route_seconds, "n_routed": self.n_routed, "n_bursts": self.n_bursts,
                "max_burst": self.max_burst,
                "route_lag_mean_s": self.route_lag_sum / self.n_routed if self.n_routed else float("nan"),
                "route_lag_max_s": self.route_lag_max,
                "views_source": VIEWS_SOURCE, "policy_import": IMPORT_MODE["mode"],
                "kv_blocks": [s.kv.npu_pool.num_blocks for s in self.prefill_schedulers],
                "end_outstanding": list(self._outstanding), "end_pending_prefill": list(self._pending_prefill)}


def make_router_class(base_cls, request_cls, policy, seed: int):
    """A subclass of LLMServingSim's ``Router`` (``base_cls``) that routes with ``policy``."""

    class PolicyRouter(PolicyRouterCore, base_cls):
        POLICY = policy
        SEED = seed
        REQUEST_CLS = request_cls
        INSTANCE = None

        def __init__(self, num_instances, schedulers, req_num, routing_policy="RR", seed=42):
            base_cls.__init__(self, num_instances, schedulers, req_num, "RR", seed)
            if self.decode_schedulers:
                raise ValueError("P/D disaggregation is not supported by the policy router")
            self._core_init()
            type(self).INSTANCE = self

    return PolicyRouter
