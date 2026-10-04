"""
Learned (RL) global scheduler built on the evolved-router harness (EXP12).

The router sees exactly what an evolved policy sees (``ReplicaView``/``RequestView``:
no output lengths, no future arrivals, no simulator objects). A *design module* at
``policy_path`` decides what is learned and how:

    def replica_features(request, replica, replicas, ctx) -> sequence of floats
        Feature vector of one candidate replica for this request (fixed length).
        ``replicas`` is the full list of views, so cluster context may be included.
    def reward(outcome) -> float                                   (optional*)
        Reward of one routing decision, from the request's observed outcome
        (``Outcome``: latencies are only known once the request has finished).
    def interval_reward(replicas_before, replicas_after, dt, ctx) -> float   (optional*)
        Reward accrued between this decision and the next one (dt simulated seconds),
        e.g. -dt * (requests in system): the Bradtke & Duff (1995) interval cost, whose
        undiscounted sum equals total latency (Little's law). Known at the next decision,
        so it needs no completion and charges a decision for slowing *other* requests.
    * at least one of the two; when both are defined the rewards add.
    def candidate_mask(request, replicas, ctx) -> list of bools              (optional)
        Replicas the learner may choose for this request (e.g. exclude replicas far
        above the mean backlog). Applied to greedy and random actions and to the
        Double-DQN argmax at the successor state; an all-False mask allows every replica.
    CONFIG = {...}
        Learning settings; see ``DEFAULTS`` for the keys and their ranges.

The harness owns the learner: one MLP with shared weights scores every replica,
Q(s, a) = f(x_a) (per-replica scoring as in Decima and Lodestar; independent of the
replica count). Two algorithms:

* ``bandit``: regression of Q(s, a) on the decision's own reward (gamma = 0).
* ``dqn``: Double DQN (van Hasselt et al., 2016) on a semi-MDP: the successor state of
  decision i is the state at decision i+1 and the discount is exp(-beta * dt) for the
  simulated time dt between them (Bradtke & Duff, 1995). Target:
  y = r + exp(-beta dt) * Q_target(s', argmax_a Q_online(s', a)), Huber loss, hard
  target-network copies every ``target_update`` gradient steps.

Optional teacher (``teacher_path``, an evolved policy module): during training it can
act with probability ``teacher_mix`` (jump-start, Uchendu et al., 2023), and a DQfD
large-margin loss (Hester et al., 2018) pulls the greedy action towards the teacher's
with weight ``imitation_weight``.
"""

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.entities.replica import Replica
from vidur.scheduler.global_scheduler.evolved_global_scheduler import (
    EvolvedGlobalScheduler,
    PolicyError,
    load_policy,
)
from vidur.types.replica_id import ReplicaId

DEFAULTS = {
    "algorithm": "bandit",  # "bandit" | "dqn"
    "beta": 0.05,  # dqn: discount per simulated second, gamma_i = exp(-beta * dt)
    "hidden": [64, 64],
    "lr": 1e-3,
    "batch_size": 64,
    "buffer_size": 50000,
    "train_every": 4,  # decisions per gradient step
    "learning_starts": 500,  # completed transitions before training starts
    "target_update": 500,  # gradient steps between target-network copies (dqn)
    "epsilon_start": 0.2,
    "epsilon_end": 0.02,
    "epsilon_decay_decisions": 20000,
    "teacher_mix": 0.0,  # probability the teacher acts (training only), at the start
    "imitation_weight": 0.0,  # DQfD margin-loss weight, at the start
    # Teacher phase-out (kickstarting, Schmitt et al. 2018): both decay linearly to their *_end
    # values over teacher_decay_decisions (counted across episodes). None = no decay.
    "teacher_mix_end": None,
    "imitation_weight_end": None,
    "teacher_decay_decisions": 30000,
    "imitation_margin": 0.5,
    "grad_clip": 10.0,
    "reward_clip": 100.0,  # harness safety bound on |reward|
    # Keep the replay buffer across training episodes (saved beside the checkpoint), as in
    # standard DQN; without it every episode starts empty and the net forgets earlier episodes'
    # workloads. False reproduces protocol v1/v2 behaviour.
    "persist_buffer": True,
}
RANGES = {
    "beta": (0.0, 10.0), "lr": (1e-5, 1e-1), "batch_size": (8, 1024), "buffer_size": (1000, 500000),
    "train_every": (1, 64), "learning_starts": (0, 100000), "target_update": (1, 100000),
    "epsilon_start": (0.0, 1.0), "epsilon_end": (0.0, 1.0), "epsilon_decay_decisions": (1, 10**7),
    "teacher_mix": (0.0, 1.0), "imitation_weight": (0.0, 100.0), "imitation_margin": (0.0, 100.0),
    "teacher_decay_decisions": (1, 10**7),
    "grad_clip": (0.01, 1000.0), "reward_clip": (0.01, 1e6),
}
MAX_FEATURES = 64
MAX_HIDDEN = 512


def resolve_config(user: Optional[dict]) -> dict:
    cfg = dict(DEFAULTS)
    for k, v in (user or {}).items():
        if k not in DEFAULTS:
            raise PolicyError(f"unknown CONFIG key {k!r}")
        cfg[k] = v
    if not isinstance(cfg["persist_buffer"], bool):
        raise PolicyError(f"CONFIG persist_buffer must be True or False, got {cfg['persist_buffer']!r}")
    if cfg["algorithm"] not in ("bandit", "dqn"):
        raise PolicyError(f"CONFIG algorithm must be 'bandit' or 'dqn', got {cfg['algorithm']!r}")
    for k, (lo, hi) in (("teacher_mix_end", (0.0, 1.0)), ("imitation_weight_end", (0.0, 100.0))):
        if cfg[k] is not None and (not isinstance(cfg[k], (int, float)) or isinstance(cfg[k], bool)
                                   or not lo <= cfg[k] <= hi):
            raise PolicyError(f"CONFIG {k}={cfg[k]!r} outside [{lo}, {hi}] (or None)")
    for k, (lo, hi) in RANGES.items():
        if not isinstance(cfg[k], (int, float)) or isinstance(cfg[k], bool) or not lo <= cfg[k] <= hi:
            raise PolicyError(f"CONFIG {k}={cfg[k]!r} outside [{lo}, {hi}]")
    h = cfg["hidden"]
    if (not isinstance(h, (list, tuple)) or not 1 <= len(h) <= 4
            or not all(isinstance(d, int) and 1 <= d <= MAX_HIDDEN for d in h)):
        raise PolicyError(f"CONFIG hidden must be 1-4 ints in [1, {MAX_HIDDEN}], got {h!r}")
    for k in ("batch_size", "buffer_size", "train_every", "learning_starts", "target_update",
              "epsilon_decay_decisions"):
        cfg[k] = int(cfg[k])
    return cfg


@dataclass(frozen=True)
class Outcome:
    """What the router learns about a request after it finished (no future information)."""

    ttft: float  # arrival -> first token (s)
    e2e: float  # arrival -> last token (s)
    tpot: Optional[float]  # (e2e - ttft) / (n_decode - 1); None for single-token outputs
    num_prefill_tokens: int
    num_decode_tokens: int  # known only now, after completion
    cached_tokens_at_routing: int  # cached prefix on the chosen replica when routed
    replica: int
    routed_at: float


def _build_net(torch, in_dim: int, hidden: List[int]):
    layers, d = [], in_dim
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


class _Decision:
    __slots__ = ("rid", "x", "a", "t", "teacher_a", "views", "mask", "reward", "r_interval", "next_x",
                 "next_mask", "dt", "done")

    def __init__(self, rid, x, a, t, teacher_a, views, mask):
        self.rid, self.x, self.a, self.t, self.teacher_a, self.views = rid, x, a, t, teacher_a, views
        self.mask = mask
        self.next_mask = None
        self.reward = None  # outcome reward (None until the request completes)
        self.r_interval = None  # interval reward (None until the next decision)
        self.next_x = None
        self.dt = 0.0
        self.done = False


class LearnedGlobalScheduler(EvolvedGlobalScheduler):
    REQUIRES_ROUTER = False

    def __init__(self, config: SimulationConfig, replicas: Dict[ReplicaId, Replica]):
        import torch

        torch.set_num_threads(1)  # many simulations run side by side; keep runs deterministic
        self._torch = torch
        super().__init__(config, replicas)
        gs = config.cluster_config.global_scheduler_config
        design = self._policy
        if not callable(getattr(design, "replica_features", None)):
            raise PolicyError("design module must define replica_features()")
        self._outcome_reward = getattr(design, "reward", None)
        self._interval_reward = getattr(design, "interval_reward", None)
        self._mask_fn = getattr(design, "candidate_mask", None)
        if not (callable(self._outcome_reward) or callable(self._interval_reward)):
            raise PolicyError("design module must define reward(outcome) and/or interval_reward(...)")
        self._cfg = resolve_config(getattr(design, "CONFIG", None))
        self._training = gs.mode == "training"
        self._checkpoint_out = gs.checkpoint_out
        self._teacher = load_policy(gs.teacher_path) if gs.teacher_path else None
        if self._teacher is not None and not hasattr(self._teacher, "choose"):
            raise PolicyError("teacher policy must define choose()")
        self._teacher_ctx = None
        if self._teacher is not None:
            from vidur.scheduler.global_scheduler.evolved_global_scheduler import PolicyContext

            self._teacher_ctx = PolicyContext(random.Random(gs.seed + 1), len(self._replica_ids))
        self._rng = random.Random(gs.seed)
        self._net = None
        self._target = None
        self._opt = None
        self._feature_dim = None
        self._pending_ckpt = None
        self._decisions: Dict[int, _Decision] = {}  # request id -> decision awaiting reward/successor
        self._routed: Dict[int, Tuple[float, int, int]] = {}  # request id -> (routed_at, replica, cached)
        self._prev: Optional[_Decision] = None
        self._buffer: List[tuple] = []
        self._buf_pos = 0
        self._n_decisions = 0  # carried across episodes via the checkpoint (epsilon schedule)
        self._n_steps = 0
        self._n_stored = 0
        if gs.checkpoint_in:  # after the counters: loading restores them
            self._pending_ckpt = torch.load(gs.checkpoint_in, map_location="cpu", weights_only=False)
            self._init_net(int(self._pending_ckpt["feature_dim"]))
            if self._training and self._cfg["persist_buffer"]:
                self._load_buffer(gs.checkpoint_in + ".buffer.npz")
        self._stats = {"loss": [], "q": [], "reward": [], "interval_reward": [], "explore": 0, "teacher_acted": 0,
                       "teacher_agree": 0, "teacher_seen": 0, "actions": [0] * len(self._replica_ids),
                       "train_seconds": 0.0, "dropped_unfinished": 0}

    # ----------------------------------------------------------------- network

    def _init_net(self, feature_dim: int) -> None:
        torch = self._torch
        if not 1 <= feature_dim <= MAX_FEATURES:
            raise PolicyError(f"replica_features must return 1..{MAX_FEATURES} values, got {feature_dim}")
        self._feature_dim = feature_dim
        self._net = _build_net(torch, feature_dim, list(self._cfg["hidden"]))
        if self._pending_ckpt is not None:
            ck = self._pending_ckpt
            if list(ck["hidden"]) != list(self._cfg["hidden"]):
                raise PolicyError(f"checkpoint hidden {ck['hidden']} != CONFIG hidden {self._cfg['hidden']}")
            self._net.load_state_dict(ck["net"])
            self._n_decisions = int(ck.get("decisions", 0))
            self._n_steps = int(ck.get("steps", 0))
        self._target = _build_net(torch, feature_dim, list(self._cfg["hidden"]))
        self._target.load_state_dict(self._net.state_dict())
        self._opt = torch.optim.Adam(self._net.parameters(), lr=self._cfg["lr"])
        if self._pending_ckpt is not None and "opt" in self._pending_ckpt:
            self._opt.load_state_dict(self._pending_ckpt["opt"])

    def _features(self, rv, views) -> np.ndarray:
        rows = [self._policy.replica_features(rv, v, views, self._ctx) for v in views]
        try:
            x = np.asarray(rows, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise PolicyError(f"replica_features returned non-numeric or ragged values: {e}") from e
        if x.ndim != 2 or x.shape[0] != len(views):
            raise PolicyError(f"replica_features must return one flat vector per replica, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise PolicyError("replica_features returned NaN or inf")
        if self._net is None:
            self._init_net(x.shape[1])
        elif x.shape[1] != self._feature_dim:
            raise PolicyError(f"replica_features length changed from {self._feature_dim} to {x.shape[1]}")
        return x

    def _q(self, net, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return net(self._torch.from_numpy(x)).squeeze(-1).numpy()

    def _scheduled(self, key: str) -> float:
        """teacher_mix / imitation_weight with the optional linear phase-out applied."""
        c = self._cfg
        start, end = c[key], c[key + "_end"]
        if end is None:
            return start
        frac = min(1.0, self._n_decisions / c["teacher_decay_decisions"])
        return start + frac * (end - start)

    def _epsilon(self) -> float:
        c = self._cfg
        frac = min(1.0, self._n_decisions / c["epsilon_decay_decisions"])
        return c["epsilon_start"] + frac * (c["epsilon_end"] - c["epsilon_start"])

    # ----------------------------------------------------------------- routing

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()
        burst = self._request_queue
        self._request_queue = []
        if not burst:
            return []
        from dataclasses import replace

        from vidur.scheduler.global_scheduler.evolved_global_scheduler import RequestView

        self._ctx.now = burst[0].arrived_at
        mapping = []
        for k, r in enumerate(burst):
            cached = self._cached_prefix_tokens(r)
            rv = RequestView(
                arrived_at=r.arrived_at,
                num_prefill_tokens=r.num_prefill_tokens,
                session_id=r.session_id,
                cached_prefix_tokens=cached,
                session_last_replica=self._session_last.get(r.session_id, -1) if r.session_id is not None else -1,
                position_in_burst=k,
                burst_size=len(burst),
            )
            views = self._replica_views()
            t0 = time.perf_counter()
            try:
                x = self._features(rv, views)
                mask = self._mask(rv, views)
                a, teacher_a = self._act(rv, views, x, mask)
            except PolicyError:
                raise
            except Exception as e:
                raise PolicyError(f"design raised {type(e).__name__}: {e}") from e
            finally:
                self.policy_calls += 1
                self.policy_seconds += time.perf_counter() - t0
            self._record(r, x, a, teacher_a, cached, views, mask)
            mapping.append(self._assign(r, a, cached))
        return mapping

    def _mask(self, rv, views) -> np.ndarray:
        n = len(views)
        if not callable(self._mask_fn):
            return np.ones(n, dtype=bool)
        m = self._mask_fn(rv, views, self._ctx)
        try:
            m = np.asarray([bool(v) for v in m], dtype=bool)
        except TypeError as e:
            raise PolicyError(f"candidate_mask must return a sequence of bools: {e}") from e
        if m.shape != (n,):
            raise PolicyError(f"candidate_mask must return {n} values, got {m.shape}")
        return m if m.any() else np.ones(n, dtype=bool)

    def _act(self, rv, views, x, mask) -> Tuple[int, int]:
        teacher_a = -1
        if self._teacher is not None and self._training:
            self._teacher_ctx.now = self._ctx.now
            teacher_a = self._check(self._teacher.choose(rv, views, self._teacher_ctx))
        q = np.where(mask, self._q(self._net, x), -np.inf)
        greedy = int(np.flatnonzero(q == q.max())[0]) if np.isfinite(q.max()) else int(np.flatnonzero(mask)[0])
        a = greedy
        if self._training:
            if teacher_a >= 0:
                self._stats["teacher_seen"] += 1
                self._stats["teacher_agree"] += int(teacher_a == greedy)
            u = self._rng.random()
            if teacher_a >= 0 and u < self._scheduled("teacher_mix"):
                a = teacher_a
                self._stats["teacher_acted"] += 1
            elif self._rng.random() < self._epsilon():
                allowed = np.flatnonzero(mask)
                a = int(allowed[self._rng.randrange(len(allowed))])
                self._stats["explore"] += 1
        self._n_decisions += 1
        self._stats["actions"][a] += 1
        return a, teacher_a

    def _record(self, r: Request, x, a, teacher_a, cached, views, mask) -> None:
        self._routed[r.id] = (r.arrived_at, a, cached[a])
        if not self._training:
            return
        d = _Decision(r.id, x, a, r.arrived_at, teacher_a, views, mask)
        if self._prev is not None:
            p = self._prev
            p.next_x = x
            p.next_mask = mask
            p.dt = max(0.0, r.arrived_at - p.t)
            if callable(self._interval_reward):
                p.r_interval = self._safe_reward(self._interval_reward, p.views, views, p.dt, self._ctx)
                self._stats["interval_reward"].append(p.r_interval)
            p.views = None  # no longer needed; keep the replay buffer small
            self._maybe_store(p)
        self._prev = d
        self._decisions[r.id] = d

    # ---------------------------------------------------------------- learning

    def on_request_end(self, request: Request) -> None:
        super().on_request_end(request)
        routed = self._routed.pop(request.id, None)
        if routed is None or not self._training:
            return
        d = self._decisions.get(request.id)
        if d is None:
            return
        ttft = request.prefill_completed_at - request.arrived_at
        e2e = request.completed_at - request.arrived_at
        n_dec = request.original_num_decode_tokens
        out = Outcome(ttft=ttft, e2e=e2e, tpot=(e2e - ttft) / (n_dec - 1) if n_dec > 1 else None,
                      num_prefill_tokens=request.original_num_prefill_tokens, num_decode_tokens=n_dec,
                      cached_tokens_at_routing=routed[2], replica=routed[1], routed_at=routed[0])
        if callable(self._outcome_reward):
            d.reward = self._safe_reward(self._outcome_reward, out)
            self._stats["reward"].append(d.reward)
        self._maybe_store(d)

    def _safe_reward(self, fn, *args) -> float:
        try:
            r = float(fn(*args))
        except Exception as e:
            raise PolicyError(f"{fn.__name__}() raised {type(e).__name__}: {e}") from e
        if not math.isfinite(r):
            raise PolicyError(f"{fn.__name__}() returned {r!r}")
        clip = self._cfg["reward_clip"]
        return max(-clip, min(clip, r))

    def _maybe_store(self, d: _Decision) -> None:
        if callable(self._outcome_reward) and d.reward is None:
            return  # outcome not known yet
        if callable(self._interval_reward) and d.r_interval is None and not d.done:
            return  # next decision not seen yet
        if self._cfg["algorithm"] == "dqn" and not d.done and d.next_x is None:
            return
        gamma = 0.0 if (self._cfg["algorithm"] == "bandit" or d.done) else math.exp(-self._cfg["beta"] * d.dt)
        nx = d.next_x if d.next_x is not None else d.x
        nm = d.next_mask if d.next_mask is not None else d.mask
        total = (d.reward or 0.0) + (d.r_interval or 0.0)
        item = (d.x, d.a, total, nx, gamma, d.teacher_a, nm)
        cap = self._cfg["buffer_size"]
        if len(self._buffer) < cap:
            self._buffer.append(item)
        else:
            self._buffer[self._buf_pos] = item
            self._buf_pos = (self._buf_pos + 1) % cap
        self._decisions.pop(d.rid, None)
        self._n_stored += 1
        if len(self._buffer) >= self._cfg["learning_starts"] and self._n_stored % self._cfg["train_every"] == 0:
            self._train_step()

    def _train_step(self) -> None:
        torch, c = self._torch, self._cfg
        t0 = time.perf_counter()
        idx = [self._rng.randrange(len(self._buffer)) for _ in range(c["batch_size"])]
        batch = [self._buffer[i] for i in idx]
        x = torch.from_numpy(np.stack([b[0] for b in batch]))  # [B, N, F]
        a = torch.tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        nx = torch.from_numpy(np.stack([b[3] for b in batch]))
        g = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        ta = torch.tensor([b[5] for b in batch], dtype=torch.long)
        nm = torch.from_numpy(np.stack([b[6] for b in batch]))  # [B, N] allowed at s'

        q_all = self._net(x).squeeze(-1)  # [B, N]
        q = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c["algorithm"] == "dqn":
                q_sel = self._net(nx).squeeze(-1).masked_fill(~nm, float("-inf"))
                a_star = q_sel.argmax(dim=1, keepdim=True)  # online net selects among allowed
                q_next = self._target(nx).squeeze(-1).gather(1, a_star).squeeze(1)  # target net evaluates
                y = r + g * q_next
            else:
                y = r
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        iw = self._scheduled("imitation_weight")
        if iw > 0:
            has = ta >= 0
            if bool(has.any()):
                qa, tav = q_all[has], ta[has]
                margin = torch.full_like(qa, c["imitation_margin"])
                margin.scatter_(1, tav.unsqueeze(1), 0.0)
                q_e = qa.gather(1, tav.unsqueeze(1)).squeeze(1)
                loss = loss + iw * ((qa + margin).max(dim=1).values - q_e).mean()
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), c["grad_clip"])
        self._opt.step()
        self._n_steps += 1
        if c["algorithm"] == "dqn" and self._n_steps % c["target_update"] == 0:
            self._target.load_state_dict(self._net.state_dict())
        if self._n_steps % 50 == 0:
            self._stats["loss"].append(float(loss.item()))
            self._stats["q"].append(float(q.mean().item()))
        self._stats["train_seconds"] += time.perf_counter() - t0

    # ---------------------------------------------------------------- end of run

    def _load_buffer(self, path: str) -> None:
        import os

        if not os.path.exists(path):
            return
        z = np.load(path)
        if z["x"].ndim != 3 or z["x"].shape[1] != len(self._replica_ids) or z["x"].shape[2] != self._feature_dim:
            return  # different replica count or feature size: start empty
        cap = self._cfg["buffer_size"]
        keep = slice(max(0, len(z["a"]) - cap), None)  # newest items if the capacity shrank
        self._buffer = [(x, int(a), float(r), nx, float(g), int(ta), nm) for x, a, r, nx, g, ta, nm in zip(
            z["x"][keep], z["a"][keep], z["r"][keep], z["nx"][keep], z["g"][keep], z["ta"][keep], z["nm"][keep])]
        self._buf_pos = int(z["pos"]) % cap if len(self._buffer) >= cap else 0
        self._stats_loaded_buffer = len(self._buffer)

    def _save_buffer(self, path: str) -> None:
        if not self._buffer:
            return
        b = self._buffer
        np.savez_compressed(path, x=np.stack([i[0] for i in b]), a=np.array([i[1] for i in b]),
                            r=np.array([i[2] for i in b], dtype=np.float32), nx=np.stack([i[3] for i in b]),
                            g=np.array([i[4] for i in b], dtype=np.float32), ta=np.array([i[5] for i in b]),
                            nm=np.stack([i[6] for i in b]), pos=np.array(self._buf_pos))

    def save_final_checkpoint(self) -> None:
        if self._training and self._prev is not None:
            self._prev.done = True  # last decision of the episode: no successor
            self._maybe_store(self._prev)
        self._stats["dropped_unfinished"] = len(self._decisions)
        if not self._checkpoint_out:
            return
        torch = self._torch
        if self._net is not None:
            torch.save({"net": self._net.state_dict(), "opt": self._opt.state_dict(),
                        "feature_dim": self._feature_dim, "hidden": list(self._cfg["hidden"]),
                        "decisions": self._n_decisions, "steps": self._n_steps}, self._checkpoint_out)
            if self._training and self._cfg["persist_buffer"]:
                self._save_buffer(self._checkpoint_out + ".buffer.npz")
        with open(self._checkpoint_out + ".stats.json", "w") as f:
            json.dump(self.diagnostics(), f)

    def diagnostics(self) -> dict:
        s = self._stats

        def summ(v):
            if not v:
                return None
            a = np.asarray(v, dtype=float)
            return {"mean": float(a.mean()), "p5": float(np.percentile(a, 5)),
                    "p95": float(np.percentile(a, 95)), "first": float(a[: max(1, len(a) // 10)].mean()),
                    "last": float(a[-max(1, len(a) // 10):].mean())}

        n = max(1, sum(s["actions"]))
        return {"mode": "training" if self._training else "inference", "decisions": self._n_decisions,
                "gradient_steps": self._n_steps, "epsilon_now": self._epsilon(),
                "explore_frac": s["explore"] / n, "teacher_acted_frac": s["teacher_acted"] / n,
                "teacher_mix_now": self._scheduled("teacher_mix"),
                "imitation_weight_now": self._scheduled("imitation_weight"),
                "teacher_agreement": (s["teacher_agree"] / s["teacher_seen"]) if s["teacher_seen"] else None,
                "action_share": [round(c / n, 4) for c in s["actions"]],
                "reward": summ(s["reward"]), "interval_reward": summ(s["interval_reward"]),
                "loss": summ(s["loss"]), "q_mean": summ(s["q"]),
                "train_seconds": s["train_seconds"], "policy_seconds": self.policy_seconds,
                "unfinished_decisions_dropped": s["dropped_unfinished"],
                "buffer_size_end": len(self._buffer),
                "buffer_loaded_from_previous_episode": getattr(self, "_stats_loaded_buffer", 0)}
