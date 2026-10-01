"""
EXP12: evolve the *design* of an RL router (features, reward, learning settings) with the
same loop that evolves router code (evolve/driver.py --task rl).

A candidate is a design module for the LEARNED global scheduler
(vidur/scheduler/global_scheduler/learned_global_scheduler.py). The harness, not the
candidate, owns the learner, the training protocol and the score:

  train   TRAIN_EPISODES in order, each a fresh simulation that starts from the previous
          episode's network (8R windows of the Tool and Conv traces at scale 0.75/1.0,
          perturbations 4-7: arrival seeds never used by any evaluation scenario). The
          frozen evolved finalist is available as a teacher; the design decides whether to
          use it (CONFIG teacher_mix / imitation_weight, both 0 by default).
  score   greedy inference with the trained network on evaluate.STAGE1 (full traces,
          perturbation 1), scored exactly like an evolved router: P99 E2E/TTFT/TPOT
          relative to the per-scenario best heuristic (same fitness, same baselines).
  hold-out  frozen finalists only: evaluate.HELDOUT (perturbations 2-3, the Synth trace
          never seen in training, 16 replicas).

Training compute is part of the cost and is reported (train_wall_s).
"""

import json
import tempfile
from pathlib import Path
from typing import List, Optional

from evolve import evaluate as ev

REPO = Path(__file__).resolve().parents[1]
TEACHER = REPO / "evolve" / "finalists" / "claude_r1_76be082eb0f5.py"
SEED_DESIGNS = ["seed_ttft_bandit", "seed_dqn_teacher", "seed_dqn_interval"]

S = ev.Scenario
TRAIN_EPISODES = [
    S("Tool", 8, 0.75, 2000, 4000, 4), S("Conv", 8, 0.75, 1000, 4000, 4),
    S("Tool", 8, 1.0, 10000, 4000, 5), S("Conv", 8, 1.0, 6000, 4000, 5),
    S("Tool", 8, 0.75, 16000, 4000, 6), S("Conv", 8, 0.75, 7000, 4000, 6),
    S("Tool", 8, 1.0, 5000, 4000, 7), S("Conv", 8, 1.0, 3000, 4000, 7),
]
CRASH_EPISODE = S("Tool", 4, 1.5, 12000, 150, 1)
# Plumbing check only (driver --scenarios smoke): two tiny training episodes.
SMOKE_EPISODES = [S("Tool", 4, 1.5, 12000, 300, 4), S("Conv", 4, 2.0, 6000, 300, 4)]


def learned_args(mode: str, ckpt_in: Optional[Path] = None, ckpt_out: Optional[Path] = None,
                 teacher: Optional[Path] = TEACHER) -> List[str]:
    args = ["--learned_global_scheduler_config_mode", mode]
    if ckpt_in:
        args += ["--learned_global_scheduler_config_checkpoint_in", str(ckpt_in)]
    if ckpt_out:
        args += ["--learned_global_scheduler_config_checkpoint_out", str(ckpt_out)]
    if teacher and mode == "training":
        args += ["--learned_global_scheduler_config_teacher_path", str(teacher)]
    return args


def crash_check(design: str, launcher=None) -> dict:
    """Train one tiny episode and run it greedily; catches contract errors in seconds."""
    with tempfile.TemporaryDirectory(prefix="rl_crash_") as tmp:
        ck = Path(tmp) / "net.pt"
        r = ev.run_one("learned", CRASH_EPISODE, policy=design, timeout=900, launcher=launcher,
                       extra=learned_args("training", ckpt_out=ck))
        if not r.get("ok"):
            return r
        if not ck.exists():  # a design whose features were never called cannot be scored
            return {"ok": False, "reason": "no network was created (replica_features never called?)"}
        return ev.run_one("learned", CRASH_EPISODE, policy=design, timeout=900, launcher=launcher,
                          extra=learned_args("inference", ckpt_in=ck))


def train(design: str, work_dir: Path, episodes=None, launcher=None, timeout=3600) -> dict:
    """Run the training episodes in sequence; return per-episode tails and learner diagnostics."""
    episodes = episodes or TRAIN_EPISODES
    work_dir.mkdir(parents=True, exist_ok=True)
    prev, log = None, []
    for k, sc in enumerate(episodes):
        ck = work_dir / f"net_ep{k}.pt"
        r = ev.run_one("learned", sc, policy=design, timeout=timeout, launcher=launcher,
                       extra=learned_args("training", ckpt_in=prev, ckpt_out=ck))
        if not r.get("ok"):
            return {"ok": False, "reason": f"training episode {k} ({sc.tag}): {r.get('reason')}",
                    "stderr": r.get("stderr", ""), "episodes": log}
        stats = json.loads((Path(str(ck) + ".stats.json")).read_text())
        log.append({"episode": sc.tag, "p99_e2e": r["p99_e2e"], "p99_ttft": r["p99_ttft"],
                    "p99_tpot": r["p99_tpot"], "mean_e2e": r["mean_e2e"], "wall_s": r["wall_s"],
                    "learner": stats})
        prev = ck
    return {"ok": True, "checkpoint": str(prev), "episodes": log,
            "train_wall_s": sum(e["wall_s"] for e in log)}


def evaluate_design(design: str, work_dir: Path, scenarios=None, workers=4, launcher=None,
                    timeout=3600, episodes=None) -> dict:
    scenarios = scenarios or ev.STAGE1
    tr = train(design, work_dir, episodes=episodes, launcher=launcher, timeout=timeout)
    if not tr["ok"]:
        return {"combined_score": -1e9, "valid": False, "failures": {"training": tr["reason"]},
                "training": tr}
    base = ev.ensure_baselines(scenarios, workers=workers, launcher=launcher, timeout=timeout)
    extra = learned_args("inference", ckpt_in=Path(tr["checkpoint"]))
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ev.run_one, "learned", sc, design, timeout, launcher, extra) for sc in scenarios]
        results = [f.result() for f in futs]
    failures = {sc.tag: r for sc, r in zip(scenarios, results) if not r.get("ok")}
    if failures:
        return {"combined_score": -1e9, "valid": False, "failures": failures, "training": tr}
    out = ev.score(dict(zip(scenarios, results)), base)
    out.update({"valid": True, "code_version": ev.code_version(), "training": tr,
                "eval_wall_s": sum(r["wall_s"] for r in results),
                "scenarios": [sc.tag for sc in scenarios]})
    return out


def feedback_text(res: dict) -> str:
    """Score feedback plus 'reward reflection' (Eureka): what the learner did in training."""
    lines = [ev.feedback_text(res)] if res.get("valid") else [f"INVALID: {res.get('failures')}"]
    tr = res.get("training") or {}
    eps = tr.get("episodes") or []
    if eps:
        lines.append("Training episodes (P99 E2E / TTFT / TPOT in s; learner diagnostics):")
        for e in eps:
            L = e["learner"]
            rw = L.get("reward") or L.get("interval_reward") or {}
            ls, q = L.get("loss") or {}, L.get("q_mean") or {}
            lines.append(
                f"  {e['episode']}: {e['p99_e2e']:.1f} / {e['p99_ttft']:.1f} / {e['p99_tpot']:.3f}; "
                f"steps={L['gradient_steps']} eps={L['epsilon_now']:.3f} explore={L['explore_frac']:.3f} "
                f"teacher_acted={L['teacher_acted_frac']:.3f} agree={L['teacher_agreement']} "
                f"reward mean={rw.get('mean')} first/last={rw.get('first')}/{rw.get('last')} "
                f"loss first/last={ls.get('first')}/{ls.get('last')} Q first/last={q.get('first')}/{q.get('last')} "
                f"action_share={L['action_share']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a systems and reinforcement-learning researcher designing an RL-trained request
router for an LLM inference cluster. You write a DESIGN MODULE that defines what the router
learns from and how; a fixed harness trains it in a simulator and scores it.

## Setting
- Vidur discrete-event simulator; Llama-2-7b replicas (A100, vLLM-v1 scheduler with chunked
  prefill and prefix caching), 122,752 KV-cache tokens per replica.
- Mooncake production traces (conversation and tool/agent). Requests arrive in bursts;
  prompts are long (median ~7k tokens) and share prefixes, so prefix-cache hits matter.
- The router assigns each arriving request to one replica immediately; replicas run FCFS.

## Objective (fixed by the harness)
Minimise TAIL latency: P99 end-to-end, P99 time-to-first-token and P99 time-per-output-token
per scenario, relative to the best of five heuristic routers (sticky-LOR, LOR, round-robin,
random, least-uncached-prefill). Mean E2E more than 5% worse than the best heuristic is
penalised. Fitness 0 = matches the best heuristic on every tail; positive = better.
A hand-designed (evolved) router already reaches about +0.16; its key idea is that a cache
miss is expensive even at light load because its prefill chunks steal steps from requests
that are already decoding, so it prices uncached tokens higher when decodes are resident,
plus convex guards against load imbalance, decode-batch growth and KV pressure.

## The learner (fixed; you configure it)
One MLP with shared weights scores every replica: Q(s, a) = f(replica_features(request, a, ...)).
The router picks argmax Q (epsilon-greedy while training). Algorithms (CONFIG["algorithm"]):
- "bandit": regress Q(s, a) on the decision's own reward (no bootstrapping).
- "dqn": Double DQN on a semi-MDP. Successor state = state at the next routing decision;
  discount exp(-beta * dt) with dt the simulated seconds between decisions; target network
  copied every target_update gradient steps; Huber loss.
A teacher (the evolved router above) is available during training only:
  teacher_mix = probability the teacher's action is executed instead of the learner's;
  imitation_weight / imitation_margin = DQfD large-margin loss towards the teacher's action.
At evaluation the learner acts alone and greedily.
Training protocol: 8 episodes in order (4,000-request windows of the Tool and Conv traces,
8 replicas, loads 0.75 and 1.0, arrival seeds never used for scoring), each continuing from
the previous network (optimizer state kept, replay buffer starts empty each episode).
Scoring: full Tool and Conv traces, 8 replicas, loads 1.0 and 0.75 (~24k and ~12k requests).
Episodes are short relative to the full traces: settings must learn within ~32k decisions.

## Contract (strict)
Define at module level:
    def replica_features(request, replica, replicas, ctx) -> list of floats
        Same length on every call (1..64). Keep values O(1) (scale them).
    def reward(outcome) -> float                                          (optional*)
        Reward of the decision that routed this request, known when it completes.
    def interval_reward(replicas_before, replicas_after, dt, ctx) -> float (optional*)
        Reward accrued between this decision and the next (dt simulated seconds), e.g.
        -dt * requests-in-system (Bradtke & Duff interval cost; sums to total latency by
        Little's law). Known at the next decision; charges a decision for slowing OTHER
        requests through the bootstrap. Not available to "bandit" credit beyond one step.
    * define at least one; if both, they add. |reward| is clipped at reward_clip.
    CONFIG = {...}   # optional; keys and allowed ranges below
Views are frozen dataclasses:
  request: arrived_at, num_prefill_tokens, session_id, cached_prefix_tokens (tuple, per replica),
           session_last_replica (-1 if new session), position_in_burst, burst_size
  replica / replicas[i]: index, outstanding_requests, pending_prefill_tokens (uncached prompt
           tokens routed here, not yet prefilled), waiting, running, kv_usage (0..1), kv_capacity_tokens
  ctx: rng, now (simulated s), num_replicas, state (dict for memory across calls)
  outcome (in reward): ttft, e2e, tpot (None if one output token), num_prefill_tokens,
           num_decode_tokens, cached_tokens_at_routing, replica, routed_at
Output lengths are NOT visible when routing. Do not make features depend on replica.index
itself (weights are shared across replicas; the index carries no meaning).
CONFIG keys (defaults): algorithm "bandit"|"dqn"; beta 0.05 [0,10]; hidden [64, 64] (1-4 layers,
each 1..512); lr 1e-3 [1e-5,0.1]; batch_size 64 [8,1024]; buffer_size 50000 [1e3,5e5];
train_every 4 [1,64]; learning_starts 500; target_update 500; epsilon_start 0.2, epsilon_end 0.02,
epsilon_decay_decisions 20000; teacher_mix 0.0 [0,1]; imitation_weight 0.0 [0,100];
imitation_margin 0.5; grad_clip 10; reward_clip 100.
Allowed imports: math, heapq, collections, itertools, functools, statistics, bisect.
No file, clock, network, os/sys access, no global/nonlocal, no dunder attributes,
no getattr/eval/exec. Keep per-call work O(replicas).

## Answer format
Read the parent's score AND its training diagnostics (reward and loss trends, Q values,
exploration, teacher agreement, action shares across replicas). Form one concrete hypothesis
about why it learns what it learns, and change the design to test it. Reply with a short
paragraph stating the hypothesis, then the COMPLETE new design module in one ```python block.
"""
