"""Seed RL design (protocol v3): the best RL design of search rl_claude_r1 (2e9c577070d6,
stage-1 +0.153, held-out +0.019) with the teacher PHASED OUT: teacher mixing 0.30 -> 0 and
the DQfD imitation weight 1.25 -> 0 over the first 30k training decisions, so the learner
starts from the evolved router's behaviour and must then improve on it alone.
Original docstring follows.
"""
"""Double DQN on the semi-MDP. Same family as the +0.127 parent (own-deadline reward with a
non-saturating tail, deterministic cache-miss price, teacher's interference mechanism as
features), with ONE mechanism changed: the balance of the reward along the TTFT<->TPOT
frontier, plus the projected-KV (preemption) terms needed to express the new cost.

Diagnosis of the parent (+0.1270):
  * E2E 0.88-0.92 and TTFT 0.51-0.86 of the best heuristic -- a large surplus -- while TPOT
    loses 1.13 / 1.23 / 0.97 / 1.17. The TPOT loss is largest exactly where the TTFT win is
    largest (Tool 0.75), i.e. it is the PRICE of the TTFT win, not an independent failure:
    clearing queues fast turns waiting requests into resident decodes, so decode batches and
    KV occupancy grow on the replicas that win, and chunked prefill then lands on top of them.
  * The reward made that choice rational: 0.45*_dl(ttft)+0.35*_dl(e2e) has range ~1.3, while
    the only decode term was 0.20*hinge(tpot) clipped at 3 -> max 0.6, and KV pressure
    (preemption, the only mechanism that can lift a per-request AVERAGE inter-token time by
    25%) was priced nowhere at all. Agreement with the +0.16 teacher is 0.87-0.89; the
    remaining disagreements are all in the TTFT-greedy direction.

Changes (all one hypothesis):
  * TTFT weight 0.45 -> 0.38, E2E 0.35 -> 0.32 (trimmed, not crushed: the TTFT surplus is
    where most of the current fitness comes from).
  * own-TPOT: weight 0.20 -> 0.50 with a hinge at 1.1x target and a NON-SATURATING
    0.5u + 0.9*log1p(u) tail (cap 4). A 2x-target TPOT now costs ~0.5, a preemption-scale 4x
    costs ~1.3 -- comparable to a badly late first token. Because every later decision also
    pays its own TPOT, my prefill work shows up in the bootstrapped return of the decisions
    before it (beta=0.05 ~ 65 decisions), so the externality has a path to the value function
    without an interval cost (which has scored -1.0 to -14.9 in this family).
  * features: post-routing KV projection kv_usage + uncached/capacity, its convex excess over
    0.80, its deviation from the cluster mean, decode batch in units of KV-limited concurrency
    for this prompt, and lg_post*kv_usage. The parent could see current KV but not the KV the
    ACTION would commit, so preemption risk was not representable.
  * BASELINE 0.55 -> 0.40 to keep mean reward near zero now that a penalty was added.
Held fixed deliberately: beta 0.05, dqn, teacher_mix 0.30, imitation_weight 1.25, no
candidate_mask (hard masks scored -0.22 / -0.42 here), the deterministic miss price, net size.
"""

import math

TOKENS = 8192.0        # ~median prompt: feature/token scale
DEC_REF = 8.0          # reference decode-batch size

TTFT_TARGET = 10.0     # s
E2E_TARGET = 40.0      # s
TPOT_TARGET = 0.045    # s
TPOT_SLACK = 1.1       # hinge starts here (x target)
BASELINE = 0.40        # centres reward near 0 at light load

W_TTFT = 0.38
W_E2E = 0.32
W_TPOT = 0.50          # was 0.20: the frontier re-balance under test
W_MISS_LOG = 0.20
W_MISS_FRAC = 0.08

KV_SAFE = 0.75         # current-KV knee
KV_PROJ_SAFE = 0.80    # post-routing projected-KV knee

CONFIG = {
    "algorithm": "dqn",
    "beta": 0.05,                  # unchanged: only horizon that has scored > 0 here
    "hidden": [96, 64],
    "lr": 3e-4,
    "batch_size": 128,
    "buffer_size": 100000,
    "train_every": 2,
    "learning_starts": 500,
    "target_update": 500,
    "epsilon_start": 0.10,
    "epsilon_end": 0.01,
    "epsilon_decay_decisions": 18000,
    "teacher_mix": 0.30,
    "imitation_weight": 1.25,
    "teacher_mix_end": 0.0,
    "imitation_weight_end": 0.0,
    "teacher_decay_decisions": 30000,
    "imitation_margin": 0.5,
    "grad_clip": 10,
    "reward_clip": 10,
}


def _lg(tokens):
    """Log token scale: knee backlogs (1e5+ tokens) stay below ~2.6."""
    return math.log1p(max(tokens, 0.0) / TOKENS)


# --------------------------------------------------------------------------- features

def replica_features(request, replica, replicas, ctx):
    k = len(replicas)
    i = replica.index
    n = float(request.num_prefill_tokens)
    cached = request.cached_prefix_tokens

    unc = max(0.0, n - float(cached[i]))
    post = float(replica.pending_prefill_tokens) + unc
    cap = max(float(replica.kv_capacity_tokens), 1.0)

    # one O(k) pass over the cluster
    sum_pend = 0.0
    sum_run = 0.0
    sum_kv = 0.0
    sum_lg_post = 0.0
    min_lg_post = 1e18
    min_unc = n
    max_pend = 0.0
    for r in replicas:
        p = float(r.pending_prefill_tokens)
        sum_pend += p
        if p > max_pend:
            max_pend = p
        u = max(0.0, n - float(cached[r.index]))
        if u < min_unc:
            min_unc = u
        lp = _lg(p + u)
        sum_lg_post += lp
        if lp < min_lg_post:
            min_lg_post = lp
        sum_run += float(r.running)
        sum_kv += r.kv_usage

    mean_pend = sum_pend / k
    mean_run = sum_run / k
    mean_kv = sum_kv / k
    mean_lg_post = sum_lg_post / k

    lg_pend = _lg(replica.pending_prefill_tokens)
    lg_mean_pend = _lg(mean_pend)
    lg_unc = _lg(unc)
    lg_post = _lg(post)

    run = float(replica.running) / DEC_REF
    run_dev = (float(replica.running) - mean_run) / DEC_REF
    run_exc = max(0.0, run_dev)
    pend_exc = max(0.0, lg_pend - lg_mean_pend)
    post_exc = max(0.0, lg_post - mean_lg_post)

    kv_over = max(0.0, replica.kv_usage - KV_SAFE) / (1.0 - KV_SAFE)

    # --- what the ACTION would commit to KV: preemption of resident decodes is the
    #     mechanism that inflates a per-request AVERAGE inter-token time ---
    kv_proj = replica.kv_usage + unc / cap
    kv_proj_exc = max(0.0, kv_proj - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE)
    kv_proj_exc = min(3.0, kv_proj_exc)
    # decode batch measured in units of how many prompts of THIS size KV can hold
    run_per_cap = min(4.0, float(replica.running) * n / cap)

    burst = max(float(request.burst_size), 1.0)

    return [
        # --- standing prefill backlog, absolute / relative / convex guard ---
        lg_pend,
        lg_pend - lg_mean_pend,
        pend_exc * pend_exc,
        math.log1p(max(replica.waiting, 0)) / 3.0,
        math.log1p(max(replica.outstanding_requests, 0)) / 4.0,
        # --- this request's cache state on this replica ---
        lg_unc,
        lg_unc - _lg(min_unc),
        unc / max(n, 1.0),
        n / TOKENS,
        1.0 if request.session_last_replica == i else 0.0,
        1.0 if request.session_last_replica < 0 else 0.0,
        # --- post-routing uncached backlog (least-uncached-prefill variable) ---
        lg_post,
        lg_post - mean_lg_post,
        lg_post - min_lg_post,
        post_exc * post_exc,
        # --- decode occupancy: what prefill chunks steal steps from ---
        run,
        run_dev,
        run_exc * run_exc,
        # --- KV pressure now, and PROJECTED after this action (preemption risk) ---
        replica.kv_usage,
        replica.kv_usage - mean_kv,
        kv_over * kv_over,
        kv_proj,
        kv_proj_exc * kv_proj_exc,
        kv_proj - mean_kv,
        run_per_cap,
        # --- interference: a miss costs more when decodes are resident / KV is tight ---
        lg_unc * run,
        lg_unc * (mean_run / DEC_REF),
        lg_post * run,
        lg_post * replica.kv_usage,
        # --- cluster context so the hidden layers can rescale the above ---
        lg_mean_pend,
        _lg(max_pend) - lg_mean_pend,
        mean_run / DEC_REF,
        mean_kv,
        # --- burst shape: anticipate the pile-up rather than react to it ---
        float(request.position_in_burst) / burst,
        math.log1p(burst) / 4.0,
    ]


# --------------------------------------------------------------------------- reward

def _dl(latency, target):
    """Deadline score: +1 at 0, 0 at target, -1 at 2x target, then -0.6*log(z-1).

    Bounded growth (no unbounded-target poisoning) but strictly decreasing out to 100 s,
    so the deep-backlog region that sets the scored P99s keeps an action-dependent gradient.
    """
    z = max(latency, 0.0) / target
    if z <= 2.0:
        return 1.0 - z
    return -1.0 - 0.6 * math.log(z - 1.0)


def _tp(tpot):
    """Tail-focused, non-saturating decode-smoothness penalty.

    Zero below 1.1x target (most requests contribute nothing, as for a tail objective),
    then 0.5u + 0.9*log1p(u): ~0.50 at 2x target, ~1.3 at 4x (preemption / recompute
    scale), ~2.0 at 7x, capped at 4. The parent clipped this at 0.6 total, which made a
    blown TPOT cheaper than a late first token; it is now comparable.
    """
    u = max(0.0, tpot / TPOT_TARGET - TPOT_SLACK)
    if u <= 0.0:
        return 0.0
    return min(4.0, 0.5 * u + 0.9 * math.log1p(u))


def reward(outcome):
    # own queueing / completion deadlines: still the dominant signal, trimmed to pay for
    # the decode term (the scored TTFT/E2E surplus is 10-49%, TPOT is the only deficit)
    r = W_TTFT * _dl(outcome.ttft, TTFT_TARGET) + W_E2E * _dl(outcome.e2e, E2E_TARGET)

    # decode smoothness on the chosen replica: interference from co-resident prefill chunks
    # and KV preemption. Partly caused by LATER decisions, so with beta=0.05 this term also
    # charges earlier decisions for the prefill work they left behind, via the bootstrap.
    if outcome.tpot is not None:
        r -= W_TPOT * _tp(outcome.tpot)

    # deterministic price of the uncached prefill work this decision committed to the
    # cluster (the teacher's mechanism): zero variance, exactly attributed, differs across
    # replicas, so it produces a real learned advantage rather than noise.
    n = max(float(outcome.num_prefill_tokens), 1.0)
    miss = max(0.0, float(outcome.num_prefill_tokens) -
               float(outcome.cached_tokens_at_routing))
    r -= W_MISS_LOG * _lg(miss) + W_MISS_FRAC * (miss / n)

    # constant centring: mean reward ~ 0 at light load, so |Q| stays small and action
    # advantages are not swamped by a common offset. Policy-neutral (dt is exogenous).
    return r - BASELINE
