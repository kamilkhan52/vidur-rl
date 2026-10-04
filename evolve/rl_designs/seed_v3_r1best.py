"""RL design: Double DQN on the semi-MDP, re-conditioned return + anti-forgetting replay
+ explicit load-regime conditioning.

Diagnosis of the parent (-2.9943)
  * Near-uniform action shares, flat loss, Q drifting 0.56 -> 29.5: with beta=0.05 and ~25 ms
    between decisions the return is a huge state-independent offset built from OTHER requests'
    rewards, so the inter-replica advantage that argmax consumes is a sub-percent residual.
  * The reward saturated at -1 beyond 2x target, i.e. the whole scored regime (6-68 s TTFT,
    600 s on the ramp) sat in the zero-gradient region (training mean reward literally hit -1.0).
  * Teacher acted on 30 % of decisions with NO decay, so the acceptable training curves were
    partly the teacher's; alone, the net collapsed.
  * NEW here: training is sequential and the last three episodes are Tool, while the buffer
    (50 k) barely covers 60 k decisions and only ~15 k gradient steps are taken. The finalist is
    therefore a Tool-specialised net; the Synth mechanisms (prefix packing, in-flight prefix
    credit) are forgotten -- which is precisely where it failed at scoring (Synth 2.25x,
    Synth-ramp 8.89x) even though its own Synth-ramp TRAINING episode matched the best
    heuristic (76 s).

Changes under test
  1. Anti-forgetting replay: buffer_size 500 k (never evicts over ~60 k decisions),
     batch_size 256, train_every 2 (~30 k gradient steps, ~130 epochs over the whole history),
     lr 2.5e-4, target_update 1000. Every gradient step mixes all ten episodes, so the late
     Tool episodes cannot overwrite the Synth rule.
  2. Load-regime conditioning: EWMA arrival rate and token rate on two time constants (2 s,
     30 s), exposed as a utilisation level and a short/long TREND, plus interactions with the
     uncached-token cost. The measured mechanism "a cache miss must be priced higher as load
     rises" is thereby conditioned on where load IS GOING, not only on instantaneous backlog --
     the thing a fixed constant cannot do on a ramp.
  3. Re-conditioned credit (kept from the best known design): tail-only LOG rewards, exactly 0
     below threshold and unbounded above, so r_bar ~ 0 in healthy states (no Q drift) and the
     gradient survives at 600 s; beta=0.5 (~2 s horizon, tens of decisions) so values actually
     propagate; convex interval cost on backlog IMBALANCE and decode-batch imbalance (the TPOT
     externality caused by LATER decisions) plus a small convex LEVEL term so total cluster work
     is not free.
  4. Teacher truly phased out (mix 0.4 -> 0, imitation 1.0 -> 0 by 26 k decisions) with
     epsilon_end 0.03: the hard half of training runs on the learner's own distribution.
"""

import math

TOKENS = 8192.0            # ~median prompt: token feature scale
DEC_REF = 8.0              # reference decode-batch size
PREFILL_RATE = 6000.0      # rough tokens/s per replica, only a scale for wait estimates
INFLIGHT_WINDOW_S = 60.0   # a previous turn older than this is irrelevant
TAU_S = 2.0                # short arrival-rate EWMA time constant (s)
TAU_L = 30.0               # long arrival-rate EWMA time constant (s)
BETA = 0.5                 # must match CONFIG["beta"]

# --- tail thresholds: at/below these a request is "fine" and contributes exactly 0.
#     Scored P99s live at 6-68 s TTFT / 16-76 s E2E / 0.047-0.085 s TPOT, far above.
TTFT_T = 1.5
E2E_T = 15.0
TPOT_T = 0.035

W_TTFT = 1.00              # queueing: 38-94 % of a tail request's time
W_E2E = 0.30               # small: E2E is mostly output length, i.e. routing noise
W_TPOT = 1.60              # the systematic deficit in every scored scenario (1.04-1.28x)
W_MISS = 0.15              # deterministic, zero-variance price of committed uncached work

W_INT_PREFILL = 0.90       # convex imbalance of prefill backlog (tails come from the hot spot)
W_INT_DECODE = 0.60        # convex imbalance of decode batch (TPOT externality)
W_INT_LEVEL = 0.15         # small convex term on the LEVEL: total backlog is not free

KV_SAFE = 0.75
KV_PROJ_SAFE = 0.85

CONFIG = {
    "algorithm": "dqn",
    "beta": BETA,
    "hidden": [128, 96],
    "lr": 2.5e-4,
    "batch_size": 256,
    "buffer_size": 500000,            # non-evicting over the whole training run
    "train_every": 2,                 # ~30k gradient steps: replay all episodes many times
    "learning_starts": 1500,
    "target_update": 1000,
    "epsilon_start": 0.15,
    "epsilon_end": 0.03,              # keep exploring once the teacher is gone
    "epsilon_decay_decisions": 30000,
    "teacher_mix": 0.4,
    "teacher_mix_end": 0.0,
    "imitation_weight": 1.0,
    "imitation_weight_end": 0.0,
    "teacher_decay_decisions": 26000,
    "imitation_margin": 0.5,
    "grad_clip": 10,
    "reward_clip": 10,
}


def _lg(tokens):
    """Log token scale: knee backlogs (1e5+ tokens) stay below ~2.6."""
    return math.log1p(max(tokens, 0.0) / TOKENS)


# --------------------------------------------------------------------------- memory

def _decision_memory(request, ctx):
    """Update per-decision memory ONCE (first replica probe), return ctx.state.

    Holds (a) the session's previous turn, so the in-flight prefix credit is representable,
    and (b) two-timescale EWMAs of the arrival and token rate, so the policy can see the
    load REGIME and its trend instead of inferring it from instantaneous backlog.
    O(1) per probe, O(1) per decision.
    """
    state = ctx.state
    key = (request.session_id, request.arrived_at, request.num_prefill_tokens)
    if state.get("last_key") == key:
        return state
    state["last_key"] = key

    now = ctx.now
    last_t = state.get("last_t")
    if last_t is not None and now > last_t:
        gap = now - last_t
        if gap < 1e-4:
            gap = 1e-4
        inst_r = 1.0 / gap
        inst_t = float(request.num_prefill_tokens) / gap
        ws = 1.0 - math.exp(-gap / TAU_S)
        wl = 1.0 - math.exp(-gap / TAU_L)
        lam_s = state.get("lam_s")
        lam_l = state.get("lam_l")
        tok_s = state.get("tok_s")
        tok_l = state.get("tok_l")
        state["lam_s"] = inst_r if lam_s is None else lam_s + ws * (inst_r - lam_s)
        state["lam_l"] = inst_r if lam_l is None else lam_l + wl * (inst_r - lam_l)
        state["tok_s"] = inst_t if tok_s is None else tok_s + ws * (inst_t - tok_s)
        state["tok_l"] = inst_t if tok_l is None else tok_l + wl * (inst_t - tok_l)
    state["last_t"] = now

    table = state.get("turns")
    if table is None:
        table = {}
        state["turns"] = table
    sid = request.session_id
    prev = table.get(sid) if sid is not None else None
    state["prev"] = prev
    if sid is not None:
        table[sid] = (request.arrived_at, request.num_prefill_tokens)
        if len(table) > 20000:
            cutoff = request.arrived_at - INFLIGHT_WINDOW_S
            state["turns"] = dict((k, v) for k, v in table.items() if v[0] > cutoff)
    return state


# --------------------------------------------------------------------------- features

def replica_features(request, replica, replicas, ctx):
    state = _decision_memory(request, ctx)
    k = len(replicas)
    i = replica.index
    n = float(request.num_prefill_tokens)
    n_safe = max(n, 1.0)
    cached = request.cached_prefix_tokens
    home_idx = request.session_last_replica

    # ---- in-flight prefix credit ---------------------------------------------------------
    # the session's previous turn may still be QUEUED on its replica, so cached_prefix_tokens
    # does not show it yet although it will be cached there by the time we prefill.
    credit = 0.0
    prev_age = 1.0
    has_prev = 0.0
    prev = state.get("prev")
    if prev is not None:
        age = request.arrived_at - prev[0]
        if age <= INFLIGHT_WINDOW_S:
            has_prev = 1.0
            prev_age = max(0.0, age) / INFLIGHT_WINDOW_S
            if home_idx >= 0:
                credit = min(float(prev[1]), n)

    # ---- one O(k) pass over the cluster --------------------------------------------------
    sum_pend = 0.0
    max_pend = 0.0
    sum_run = 0.0
    max_run = 0.0
    sum_kv = 0.0
    sum_lwait = 0.0
    sum_lg_post = 0.0
    min_lg_post = 1e18
    min_unc = n
    min_wself = 1e18
    for r in replicas:
        j = r.index
        p = float(r.pending_prefill_tokens)
        sum_pend += p
        if p > max_pend:
            max_pend = p
        c = float(cached[j])
        if j == home_idx and credit > c and (r.waiting > 0 or p > 0.0):
            c = credit
        u = max(0.0, n - c)
        if u < min_unc:
            min_unc = u
        lp = _lg(p + u)
        sum_lg_post += lp
        if lp < min_lg_post:
            min_lg_post = lp
        w = (p + u) / PREFILL_RATE
        if w < min_wself:
            min_wself = w
        rr = float(r.running)
        sum_run += rr
        if rr > max_run:
            max_run = rr
        sum_kv += r.kv_usage
        sum_lwait += math.log1p(max(r.waiting, 0)) / 3.0

    mean_pend = sum_pend / k
    mean_run = sum_run / k
    mean_kv = sum_kv / k
    mean_lwait = sum_lwait / k
    mean_lg_post = sum_lg_post / k
    lg_mean_pend = _lg(mean_pend)

    # ---- load regime (level and trend) ---------------------------------------------------
    lam_s = state.get("lam_s")
    lam_l = state.get("lam_l")
    tok_s = state.get("tok_s")
    if lam_s is None:
        lam_s = 0.0
    if lam_l is None:
        lam_l = lam_s
    if tok_s is None:
        tok_s = 0.0
    util = math.tanh(tok_s / (k * PREFILL_RATE))
    trend = math.tanh(math.log((lam_s + 0.1) / (lam_l + 0.1)))
    rising = max(0.0, trend)

    # ---- this action ---------------------------------------------------------------------
    is_home = 1.0 if home_idx == i else 0.0
    c_here = float(cached[i])
    c_eff = c_here
    use_credit = 0.0
    if home_idx == i and credit > c_here and (
            replica.waiting > 0 or replica.pending_prefill_tokens > 0):
        use_credit = (credit - c_here) / n_safe
        c_eff = credit
    unc_raw = max(0.0, n - c_here)
    unc = max(0.0, n - c_eff)

    cap = max(float(replica.kv_capacity_tokens), 1.0)
    pend = float(replica.pending_prefill_tokens)
    post = pend + unc

    lg_pend = _lg(pend)
    lg_post = _lg(post)
    lg_unc = _lg(unc)

    run_n = float(replica.running) / DEC_REF
    run_dev = (float(replica.running) - mean_run) / DEC_REF
    pend_dev = lg_pend - lg_mean_pend
    post_dev = lg_post - mean_lg_post
    lwait = math.log1p(max(replica.waiting, 0)) / 3.0

    kv_over = max(0.0, replica.kv_usage - KV_SAFE) / (1.0 - KV_SAFE)
    kv_proj = replica.kv_usage + unc / cap
    kv_proj_exc = min(3.0, max(0.0, kv_proj - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE))

    congestion = min(mean_pend / n_safe, 16.0) / 16.0
    burst = max(float(request.burst_size), 1.0)

    w_self = post / PREFILL_RATE
    lw_self = math.log1p(w_self)
    ext = math.log1p((unc / PREFILL_RATE) * (1.0 + max(replica.waiting, 0)))

    return [
        # --- request / cluster / REGIME context (identical across actions: lets the hidden
        #     layers rescale every per-replica term by the regime, which is exactly where the
        #     heuristics' fixed constants break) ---
        min(n / TOKENS, 4.0),
        1.0 if home_idx < 0 else 0.0,
        float(request.position_in_burst) / burst,
        math.log1p(burst) / 4.0,
        lg_mean_pend,
        _lg(max_pend) - lg_mean_pend,
        mean_run / DEC_REF,
        (max_run - mean_run) / DEC_REF,
        mean_kv,
        congestion,
        mean_lwait,
        util,                                   # offered token rate / cluster prefill capacity
        trend,                                  # load RISING (+) or falling (-)
        has_prev,
        prev_age,
        # --- standing prefill backlog on this replica: level, relative, convex guard ---
        lg_pend,
        pend_dev,
        max(0.0, pend_dev) ** 2,
        lwait,
        lwait - mean_lwait,
        math.log1p(max(replica.outstanding_requests, 0)) / 4.0,
        # --- decode occupancy: what prefill chunks steal steps from (TPOT) ---
        run_n,
        run_dev,
        max(0.0, run_dev) ** 2,
        # --- KV now and PROJECTED after this action (preemption lifts average TPOT) ---
        replica.kv_usage,
        replica.kv_usage - mean_kv,
        kv_over * kv_over,
        kv_proj,
        kv_proj_exc * kv_proj_exc,
        min(4.0, float(replica.running) * n / cap),
        # --- cache state of THIS request here, raw and with in-flight credit, plus the credit
        #     itself so its trust level is learned rather than hand-set ---
        _lg(unc_raw),
        lg_unc,
        unc / n_safe,
        lg_unc - _lg(min_unc),
        is_home,
        use_credit,
        # --- post-routing backlog this action commits ---
        lg_post,
        post_dev,
        lg_post - min_lg_post,
        max(0.0, post_dev) ** 2,
        lw_self,
        lw_self - math.log1p(min_wself),
        ext,                                    # work added x requests already queued here
        # --- interference: a miss costs more with resident decodes, high load, RISING load ---
        lg_unc * run_n,
        lg_unc * (mean_run / DEC_REF),
        lg_unc * congestion,
        lg_unc * util,
        lg_unc * rising,
        lg_post * replica.kv_usage,
    ]


# --------------------------------------------------------------------------- reward

def _tail(x, target):
    """Tail-only log penalty: exactly 0 at/below target, then log(x/target), unbounded.

    * mean reward ~ 0 in healthy states, so the bootstrapped return carries no large
      state-independent offset and |Q| stays O(1) instead of drifting to 30+;
    * the gradient does NOT vanish in the deep-backlog regime that sets the scored P99s
      (a factor-2 worse tail costs log 2 whether the tail is 6 s or 600 s), which is also the
      quantity the fitness integrates (log P99 ratios).
    """
    if x is None:
        return 0.0
    u = max(x, 0.0) / target
    if u <= 1.0:
        return 0.0
    return math.log(u)


def reward(outcome):
    r = -W_TTFT * _tail(outcome.ttft, TTFT_T)
    r -= W_E2E * _tail(outcome.e2e, E2E_T)
    if outcome.tpot is not None:
        r -= W_TPOT * _tail(outcome.tpot, TPOT_T)

    # Deterministic, exactly attributed price of the uncached prefill work this decision
    # committed. Under light load every tail term is 0, so this term alone makes the policy
    # cache-affine; as backlog grows the unbounded tail terms dominate and the crossover --
    # the thing heuristics hard-code as a constant -- is learned from the state.
    miss = max(0.0, float(outcome.num_prefill_tokens) -
               float(outcome.cached_tokens_at_routing))
    r -= W_MISS * _lg(miss)
    return r


def interval_reward(replicas_before, replicas_after, dt, ctx):
    """Low-variance cost accrued until the next decision (Bradtke-Duff interval cost).

    Charged on the state the decision leaves behind: the convex IMBALANCE of prefill backlog
    (tails come from the most backed-up replica) and of the decode batch (the TPOT externality
    that LATER routing decisions inflict on requests already decoding), plus a small convex
    term on the LEVEL so that total cluster work is not free. Imbalance is zero when the
    cluster is level, so packing prefix-sharing sessions stays free as long as it adds little
    uncached work -- unlike a pure level cost, which degenerates to the weakest heuristic.
    """
    k = len(replicas_after)
    if k == 0 or dt <= 0.0:
        return 0.0
    sum_lp = 0.0
    max_lp = 0.0
    sum_run = 0.0
    max_run = 0.0
    for r in replicas_after:
        lp = _lg(r.pending_prefill_tokens)
        sum_lp += lp
        if lp > max_lp:
            max_lp = lp
        rr = float(r.running)
        sum_run += rr
        if rr > max_run:
            max_run = rr
    mean_lp = sum_lp / k
    exc_p = max(0.0, max_lp - mean_lp)
    exc_d = max(0.0, max_run - sum_run / k) / DEC_REF
    return -dt * (W_INT_PREFILL * exc_p * exc_p
                  + W_INT_DECODE * exc_d * exc_d
                  + W_INT_LEVEL * mean_lp * mean_lp)
