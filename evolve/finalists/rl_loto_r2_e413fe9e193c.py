"""Bandit advantage learning: outcome centred by a congestion-conditioned baseline.

Parent (-0.4657) failed in one way, in two places:
  * CREDIT. With dqn/beta=0.8 the return multiplies the per-decision reward by ~9. The
    reward is only zero in healthy states; at the knee EVERY action earns ~-3, so the
    target is an action-INDEPENDENT level (Q -0.4 -> -26, loss 0.21 -> 1.5) and the
    across-replica advantage falls under the regression residual. Action share collapses
    (0.31 on one replica) and training P99 TTFT hits 518 s (ramp) / 1558 s (mix) --
    the same regimes as the worst scored scenario (Synth ramp 2.15x TTFT).
  * FEATURES. _cl(.,0,4) on log-token levels saturates at ~440k queued tokens and the
    interaction terms (lg_unc*lg_mean_pend ~ 7) clip at 4 much earlier, so at the knee
    many inputs read the same constant on every replica.

Changes (one hypothesis: regress an ADVANTAGE, and stay legible at the knee):
  1. algorithm "bandit": no bootstrap, hence no discounted congestion level in the
     target and no moving target. Targets are bounded and i.i.d. given the state.
  2. reward = -(convex deviation of log-latency from an ONLINE baseline keyed by
     (cluster backlog bucket x prompt size bucket) captured at routing). The baseline is
     action-independent, so for a bandit subtracting it is exactly policy-neutral; it
     removes the regime level in every regime (ramp peak, burst, mix, unseen trace)
     without the lag of a global EMA and without ever saturating. The learned quantity
     becomes "how much better/worse than the cluster norm does THIS replica do".
  3. The externality the bootstrap used to carry is replaced by a deterministic price
     (uncached tokens, uncached x resident decodes, projected KV, convex decode batch)
     entered as cost[a] - mean_a cost[a]: ZERO MEAN across replicas, so it adds pure
     direction and no level. It is also exposed as a feature so its trust is learned.
  4. All levels/interactions de-saturated (clips 8 / 12); deviations, ranks and gaps
     unchanged. Two-timescale regime EWMAs and backlog-growth trend kept (the ramp).
  5. Teacher 0.35 -> 0.02 by 22k decisions; bandit regression is unbiased under
     off-policy teacher data, so teacher episodes are free state coverage.
"""

import math
import collections

TOKENS = 8192.0
DEC_REF = 8.0
PREFILL_RATE = 6000.0
WINDOW_S = 60.0
CREDIT = 0.9
TAU_S = 3.0
TAU_L = 30.0

TTFT_REF = 1.0         # log scale, NOT a deadline
E2E_REF = 20.0
TPOT_REF = 0.040

W_TTFT = 1.00
W_E2E = 0.25           # per-request E2E is dominated by invisible output length
W_TPOT = 1.10          # the chronic deficit of this lineage (1.01-1.31x)

# deterministic, zero-mean-across-replicas externality price
W_MISS = 0.25
W_INTF = 0.10          # uncached work landing on resident decodes
W_KVP = 0.25
W_DEC = 0.20

KV_SAFE = 0.75
KV_PROJ_SAFE = 0.85
DEC_SAFE = 0.75

CLIP_L = 8.0           # level clip (~24M tokens): never pins at the knee
CLIP_D = 6.0
CLIP_X = 12.0          # interaction clip

CONFIG = {
    "algorithm": "bandit",
    "beta": 0.05,
    "hidden": [128, 96],
    "lr": 3e-4,
    "batch_size": 128,
    "buffer_size": 200000,         # > all training decisions: nothing evicted
    "train_every": 2,
    "learning_starts": 500,
    "target_update": 1000,
    "epsilon_start": 0.30,
    "epsilon_end": 0.07,           # a bandit needs counterfactual coverage to the end
    "epsilon_decay_decisions": 25000,
    "teacher_mix": 0.35,
    "teacher_mix_end": 0.02,
    "imitation_weight": 0.5,
    "imitation_weight_end": 0.0,
    "teacher_decay_decisions": 22000,
    "imitation_margin": 0.3,
    "grad_clip": 10,
    "reward_clip": 6,
}

# ---- module state: routing-time record per decision, and the baseline tables -------
_PEND = {}
_ORDER = collections.deque()
_BASE = {}


def _lg(x):
    return math.log1p(max(x, 0.0) / TOKENS)


def _cl(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _psi_d(run):
    u = run / DEC_REF - DEC_SAFE
    if u <= 0.0:
        return 0.0
    return min(u * u, 4.0)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda j: vals[j])
    out = [0.0] * len(vals)
    pos = 0.0
    for j in order:
        out[j] = pos
        pos += 1.0
    return out


# --------------------------------------------------------------- per-decision memory

def _memory(request, replicas, ctx):
    """One O(k log k) pass per routing decision; the k feature calls then read it."""
    st = ctx.state
    dkey = (request.session_id, request.arrived_at, request.num_prefill_tokens)
    if st.get("key") == dkey:
        return st
    st["key"] = dkey

    k = len(replicas)
    n = float(request.num_prefill_tokens)
    cached = request.cached_prefix_tokens
    home = request.session_last_replica
    now = ctx.now

    # --- in-flight prefix credit: previous turn of this session still queued ---
    table = st.get("turns")
    if table is None:
        table = {}
        st["turns"] = table
    sid = request.session_id
    prev = table.get(sid) if sid is not None else None
    if sid is not None:
        table[sid] = (request.arrived_at, n)
        if len(table) > 20000:
            cut = request.arrived_at - WINDOW_S
            st["turns"] = dict((a, b) for a, b in table.items() if b[0] > cut)
    credit = 0.0
    st["has_prev"] = 0.0
    st["prev_age"] = 1.0
    if prev is not None:
        age = request.arrived_at - prev[0]
        if 0.0 <= age <= WINDOW_S:
            st["has_prev"] = 1.0
            st["prev_age"] = age / WINDOW_S
            if home >= 0:
                credit = min(prev[1] * CREDIT, n)
    st["credit"] = credit

    uncs = []
    lgu = []
    lgp = []
    lgo = []
    runs = []
    kvs = []
    caps = []
    lws = []
    costs = []
    sum_pend = 0.0
    max_pend = 0.0
    sum_run = 0.0
    max_run = 0.0
    sum_kv = 0.0
    sum_lw = 0.0
    sum_lo = 0.0
    min_lo = 1e18
    min_unc = n
    min_w = 1e18
    sum_cost = 0.0
    for r in replicas:
        p = float(r.pending_prefill_tokens)
        c = float(cached[r.index])
        if r.index == home and credit > c and (r.waiting > 0 or p > 0.0):
            c = credit
        u = max(0.0, n - c)
        cap = max(float(r.kv_capacity_tokens), 1.0)
        rr = float(r.running)
        lu = _lg(u)
        lo_ = _lg(p + u)
        lw = math.log1p(max(r.waiting, 0)) / 3.0
        uncs.append(u)
        lgu.append(lu)
        lgp.append(_lg(p))
        lgo.append(lo_)
        runs.append(rr)
        kvs.append(r.kv_usage)
        caps.append(cap)
        lws.append(lw)
        kpe = _cl((r.kv_usage + u / cap - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE), 0.0, 1.5)
        cst = (W_MISS * lu + W_INTF * lu * min(rr / DEC_REF, 3.0)
               + W_KVP * kpe * kpe + W_DEC * _psi_d(rr))
        costs.append(cst)
        sum_cost += cst
        sum_pend += p
        if p > max_pend:
            max_pend = p
        sum_run += rr
        if rr > max_run:
            max_run = rr
        sum_kv += r.kv_usage
        sum_lw += lw
        sum_lo += lo_
        if lo_ < min_lo:
            min_lo = lo_
        if u < min_unc:
            min_unc = u
        w = (p + u) / PREFILL_RATE
        if w < min_w:
            min_w = w

    st["uncs"] = uncs
    st["lgu"] = lgu
    st["lgp"] = lgp
    st["lgo"] = lgo
    st["runs"] = runs
    st["kvs"] = kvs
    st["caps"] = caps
    st["lws"] = lws
    st["costs"] = costs
    st["mean_cost"] = sum_cost / k
    st["rk_p"] = _ranks(lgp)
    st["rk_o"] = _ranks(lgo)
    st["rk_r"] = _ranks(runs)
    mean_pend = sum_pend / k
    st["mean_pend"] = mean_pend
    st["lg_mean_pend"] = _lg(mean_pend)
    st["lg_max_pend"] = _lg(max_pend)
    st["mean_run"] = sum_run / k
    st["max_run"] = max_run
    st["mean_kv"] = sum_kv / k
    st["mean_lw"] = sum_lw / k
    st["mean_lo"] = sum_lo / k
    st["min_lo"] = min_lo
    st["min_unc"] = min_unc
    st["min_w"] = min_w

    # --- regime: two-timescale rates and the growth of log backlog (ramps) ---
    last_t = st.get("t")
    lgm = st["lg_mean_pend"]
    if last_t is not None and now > last_t:
        gap = max(now - last_t, 1e-4)
        ws = 1.0 - math.exp(-gap / TAU_S)
        wl = 1.0 - math.exp(-gap / TAU_L)
        ir = 1.0 / gap
        it = n / gap
        a = st.get("lam_s")
        b = st.get("lam_l")
        c2 = st.get("tok_s")
        st["lam_s"] = ir if a is None else a + ws * (ir - a)
        st["lam_l"] = ir if b is None else b + wl * (ir - b)
        st["tok_s"] = it if c2 is None else c2 + ws * (it - c2)
        pl = st.get("lgm")
        if pl is not None:
            ig = (lgm - pl) / gap
            g = st.get("grow")
            st["grow"] = ig if g is None else g + ws * (ig - g)
    st["t"] = now
    st["lgm"] = lgm

    # --- register the routing-time record used by reward() ---
    cb = int(_cl(lgm * 2.0, 0.0, 7.99))
    sb = int(_cl(n / TOKENS * 1.5, 0.0, 3.99))
    pk = (round(now, 6), int(request.num_prefill_tokens))
    _PEND[pk] = (cb * 4 + sb, costs, st["mean_cost"])
    _ORDER.append(pk)
    while len(_ORDER) > 4000:
        _PEND.pop(_ORDER.popleft(), None)
    return st


# --------------------------------------------------------------------------- features

def replica_features(request, replica, replicas, ctx):
    st = _memory(request, replicas, ctx)
    i = replica.index
    k = len(replicas)
    kk = max(k - 1, 1)
    n = float(request.num_prefill_tokens)
    nn = max(n, 1.0)

    u = st["uncs"][i]
    lgu = st["lgu"][i]
    lgp = st["lgp"][i]
    lgo = st["lgo"][i]
    rr = st["runs"][i]
    kv = st["kvs"][i]
    cap = st["caps"][i]
    lw = st["lws"][i]

    lgmp = st["lg_mean_pend"]
    mean_run = st["mean_run"]
    mean_kv = st["mean_kv"]
    mean_lw = st["mean_lw"]
    mean_lo = st["mean_lo"]

    lam_s = st.get("lam_s") or 0.0
    lam_l = st.get("lam_l")
    if lam_l is None:
        lam_l = lam_s
    tok_s = st.get("tok_s") or 0.0
    grow = st.get("grow") or 0.0
    util = math.tanh(tok_s / (k * PREFILL_RATE))
    trend = math.tanh(math.log((lam_s + 0.1) / (lam_l + 0.1)))
    rising = max(0.0, trend)
    growth = math.tanh(grow)
    growing = max(0.0, growth)
    congestion = min(st["mean_pend"] / nn, 16.0) / 16.0

    pend_dev = lgp - lgmp
    post_dev = lgo - mean_lo
    run_n = rr / DEC_REF
    run_dev = (rr - mean_run) / DEC_REF
    kv_over = max(0.0, kv - KV_SAFE) / (1.0 - KV_SAFE)
    kv_proj = kv + u / cap
    kv_pe = _cl((kv_proj - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE), 0.0, 2.0)
    post = float(replica.pending_prefill_tokens) + u
    lw_self = math.log1p(post / PREFILL_RATE)
    burst = max(float(request.burst_size), 1.0)
    unc_raw = max(0.0, n - float(request.cached_prefix_tokens[i]))
    home = request.session_last_replica
    is_home = 1.0 if home == i else 0.0
    use_credit = 0.0
    if is_home > 0.0:
        d = st["credit"] - float(request.cached_prefix_tokens[i])
        if d > 0.0:
            use_credit = _cl(d / nn, 0.0, 1.0)

    d_q = _cl(lgo - lgp, 0.0, CLIP_L)
    d_d = _psi_d(rr + 1.0) - _psi_d(rr)
    d_kv = kv_pe * kv_pe - (max(0.0, kv - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE)) ** 2

    return [
        # --- request / cluster / regime (identical across actions: lets the hidden
        #     layers rescale every per-replica term by the regime) ---
        _cl(n / TOKENS, 0.0, CLIP_L),
        1.0 if home < 0 else 0.0,
        _cl(float(request.position_in_burst) / burst, 0.0, 1.0),
        _cl(math.log1p(burst) / 4.0, 0.0, 2.0),
        _cl(lgmp, 0.0, CLIP_L),
        _cl(st["lg_max_pend"] - lgmp, 0.0, CLIP_L),
        _cl(mean_run / DEC_REF, 0.0, CLIP_L),
        _cl((st["max_run"] - mean_run) / DEC_REF, 0.0, CLIP_L),
        mean_kv,
        _cl(mean_lw, 0.0, CLIP_L),
        congestion,
        util,
        trend,
        growth,
        st["has_prev"],
        st["prev_age"],
        # --- standing prefill backlog here ---
        _cl(lgp, 0.0, CLIP_L),
        _cl(pend_dev, -CLIP_D, CLIP_D),
        _cl(max(0.0, pend_dev) ** 2, 0.0, CLIP_L),
        _cl(lw, 0.0, CLIP_L),
        _cl(lw - mean_lw, -CLIP_D, CLIP_D),
        _cl(math.log1p(max(replica.outstanding_requests, 0)) / 4.0, 0.0, CLIP_L),
        st["rk_p"][i] / kk,
        # --- decode occupancy: what a prefill chunk steals steps from (TPOT) ---
        _cl(run_n, 0.0, CLIP_L),
        _cl(run_dev, -CLIP_D, CLIP_D),
        _cl(max(0.0, run_dev) ** 2, 0.0, CLIP_L),
        st["rk_r"][i] / kk,
        # --- KV now and projected after this action (preemption risk) ---
        kv,
        _cl(kv - mean_kv, -1.0, 1.0),
        _cl(kv_over * kv_over, 0.0, 4.0),
        _cl(kv_proj, 0.0, 3.0),
        _cl(kv_pe * kv_pe, 0.0, 4.0),
        _cl(rr * n / cap, 0.0, 6.0),
        # --- cache state of this request here ---
        _cl(_lg(unc_raw), 0.0, CLIP_L),
        _cl(lgu, 0.0, CLIP_L),
        _cl(u / nn, 0.0, 1.0),
        _cl(lgu - _lg(st["min_unc"]), 0.0, CLIP_L),
        is_home,
        use_credit,
        1.0 if u <= st["min_unc"] else 0.0,
        # --- post-routing backlog this action commits, and the wait it implies ---
        _cl(lgo, 0.0, CLIP_L),
        _cl(post_dev, -CLIP_D, CLIP_D),
        _cl(lgo - st["min_lo"], 0.0, CLIP_L),
        _cl(max(0.0, post_dev) ** 2, 0.0, CLIP_L),
        st["rk_o"][i] / kk,
        _cl(lw_self, 0.0, CLIP_L),
        _cl(lw_self - math.log1p(st["min_w"]), 0.0, CLIP_L),
        _cl(math.log1p(post / nn) / 3.0, 0.0, 4.0),
        # --- marginal cluster cost of this action ---
        d_q,
        _cl(d_d * 4.0, 0.0, 4.0),
        _cl(d_kv, 0.0, 4.0),
        # --- interference crossovers: a miss costs more with resident decodes, at high
        #     load, and when load / backlog is RISING (learned, not a constant) ---
        _cl(lgu * run_n, 0.0, CLIP_X),
        _cl(lgu * lgmp, 0.0, CLIP_X),
        _cl(lgu * congestion, 0.0, CLIP_X),
        _cl(lgu * util, 0.0, CLIP_X),
        _cl(lgu * rising, 0.0, CLIP_X),
        _cl(lgu * growing, 0.0, CLIP_X),
        _cl(lgo * kv, 0.0, CLIP_L),
        # --- the deterministic zero-mean externality price of this action ---
        _cl(st["costs"][i] - st["mean_cost"], -3.0, 3.0),
    ]


# --------------------------------------------------------------------------- reward

def _center(tag, bucket, x):
    """Deviation of x from an online baseline for (metric, congestion x size bucket).

    The bucket is fixed at ROUTING time and is identical for all actions of that
    decision, so for a bandit this subtraction is exactly policy-neutral: it removes the
    regime level (ramp peak, burst, mix, unseen trace) and leaves the across-replica
    advantage as the whole regression target. Count-warmed EWMA, then rate 0.01.
    """
    key = (tag, bucket)
    rec = _BASE.get(key)
    if rec is None:
        rec = [x, 0.0]
        _BASE[key] = rec
    d = x - rec[0]
    a = 1.0 / (1.0 + rec[1]) if rec[1] < 100.0 else 0.01
    rec[0] = rec[0] + a * d
    rec[1] = rec[1] + 1.0
    return d


def _g(d):
    """Convex above the norm (P99 objective), linear-damped below; never saturates."""
    if d > 0.0:
        return _cl(d + 0.3 * d * d, 0.0, 6.0)
    return _cl(0.6 * d, -2.0, 0.0)


def reward(outcome):
    key = (round(outcome.routed_at, 6), int(outcome.num_prefill_tokens))
    rec = _PEND.pop(key, None)
    if rec is None:
        bucket = -1
        det = 0.0
    else:
        bucket = rec[0]
        cs = rec[1]
        idx = outcome.replica
        det = cs[idx] - rec[2] if 0 <= idx < len(cs) else 0.0

    # Queueing is 38-94 % of a tail request's time: the dominant learnable signal, in
    # log space so it keeps a gradient from 0.1 s to 1000 s.
    r = -W_TTFT * _g(_center(0, bucket, math.log1p(max(outcome.ttft, 0.0) / TTFT_REF)))
    r -= W_E2E * _g(_center(1, bucket, math.log1p(max(outcome.e2e, 0.0) / E2E_REF)))
    if outcome.tpot is not None:
        lt = math.log(max(outcome.tpot, 1e-4) / TPOT_REF)
        r -= W_TPOT * _g(_center(2, bucket, lt))

    # Deterministic externality price, ZERO MEAN across the replicas available at that
    # decision: pure direction, no level. This is the channel that replaces the
    # bootstrap -- it charges a decision for the uncached work it commits, for dropping
    # that work where decodes are resident, and for the KV it projects.
    return r - det
