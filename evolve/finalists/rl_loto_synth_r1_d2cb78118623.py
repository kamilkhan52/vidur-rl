"""Bandit, two-channel latency credit: absolute tail pressure + centred advantage.

Parent (-0.1534) learned cleanly but on the wrong signal. Its reward was PURELY the
deviation of log-latency from an online baseline keyed by (cluster backlog x prompt
size). That baseline chases the regime, so at a ramp peak / burst pile-up / mixed
workload -- the regimes that SET the scored P99 and the worst-scenario penalty -- a 60 s
TTFT reads as "normal here" and scores ~0. Absolute tail pressure disappeared exactly
where the tail is made: TTFT fell to 0.945-1.206 (ancestors: 0.51-0.86) while TPOT stayed
1.04-1.31, i.e. it paid 0.17 of log-ratio win for 0.06 of loss it never collected.

Fix (one hypothesis): keep the centred deviation as the across-replica advantage channel
(it is policy-neutral for a bandit because the bucket is fixed at routing and shared by
all actions) and ADD an absolute, non-saturating convex penalty in log-latency. With
algorithm "bandit" there is no bootstrap to multiply an action-independent level, so the
level costs only regression capacity -- which the shared cluster features already supply.
  * TTFT: absolute 0.55*(l + 0.15 l^2), l = log1p(ttft/1.5s) -> 0.58 at 1 s, 3.2 at 60 s,
    10 at 400 s. Never saturates, so the ramp peak keeps a gradient.
  * TPOT: priced as EXCESS over target (u + 0.3u^2, u = tpot/42ms - 1) instead of a centred
    log, so the common decode level is not learned as "the norm"; weight below TTFT's
    because the objective is a log ratio and TTFT has far more headroom.
  * E2E small (dominated by invisible output length).
  * reward_clip 6 -> 12: the parent's clip removed the knee levels it now needs.
The deterministic externality price is kept, zero-mean across the replicas of that
decision (pure direction, no level), with the decode term re-based on the cluster MEAN
decode batch rather than a fixed 0.75*8 threshold, so it still bites below the knee and
on 16-replica / unseen-trace regimes where that constant is mis-set.
Features identical to the parent: the experiment is the credit, not the state.
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

TTFT_REF = 1.5         # log scale, NOT a deadline
E2E_REF = 20.0
TPOT_REF = 0.042

W_TTFT_A = 0.55        # absolute tail pressure (the channel the parent deleted)
W_TTFT_R = 0.45        # centred across-replica advantage
W_E2E_A = 0.16
W_E2E_R = 0.12
W_TPOT_A = 0.50
W_TPOT_R = 0.40

# deterministic, zero-mean-across-replicas externality price
W_MISS = 0.25
W_INTF = 0.10
W_KVP = 0.25
W_DEC = 0.30

KV_SAFE = 0.75
KV_PROJ_SAFE = 0.85

CLIP_L = 8.0
CLIP_D = 6.0
CLIP_X = 12.0

CONFIG = {
    "algorithm": "bandit",
    "beta": 0.05,
    "hidden": [128, 96],
    "lr": 3e-4,
    "batch_size": 128,
    "buffer_size": 200000,
    "train_every": 2,
    "learning_starts": 500,
    "target_update": 1000,
    "epsilon_start": 0.30,
    "epsilon_end": 0.07,
    "epsilon_decay_decisions": 25000,
    "teacher_mix": 0.35,
    "teacher_mix_end": 0.02,
    "imitation_weight": 0.5,
    "imitation_weight_end": 0.0,
    "teacher_decay_decisions": 22000,
    "imitation_margin": 0.3,
    "grad_clip": 10,
    "reward_clip": 12,
}

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
    u = run / DEC_REF - 0.75
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
    kpes = []
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
        kpes.append(_cl((r.kv_usage + u / cap - KV_PROJ_SAFE) / (1.0 - KV_PROJ_SAFE),
                        0.0, 1.5))
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

    mean_run = sum_run / k
    # --- deterministic externality price, re-based on the cluster mean decode batch so
    #     it does not depend on a fixed absolute knee constant ---
    costs = []
    sum_cost = 0.0
    for j in range(k):
        rr = runs[j]
        dv = max(0.0, (rr - mean_run) / DEC_REF)
        kp = kpes[j]
        cst = (W_MISS * lgu[j] + W_INTF * lgu[j] * min(rr / DEC_REF, 3.0)
               + W_KVP * kp * kp + W_DEC * (dv + 0.5 * dv * dv))
        costs.append(cst)
        sum_cost += cst

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
    st["mean_run"] = mean_run
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

    # --- routing-time record read back by reward() ---
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
        # --- request / cluster / regime (shared across actions: lets the hidden layers
        #     rescale every per-replica term by the regime) ---
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

    Bucket fixed at ROUTING time and identical for all actions of that decision, so for
    a bandit the subtraction is exactly policy-neutral. This is the ADVANTAGE channel.
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

    # --- TTFT: queueing is 38-94 % of a tail request's time. TWO channels: the absolute
    #     convex log level (keeps a gradient at the ramp peak / burst / mixed-workload
    #     regimes that set the scored P99 -- this is what the parent lost) and the
    #     centred deviation (clean across-replica advantage, low variance). ---
    lt = math.log1p(max(outcome.ttft, 0.0) / TTFT_REF)
    r = -W_TTFT_A * _cl(lt + 0.15 * lt * lt, 0.0, 10.0)
    r -= W_TTFT_R * _g(_center(0, bucket, lt))

    # --- E2E: small, per-request E2E is dominated by invisible output length ---
    le = math.log1p(max(outcome.e2e, 0.0) / E2E_REF)
    r -= W_E2E_A * _cl(le + 0.10 * le * le, 0.0, 6.0)
    r -= W_E2E_R * _g(_center(1, bucket, le))

    # --- TPOT: the chronic 1.04-1.31x deficit. Priced as EXCESS over target, so the
    #     common decode level is never absorbed as "the norm"; weighted below TTFT
    #     because the log-ratio objective has far more headroom on TTFT. ---
    if outcome.tpot is not None:
        z = max(outcome.tpot, 1e-4) / TPOT_REF
        u = max(0.0, z - 1.0)
        r -= W_TPOT_A * _cl(u + 0.3 * u * u, 0.0, 6.0)
        r -= W_TPOT_R * _g(_center(2, bucket, math.log(z)))

    # Deterministic externality price, ZERO MEAN across the replicas available at that
    # decision: pure direction, no level. Charges the decision for the uncached work it
    # commits, for dropping it where decodes are resident, for the KV it projects and for
    # piling decodes above the cluster mean.
    return r - det
