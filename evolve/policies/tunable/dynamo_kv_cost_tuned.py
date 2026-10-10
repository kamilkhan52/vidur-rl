"""Dynamo KV-aware router cost (see systems/dynamo_kv_cost.py) with its two tunable constants exposed, for the
equal-budget tuning control (readiness plan §5 item 4). PREFILL_SCALE = 1, TEMPERATURE = 0 reproduces
systems/dynamo_kv_cost.py exactly.

  PREFILL_SCALE  Dynamo `prefill_load_scale` (weight of prefill tokens against decode blocks).
  TEMPERATURE    Dynamo router temperature: 0 = argmin; > 0 samples softmax(-cost / (TEMPERATURE * mean cost)).
"""
import math

PREFILL_SCALE = 1.0
TEMPERATURE = 0.0


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def choose(request, replicas, ctx):
    burst = ctx.state.get("dynamo_burst")
    if burst is None or burst[0] != ctx.now:
        burst = (ctx.now, [set() for _ in replicas])
        ctx.state["dynamo_burst"] = burst
    costs = []
    active = request.active_prefix_tokens or (0,) * len(replicas)
    for i, r in enumerate(replicas):
        prefill = r.pending_prefill_tokens + request.num_prefill_tokens - request.cached_prefix_tokens[i]
        same_burst = min(_match(burst[1][i], request.prefix_hashes) * request.block_tokens, request.num_prefill_tokens)
        overlap_active = max(active[i], same_burst)
        incoming_active = request.num_prefill_tokens - overlap_active
        decode = r.kv_usage * r.kv_capacity_tokens + r.pending_prefill_tokens + incoming_active
        costs.append(PREFILL_SCALE * prefill + decode)
    if TEMPERATURE > 0:
        scale = TEMPERATURE * max(sum(costs) / len(costs), 1e-9)
        lo = min(costs)
        w = [math.exp(-(c - lo) / scale) for c in costs]
        x, acc = ctx.rng.random() * sum(w), 0.0
        pick = len(w) - 1
        for i, wi in enumerate(w):
            acc += wi
            if x < acc:
                pick = i
                break
    else:
        best = min(costs)
        c = [i for i, v in enumerate(costs) if v == best]
        pick = c[0] if len(c) == 1 else ctx.rng.choice(c)
    burst[1][pick].update(request.prefix_hashes)
    return pick
