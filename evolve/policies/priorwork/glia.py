"""Glia's Head-Room Admission (HRA) router (Hamadanian et al., arXiv 2510.27176v5 Fig. 10, pp. 15-16; v1 Fig. 9,
p. 20; the listings are token-identical). Not released in a repository; transcription in
~/vidur-rl-priorwork/notes/glia_router_verbatim.py, notes in ~/vidur-rl-priorwork/notes/glia.md.

Requests wait at the router, shortest prompt first (ties by arrival). The head request goes to the admissible replica
with the lowest projected usage (allocated KV blocks + 1.6 x the prompt blocks of requests waiting there), ties by
queue length (waiting + active), then replica order; a replica is admissible if, with the head request's
1.6 x prompt blocks added, at least 3 % of its blocks stay free. If no replica is admissible the queue waits (no
look-past). Placements within one call update the local snapshot. The harness calls dispatch() on every arrival and
after every batch that completed a request, as Glia's Vidur fork does (mit-nms/vidur_simulator 2666cb6).
Constants as printed: DECODE_TO_PREFILL_RATIO = 0.6, SAFETY_FRACTION = 0.03 (tuned by Glia for ShareGPT on 4 x A10;
used unchanged, as the paper gives no other setting).
Deviations: the prompt tokens waiting on a replica come from the router's own ledger (prompt tokens routed there
whose first token has not been seen) rather than the replica's request queue, so a request still in prefill counts
both there and in its allocated blocks until its first token; allocated blocks = kv_usage x capacity (the replica's
allocated-block count); queue length = the router's outstanding requests on the replica (pending + active).
"""
from math import ceil

DECODE_TO_PREFILL_RATIO = 0.6
SAFETY_FRACTION = 0.03


def _st(ctx, n):
    st = ctx.state.get("glia")
    if st is None:
        st = ctx.state["glia"] = {"waiting_tokens": [0] * n}
    return st


def dispatch(held, replicas, ctx):
    n = len(replicas)
    st = _st(ctx, n)
    bt = held[0].block_tokens or 16
    max_blocks = replicas[0].kv_capacity_tokens // bt
    min_free = int(max_blocks * SAFETY_FRACTION)
    alloc = [r.kv_usage * max_blocks for r in replicas]
    reserved = [ceil(st["waiting_tokens"][i] * (1 + DECODE_TO_PREFILL_RATIO) / bt) for i in range(n)]
    qlen = [r.outstanding_requests for r in replicas]
    order = sorted(range(len(held)), key=lambda k: (held[k].num_prefill_tokens, held[k].arrived_at))
    out = []
    for k in order:
        req = held[k]
        need = ceil(req.num_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) / bt)
        adm = [i for i in range(n) if max_blocks - (alloc[i] + reserved[i] + need) >= min_free]
        if not adm:
            break
        t = min(adm, key=lambda i: (alloc[i] + reserved[i], qlen[i], i))
        out.append((k, t))
        reserved[t] += need
        qlen[t] += 1
        st["waiting_tokens"][t] += req.num_prefill_tokens
    return out


def observe(outcome, ctx):
    if outcome.event == "first_token":
        _st(ctx, ctx.num_replicas)["waiting_tokens"][outcome.replica] -= outcome.num_prefill_tokens
