"""Engram's best router (Karimi et al., arXiv 2603.21321v1 Fig. 21, pp. 29-30; the code is run 9's best_solution in
Zenodo record 20015246, token-identical to the figure, mean E2E 23.18 s on Engram's benchmark). Transcription with
provenance: ~/vidur-rl-priorwork/notes/engram_router_verbatim.py; notes: ~/vidur-rl-priorwork/notes/engram.md.
Repos: github.com/mit-nms/Engram @ 5295858 and mit-nms/vidur_simulator @ 2666cb6 (MIT).

Requests wait at the router, shortest prompt first (ties by arrival). The head request may go to a replica whose
waiting queue is shorter than 1 if any of its requests is decoding, else shorter than 4, and whose projected KV
utilisation (allocated blocks + the request's prompt + 6 tokens, in blocks) stays <= 0.95; among those, the lowest
projected utilisation, then fewer waiting, then fewer decoding, then replica order. If none qualifies the queue waits
(no look-past). Placements within one call update the local snapshot (allocated blocks and waiting count).
Deviations: replicas run our vllm_v1 scheduler, not Sarathi; "decoding" requests are those whose first token the
router has seen and whose end it has not (from observe()), not the replica's active queue scan; waiting = the
replica's waiting-queue length from the views.
"""
from math import ceil

DECODE_HEADROOM = 6
MEM_UTIL_LIMIT = 0.95
CAP_WITH_DECODE = 1
CAP_PREFILL_ONLY = 4


def _st(ctx, n):
    st = ctx.state.get("engram")
    if st is None:
        st = ctx.state["engram"] = {"decoding": [0] * n}
    return st


def dispatch(held, replicas, ctx):
    n = len(replicas)
    st = _st(ctx, n)
    bt = held[0].block_tokens or 16
    num_blocks = replicas[0].kv_capacity_tokens // bt
    alloc = [r.kv_usage * num_blocks for r in replicas]
    pending = [r.waiting for r in replicas]
    decode = list(st["decoding"])
    order = sorted(range(len(held)), key=lambda k: (held[k].num_prefill_tokens, held[k].arrived_at))
    out = []
    for k in order:
        req = held[k]
        need = ceil((req.num_prefill_tokens + DECODE_HEADROOM) / bt)
        adm = []
        for i in range(n):
            cap = CAP_WITH_DECODE if decode[i] > 0 else CAP_PREFILL_ONLY
            if pending[i] >= cap:
                continue
            util = (alloc[i] + need) / num_blocks
            if util > MEM_UTIL_LIMIT:
                continue
            adm.append((util, pending[i], decode[i], i))
        if not adm:
            break
        t = min(adm)[3]
        out.append((k, t))
        alloc[t] += need
        pending[t] += 1
    return out


def observe(outcome, ctx):
    st = _st(ctx, ctx.num_replicas)
    if outcome.event == "first_token":
        st["decoding"][outcome.replica] += 1
    elif outcome.event == "end":
        st["decoding"][outcome.replica] -= 1
