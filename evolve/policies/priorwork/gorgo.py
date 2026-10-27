"""GORGO (Toniolo et al., arXiv 2602.11688v3), routing rule ported from github.com/atoniolo76/GORGO @ 4d6adafc
(pyproject.toml declares MIT; no LICENSE file). Notes: ~/vidur-rl-priorwork/notes/gorgo.md.

score_u = uncached_u + W_QUEUE * queued_u, lowest wins (gorgo/gorgo.py:280-293), where
  uncached_u = prompt tokens not covered by the longest prefix the router has sent to u before (a router-side trie
               of every routed prompt, never evicted: modal_proxy.py:428-438, 2921);
  queued_u   = full prompt tokens of requests sent to u and not yet completed (modal_proxy.py:2857-2858, 2878-2893).
W_QUEUE = 0.5, the value GORGO's online (1+1)-ES reached on its tuning window (specs/c64/eval_ts2/apr6.json; the top
of its search range [0.05, 0.5]). Ties: first replica (Python min over dict order, gorgo.py:293).
Deviations: the network term W_RTT * rtt_u is dropped (co-located replicas have equal RTT, so it never changes the
argmin); the trie is matched on 16-token prompt blocks (prefix_hashes) instead of token ids; requests of one arrival
burst see each other through the router's own ledger, as in the original (the trie is updated on dispatch).
"""

W_QUEUE = 0.5


def _match(index, hashes):
    n = 0
    for h in hashes:
        if h not in index:
            break
        n += 1
    return n


def _state(ctx, n):
    st = ctx.state.get("gorgo")
    if st is None:
        st = ctx.state["gorgo"] = {"trie": [set() for _ in range(n)], "queued": [0] * n}
    return st


def choose(request, replicas, ctx):
    st = _state(ctx, len(replicas))
    best, pick = None, 0
    for i in range(len(replicas)):
        cached = min(_match(st["trie"][i], request.prefix_hashes) * request.block_tokens, request.num_prefill_tokens)
        score = (request.num_prefill_tokens - cached) + W_QUEUE * st["queued"][i]
        if best is None or score < best:
            best, pick = score, i
    st["queued"][pick] += request.num_prefill_tokens
    st["trie"][pick].update(request.prefix_hashes)
    return pick


def observe(outcome, ctx):
    if outcome.event == "end":
        st = _state(ctx, ctx.num_replicas)
        st["queued"][outcome.replica] -= outcome.num_prefill_tokens
