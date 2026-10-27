"""Llumnix dispatch (Sun et al., OSDI'24, arXiv 2406.03243), default dispatch rule ported from
github.com/llumnix-project/llumnix-ray tag v0.1.0 @ 312edd8a (Apache-2.0; HEAD 3fb6c037 keeps the same default).
Notes: ~/vidur-rl-priorwork/notes/llumnix.md.

Send each request to the instance with the highest freeness (v0.1.0 llumnix/instance_info.py:146-157,
'remaining_steps' load; dispatch_scheduler.py:42-53, 79-91, 119-125):
  n = running + waiting;  load = -inf if n == 0 else -((free_blocks - watermark_blocks) - waiting_blocks) / n
lowest load wins, so an empty instance always wins; ties: first instance (stable sort, no randomness).
Our views: free_blocks = (1 - kv_usage) * capacity blocks; watermark = 1 % of capacity (vLLM v0 default);
waiting_blocks = the router's uncached prompt tokens routed here and not yet prefilled (pending_prefill_tokens)
in blocks; n = the router's outstanding requests on the instance (running + waiting, and requests routed earlier
in the same arrival burst, which the replica has not enqueued yet). Deviations: no migration, de-fragmentation or auto-scaling (the paper's gains over INFaaS++ come from
migration, so this is the dispatch rule only, i.e. close to the paper's INFaaS++ baseline); the waiting term counts
uncached rather than full prompt tokens; exact views as for every other router here (Llumnix polls every 0.05 s and
does not update an instance's load for requests dispatched between polls).
"""

WATERMARK = 0.01


def choose(request, replicas, ctx):
    bt = request.block_tokens or 16
    best, pick = None, 0
    for i, r in enumerate(replicas):
        n = r.outstanding_requests  # = running + waiting, plus requests routed earlier in this burst
        if n == 0:
            return i
        total = r.kv_capacity_tokens / bt
        free = (1.0 - r.kv_usage) * total
        load = -((free - WATERMARK * total) - r.pending_prefill_tokens / bt) / n
        if best is None or load < best:
            best, pick = load, i
    return pick
