"""The block hashes a router may see: those of the request's prompt only (v4.1).

A trace's ``block_hash_ids`` covers the whole sequence, prompt and output: its length is
(num_prefill_tokens + num_decode_tokens) // block_size, with the prompt's ids first and ids unique to the row for
the output positions (and, in v4 traces, the partial last prompt block). The simulated KV cache needs the full list
(decode blocks are allocated and cached too), but anything a router sees must not reveal the output length, and the
list's length does. Every router-visible use of the hashes (policy views, router-side ledgers, RL state features)
goes through ``prompt_block_hashes``; the replica-side cache (``vidur/kv_cache``, replica schedulers) keeps using
``request.block_hash_ids``.
"""

from typing import Optional, Tuple


def prompt_block_hashes(request, block_tokens: Optional[int] = None) -> Tuple[int, ...]:
    """``block_hash_ids[: num_prefill_tokens // block_tokens]`` as a tuple: the hashes of the prompt's full blocks.

    ``num_prefill_tokens`` is the trace's prompt length (``original_num_prefill_tokens``, unchanged by preemption
    restarts, which fold decoded tokens into the prefill count). ``block_tokens`` is the KV block size in tokens;
    when it is not given (or 0, no KV cache manager) the trace's ``request.block_size`` is used, which the simulator
    validates to equal the cache block size. Returns () for a request without hashes. Only full prompt blocks are
    included: an incomplete block is never shared, so a prefix match can never extend past them.
    """
    hashes = request.block_hash_ids
    if not hashes:
        return ()
    bt = block_tokens or request.block_size
    if not bt:
        return ()
    return tuple(hashes[: request.original_num_prefill_tokens // bt])
