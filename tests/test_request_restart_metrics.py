"""Preemption restarts must not corrupt per-request metrics.

``Request.restart()`` folds already-decoded tokens into the prefill count so the
request can be re-prefilled in one go. That is correct for scheduling, but the
metrics store used the mutated counts and the cache-hit recorded at the *last*
scheduling, which inflated cache-hit ratios and distorted per-token
normalisations exactly in the overloaded runs where preemption happens.
"""

from vidur.entities import Request


def make_request(prefill=100, decode=20):
    return Request(arrived_at=0.0, num_prefill_tokens=prefill, num_decode_tokens=decode,
                   block_hash_ids=None, block_size=None, session_id=0)


def test_original_token_counts_survive_restart():
    r = make_request()
    r.on_cache_hit(32)
    r.on_batch_schedule(1.0)
    r.on_batch_end(2.0, 68)  # prefill complete (32 cached + 68 processed) -> first token
    assert r.is_prefill_complete
    r.on_batch_end(3.0, 5)  # five more decode tokens
    r.restart()
    # Scheduling view: decoded tokens became prefill
    assert r.num_prefill_tokens == 106 and r.num_decode_tokens == 14
    assert r.total_tokens == 120
    # Metrics view: the trace split is preserved
    assert r.original_num_prefill_tokens == 100
    assert r.original_num_decode_tokens == 20


def test_cache_hit_recorded_at_first_scheduling_only():
    r = make_request()
    r.on_cache_hit(16)
    r.on_batch_schedule(1.0)
    r.on_batch_end(2.0, 84)
    r.on_batch_end(3.0, 3)
    r.restart()
    # After a restart the request's own blocks look "cached"; that must not count.
    r.on_cache_hit(96)
    assert r.num_prefill_tokens_cached == 16
    assert r.num_restarts == 1


def test_prefill_completed_at_is_first_token_time():
    r = make_request()
    r.on_batch_schedule(1.0)
    r.on_batch_end(2.5, 100)
    r.on_batch_end(3.0, 2)
    r.restart()
    r.on_batch_schedule(10.0)
    r.on_batch_end(12.0, 103)  # re-prefill of 100 + 3 decoded tokens
    assert r.prefill_completed_at == 2.5
