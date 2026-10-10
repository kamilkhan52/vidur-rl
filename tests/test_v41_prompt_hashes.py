"""v4.1: routers never see anything that reveals a request's output length.

A trace's block_hash_ids covers prompt and output ((prefill + decode) // block_size ids), so before v4.1 the length of
RequestView.prefix_hashes, and the router-side prefix ledgers built from it, revealed the decode length. Every
router-visible use now goes through prompt_block_hashes (the prompt's full blocks only)."""
import copy
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from conftest import base_args, build_simulator

PROMPT_A = [1, 2, 3, 4]  # 64-token prompt, 16-token blocks
RECORDING_POLICY = '''
SEEN = []

def choose(request, replicas, ctx):
    SEEN.append((ctx.now, request, len(request.prefix_hashes)))
    return request.position_in_burst % len(replicas)
'''


def _trace(path: Path, decode_a: int) -> Path:
    """A (prompt PROMPT_A, decode_a) and B in one burst at t=1; C (same prompt as A) while A is in flight; D later.
    Output-position ids are unique to the row, as in the v4 traces."""
    rows, uniq = [], [10_000]

    def row(t, prompt_ids, prefill, decode, sid):
        n_out = (prefill + decode) // 16 - len(prompt_ids)
        ids = list(prompt_ids) + list(range(uniq[0], uniq[0] + n_out))
        uniq[0] += 1000
        rows.append({"arrived_at": t, "num_prefill_tokens": prefill, "num_decode_tokens": decode,
                     "block_hash_ids": str(ids), "block_size": 16, "session_id": sid})

    row(1.0, PROMPT_A, 64, decode_a, 0)
    row(1.0, [100, 101, 102], 56, 24, 1)  # partial last prompt block: 56 // 16 = 3 prompt blocks
    row(1.0005, PROMPT_A, 64, 16, 2)  # A not yet prefilled: in flight on replica 0
    row(4.0, PROMPT_A, 64, 16, 3)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(tmp_path: Path, decode_a: int):
    pol = tmp_path / "rec.py"
    pol.write_text(RECORDING_POLICY)
    trace = _trace(tmp_path / f"trace_{decode_a}.csv", decode_a)
    argv = base_args(trace, tmp_path / f"out_{decode_a}", num_replicas=2, global_scheduler_config_type="evolved",
                     evolved_global_scheduler_config_policy_path=str(pol),
                     trace_request_generator_config_time_scale_factor="1.0")
    snaps = []  # router-owned state after every routing call
    sim = build_simulator(argv)
    gs = sim._scheduler
    orig_schedule = gs.schedule

    def schedule():
        mapping = orig_schedule()
        if mapping:
            snaps.append((gs._ctx.now, copy.deepcopy({k: getattr(gs, k) for k in (
                "_inflight", "_active", "_inflight_req", "_active_req", "_pending_prefill", "_outstanding",
                "_assigned", "_session_last")})))
        return mapping

    gs.schedule = schedule
    sim.run()
    return sim, gs._policy.SEEN, snaps


def test_prompt_block_hashes_cuts_to_prompt_blocks():
    from vidur.entities import Request
    from vidur.scheduler.global_scheduler.prompt_hashes import prompt_block_hashes

    r = Request(0.0, 56, 300, list(range((56 + 300) // 16)), 16)
    assert prompt_block_hashes(r, 16) == (0, 1, 2)  # 56 // 16 full prompt blocks
    assert prompt_block_hashes(r) == (0, 1, 2)  # falls back to the trace block size
    assert prompt_block_hashes(r, 0) == (0, 1, 2)  # no KV manager: same fallback
    r.restart()  # folds decoded tokens into the prefill count; the prompt stays the trace's
    assert prompt_block_hashes(r, 16) == (0, 1, 2)
    assert prompt_block_hashes(Request(0.0, 56, 8, None, None), 16) == ()


def test_identical_prompts_different_decode_lengths_give_identical_router_views(tmp_path):
    """A's decode length differs (16 vs 600 tokens). Every RequestView and all router-owned state up to the point
    where A could have finished in the short run must be identical; prefix_hashes has prefill // block_tokens ids."""
    sim_s, seen_s, snaps_s = _run(tmp_path, 16)
    sim_l, seen_l, snaps_l = _run(tmp_path, 600)
    assert len(seen_s) == len(seen_l) == 4
    for now, rv, n in seen_s + seen_l:
        assert rv.block_tokens == 16
        assert n == rv.num_prefill_tokens // rv.block_tokens  # a policy's len(prefix_hashes)
    early_s = [(now, asdict(rv)) for now, rv, _ in seen_s if now < 2.0]
    early_l = [(now, asdict(rv)) for now, rv, _ in seen_l if now < 2.0]
    assert len(early_s) == 3 and early_s == early_l
    # C arrives while A is in flight on replica 0: the router ledger still credits A's whole prompt
    c = early_s[2][1]
    assert c["prefix_hashes"] == tuple(PROMPT_A) and c["inflight_prefix_tokens"][0] == 64
    assert c["active_prefix_tokens"][0] == 64
    assert [s for s in snaps_s if s[0] < 2.0] == [s for s in snaps_l if s[0] < 2.0]
    assert len([s for s in snaps_s if s[0] < 2.0]) == 2
    # the ledgers hold prompt hashes only
    for _, st in snaps_s + snaps_l:
        for table in st["_inflight"] + st["_active"]:
            assert all(h < 10_000 for h in table)


def test_state_tracker_uses_prompt_hashes_only():
    """RL_CACHE_AWARE's shadow cache (in-flight blocks and the shadow hit ratio) sees prompt blocks only."""
    import types

    from vidur.entities import Request
    from vidur.scheduler.global_scheduler.state_tracker import StateTracker
    from vidur.types.replica_id import ReplicaId

    rs = types.SimpleNamespace(get_cached_prefill_length=lambda req: 0)
    rids = [ReplicaId(0), ReplicaId(1)]
    out = {}
    for decode in (16, 600):
        tr = StateTracker(2, {r: rs for r in rids}, enable_prefix_caching=True)
        a = Request(0.0, 64, decode, PROMPT_A + list(range(10_000, 10_000 + (64 + decode) // 16 - 4)), 16)
        tr.on_request_assigned(a, rids[0])
        b = Request(0.0, 64, 16, PROMPT_A + [20_000], 16)
        out[decode] = (set(tr._in_flight_blocks[rids[0]]), tr._extract_cache_hits_with_shadow(b))
    assert out[16] == out[600] == ({1, 2, 3, 4}, [1.0, 0.0])
