# Chunk A (simulator v4 changes), round 2

**Verdict: approve with minors.** All eleven round-1 findings are fixed. I reproduced each round-1 failure scenario against `v4.1-sim-fixes`, and every one now behaves as claimed.
The new `route_delay_s` and `view_refresh_s` semantics are correct and internally consistent, and both survive `sim_fork` deepcopy forks exactly. My independent golden v4 check gives 30/30 byte-identical on branch code_version `c8bb68b7636b`.
The remaining items are one minor modelling caveat for the paper (the prefix-cache probe stays live under `view_refresh_s`), one minor residual from round 1 (Agg backend), and nits. None of them blocks the merge.

Scope: `git -C /Users/kamil/vidur-rl-v41 diff a3ef5ef..v4.1-sim-fixes -- vidur tests` (f16fc4a tests and spikes, 559218e hashed `vidur/` plus `tests/test_v41_sim_fixes.py`).
Repro scripts are in the session scratchpad (`repro/r1..r5`). They were run from the worktree with `PYTHONPATH=<worktree>:<repro>` and one simulator process at a time.

## Golden check (task 3)

`python -m evolve.analysis.golden check --manifest evolve/golden/v4_Kamils-Mac-mini.json --workers 2` was run in the worktree with `EVOLVE_KEEP_DIR` unset:
```
env   code_version: c16266a8ef49 vs c8bb68b7636b
all 30 cells byte-identical (718 s)
```
This confirms the fixer's v4 claim. I did not re-run v3.

## Round-1 findings, re-tested against the branch (task 1)

| # | Round-1 finding | Status | Evidence on the branch |
|---|---|---|---|
| 1 | Stuck path crashes (queue not iterable, `ReplicaId` not JSON) | **fixed** | Round-1 scenario (wedged replica, 6 Synth requests, `vidur.main`): `exit code: 3`; `stuck_requests.json` = `[{'where':'waiting','replica':0,'request_id':0}, …2, …4]`; metrics hold 3 rows; `predictor_info.json` written; no `*.tmp*` left. |
| 2 | `route_delay_s` splits same-time bursts | **fixed** | Round-1 repro (4 requests at t=1, 1 replica, d=0.5): prefill_e2e `[0.5188]*4` against `[0.0188]*4`; e2e diff `[0.5, 0.5, 0.5, 0.5]` exactly. Tool v4 slice (58 requests, 4 burst times, 4 replicas, light load): max\|e2e diff − d\| = 1e-15 for d = 0.05 and d = 0.5. |
| 3 | `view_refresh_s` re-polls at decisions (no staleness when R < gap) | **fixed** | Round-1 scenario (Conv v4, 120 requests, 4 replicas, scale 0.5): views at R=1.0 now differ from R=0 (`views identical: False`). See the note below on why that scenario's sha does not move. |
| 4 | Block-size test passes without the validator | **fixed** | The trace is now consistent at 32 tokens, and the test requires the specific `ValueError`. Its second half builds the same trace with the validator stubbed out, which shows the validator is load-bearing. |
| 5 | Outputs written before the RL checkpoint | **fixed** | r5, four cases of save/write raising. Order is always `['save','write']`. The outputs are written when the save raises, and the save has run when the write raises. Every failure propagates (non-zero exit). When both fail, the `OSError` carries `__context__=RuntimeError('save failed')`. |
| 6 | `clamp_hits` comment | **fixed** | The comment and the `provenance()` docstring now say "beyond the prediction grid" and name the Llama-2 4032 example. |
| 7 | Provenance does not say whether the bundle was loaded | **fixed** | `predictor_info.json` → `bundle_source: {'source': 'bundle', 'bundle_sha': 'd2b2175f529f923c'}`. The test checks the sha against the file on disk. |
| 8 | Disk variant leaks the probe | **fixed** | It now calls the shared `probe_cached_prefill_tokens`. The test only checks the delegation, through a mock (see nit 4). |
| 9 | Agg only in `vidur.main` | **partly** | Agg is now set in `conftest.py` and the three a11 spike scripts. It is not set in `evolve/sim_fork.py`, the library that future planners import. See minor 2. |
| 10 | Test gaps (idempotence, observe timing, pool guard) | **fixed** | mtime-and-bytes snapshot; per-request `first_token`→`end` order with `SIM._time` equality; free-list order, cached-hash keys and ref counts. |
| 11 | observe `ctx.now`, counters, negative knobs | **fixed** | `ctx.now` is the event time during `observe` and is restored afterwards. `observe_calls` and `observe_seconds` are counted separately. A negative R gives `needs_clock = False`. |

Note on #3: the round-1 policy (`min(waiting+running, kv_usage)`, using `choose`) still gives the same `request_metrics.csv` sha for R = 0, 0.25 and 1.0 (`d4e214bd8124`), although its views differ. This is not a defect. `waiting` and `running` are constant within a burst for every R, because requests are added to replicas after `schedule()` returns. The stale argmin therefore picks the same replica as the fresh one for all 16 bursts of that slice. `test_view_refresh_below_burst_gap_changes_outcomes` uses a kv-first policy on the same slice and does show different outcomes. A campaign should not read "same sha" as "the knob is inert" without also checking the views.

## New semantics (task 2)

**`route_delay_s` (`route_delivery_event.py`, `global_schedule_event.py`)**
- One `RouteDeliveryEvent(t+d, mapping)` is created per routing call. Its priority is `(t+d, GLOBAL_SCHEDULE, id)`. At t+d it therefore runs after arrivals (2), batch-stage ends, batch ends, prefill ends and request ends (3–6), and before replica scheduling (8–9), which is the zero-delay slot. The docstring's claim holds.
- Ties with other same-slot events are broken by id. A `GlobalScheduleEvent` at t+d can only be created by an arrival handled at t+d (`request_arrival_event.py` is its only creator). Its id is therefore always larger than that of a delivery created at t. A delivery always runs before a same-instant routing call, deterministically.
- Arrival exactly at a delivery time (r2 case b and c: bursts at 1.0, 1.5 and 2.0 with d = 0.5, so burst A is delivered at 1.5 just as burst B arrives). Every request is shifted by exactly 0.5, and the scheduling delay is `{0.5}`. With R=0, the router at 1.5 sees A delivered but not yet scheduled: views `(waiting=4, running=0, outstanding=4)`. That follows from the ordering above, and it is a sensible convention: the replica has received the requests by the time the router acts at the same instant.
- Under uniform delay, two deliveries never coincide unless their arrivals did. Rounding of `t+d` is monotone, so the order of bursts is preserved.
- Fork: r4 used `route_delay_s=0.05` and `view_refresh_s=0.4` with an evolved policy on Conv v4 (80 requests, 4 replicas). It paused at event > 200 with a `RouteDeliveryEvent` pending (t = 1.5, polled tick 3), forked, and ran both copies to the end. Straight run, original after the fork, and the fork all gave sha `7e45acf0ac0a`. The event holds a list of `(ReplicaId, Request)` pairs and a set; the deepcopy memo keeps request identity consistent with the scheduler's queues. `sim_fork.at_decision` tests `isinstance(..., GlobalScheduleEvent)`, and `RouteDeliveryEvent` is not a subclass, so deliveries are not counted as decisions.

**`view_refresh_s` (`on_clock` in `Simulator._set_time`)**
- *Which state does a tick hold?* The state just before kR: every event at a time < kR, and none at time kR. `on_clock(t)` runs before the first event at t is handled, and later events at the same t do not re-snapshot, because k is unchanged. When no event falls on kR, the snapshot is taken at the first event after kR. Replica state does not change between events (nothing outside `handle_event` mutates it; `get_next_request` touches only the generator), so that snapshot equals the state at kR. A tick is therefore captured exactly at kR whether or not an event falls there. This matches the `on_clock` docstring ("a tick that falls exactly on an event time is polled before that event"), the config help text and `test_view_refresh_views_are_snapshots_of_the_last_grid_tick`.
- Evidence of consistency (route_delay 0.5, bursts at 1.0/1.5/2.0, 1 replica). The views of the routing call at 1.5 are: R=0 `(4,0,4)`, R=0.5 `(0,0,4)`, R=0.4999999 `(0,0,4)`. A tick just before 1.5 and a tick exactly at 1.5 give the same snapshot, so the convention is continuous in R for R > 0. See nit 1 for the R → 0+ edge.
- *R = 0*: `needs_clock = False`, so `Simulator._clock_hook = False`. The cost is one bool test per event, and `_replica_views` keeps the old `_poll()` on every build. The golden check confirms byte identity.
- *R > 0 cost*: one `_tick_index` (a floor and two multiplications) per event, plus one `_poll` per tick that is crossed. This is negligible.
- *What is stale*: only `waiting`, `running` and `kv_usage`. `outstanding_requests`, `pending_prefill_tokens`, `inflight_prefix_tokens`, `active_prefix_tokens` and `session_last_replica` are router-owned and exact, which is correct. `kv_capacity_tokens` is static. **`cached_prefix_tokens` is a live probe of each replica's actual cache at decision time** (see minor 1).
- `_tick_index` defines a tick as the largest k with `k*R <= t` in float arithmetic. For example, at t = 0.15 with R = 0.05, tick 3 is not reached, because `3*0.05 = 0.15000000000000002`. The test uses the same rule, so this is a definition, not a bug.

**Stuck collection, atomic writes, idempotence, `main.py`.** The collector lists exactly what `vllm_v1` and disk `is_empty()` count (global queue, waiting and running). A run that `is_empty()` reports as non-empty therefore always produces a non-empty list, which gives exit 3. Requests still in a `RouteDeliveryEvent` cannot be stranded, because the pending event keeps the loop running. JSON is serialised fully before `tmp`+`os.replace`, and the `finally` removes a leftover tmp. `_output_written` is set only after every file is in place. `test_failed_end_state_write_leaves_no_partial_file` covers the retry. The `main.py` ordering is verified in r5 above.

**Shared probe.** The body of `probe_cached_prefill_tokens` is token-for-token the body of the old `VLLMV1ReplicaScheduler.get_cached_prefill_length`. The only other change to `vllm_v1_replica_scheduler.py` is the removed `dataclasses` import, which nothing else in that file uses. The vllm_v1 path is unchanged, and the golden set agrees. Disk path: that scheduler calls `get_computed_blocks` itself for every waiting request before `allocate_slots_with_disk_cache`. The hash list the old probe leaked is recomputed deterministically there, and `prefix_cache_stats` is never read for outputs. The disk path's outputs should therefore also be unchanged. This is reasoned from the code, not run: the golden set has no disk cell.

**observe.** `ctx.now` is set to `completed_at` or `first_token_at`, which equal the simulator clock at the hook. The test asserts `now == SIM._time` for all 2n calls. It is restored in a `finally`. `policy_calls` and `policy_seconds` now count routing calls only. `LearnedGlobalScheduler` increments them itself and defines no `observe`, so it is unaffected. No `evolve/` policy defines `observe`, so the default path is unchanged.

## Tests (task 4)

The branch passes `tests/test_v41_sim_fixes.py` and `tests/test_v4_robustness.py`: 19 passed in 131 s.
**Counterfactual:** I extracted `a3ef5ef`'s `vidur/` into a scratch directory with `git archive`, with the same data and cache symlinks, and ran the branch's test files against it. **All 11 tests in `test_v41_sim_fixes.py` fail on the old code**, each on its own property:
- the stuck-path TypeError;
- the partial `stuck_requests.json`;
- `request_e2e_time` off by more than d (both delays) and the split batch;
- the snapshot mismatch `(5.54, 110)`;
- identical outcomes at R=1.0;
- `needs_clock`;
- the `ctx.now` assertion;
- `bundle_source`;
- the missing probe helper.

The 8 tests in `test_v4_robustness.py` pass on both versions. That is expected: f16fc4a only strengthened tests of code that was already correct, and the round-1 analysis established that each strengthened test now constrains its property.

---

### [minor] Under `view_refresh_s > 0`, cache-aware policies still read live replica cache contents
`vidur/scheduler/global_scheduler/evolved_global_scheduler.py:162-178` (`_cached_prefix_tokens`), `:25-27` (module docstring)

**Scenario.** A staleness experiment (R > 0) compares a cache-aware policy, for example mooncake_min_ttft or anything that reads `RequestView.cached_prefix_tokens`, with load-only policies. Only the load fields are stale. The probe reads the replica's block pool at decision time, so it reflects every eviction and every block cached by the most recent prefill. This is information a polled router would not have. Real routers either track an approximate index updated by KV events with lag (SGLang's router, Dynamo's KV router, Mooncake's conductor) or poll it. The behaviour is documented: the help text says "Router-owned counters and the prefix probe stay exact". It is still an advantage for cache-aware policies that grows with R, and a paper sentence such as "views refreshed every R" would overstate the realism.

**Evidence.** By construction: `_poll` covers only `(waiting, running, kv_usage)`, and `_cached_prefix_tokens` is computed per burst from `kv.block_pool.get_cached_block`.

**Fix.** At minimum, state it explicitly where staleness results are reported. Optionally, give the probe the same grid: probe against a block-hash set snapshotted at each tick. That would be per replica, a copy of the `cached_block_hash_to_block` keys at each tick, created only when R > 0, so the default path keeps its event sequence. Hashed: **yes** if code changes; no for the paper wording.

### [minor] Agg backend still not set where in-process simulators are built as a library
`evolve/sim_fork.py` (round-1 #9 residual)

f16fc4a sets `MPLBACKEND=Agg` in `tests/conftest.py` and the three a11 spike scripts. Any new planner or headroom script that imports `evolve.sim_fork` and calls `SimHandle.write_output()` → `ClusterMetricsStore.plot` as a background process on macOS can still hit the AppKit hang. **Fix:** add `os.environ.setdefault("MPLBACKEND", "Agg")` at the top of `evolve/sim_fork.py`, before it imports `vidur`. That file is not hashed. Alternatively set it in `vidur/metrics/cluster_metrics_store.py`, which is hashed, at the next version bump. Hashed: no (sim_fork) or yes (vidur).

### [nit] R → 0+ does not converge to R = 0 at exact coincidences
With R > 0, a decision at t sees the state *before* every event at t. With R = 0 it sees the state *after* the same-instant batch, request and route-delivery events that sort before routing. The evidence above shows this: at 1.5, R=0 gives `waiting=4` and R=0.5 gives `waiting=0`. Each convention is self-consistent and documented. The difference shows up only when an arrival lands exactly on a replica-side event time, which happens in practice only with `route_delay_s` equal to a multiple of the burst gap. Worth one line in the knob's help text. Hashed: yes if reworded there.

### [nit] The tick-on-event convention is not under test
`test_view_refresh_views_are_snapshots_of_the_last_grid_tick` encodes "state before the first event ≥ kR". With R = 0.05 on the Synth slice at scale 20, no replica event lands exactly on a tick, so an implementation that snapshots *after* the events at kR would also pass. A three-line case like the one above (delivery at 1.5, R = 0.5, assert `waiting == 0` at 1.5) would pin the convention down. Hashed: no.

### [nit] Stale comment in an old test
`tests/test_v4_robustness.py:172` says "polled once at the first decision". The snapshot is now taken at the first event, the first arrival, which still gives one distinct view. Hashed: no.

### [nit] Disk-probe test is delegation-only
`test_disk_scheduler_uses_the_side_effect_free_probe` checks that the helper is called, not that a real disk scheduler is left without state. That is acceptable, because the helper itself is covered by `test_router_probe_leaves_no_state` on vllm_v1. Hashed: no.

### Carry-overs outside chunk A (for chunk B's owner)
- `SimHandle.write_output` (`evolve/sim_fork.py:257`) still calls the private `_write_output`. Forked runs therefore never write `predictor_info.json` or `stuck_requests.json`, and a stranded fork does not surface as one.
- `evaluate.run_one` still fingerprints the bundle file in `cache/` at scoring time and ignores the new `bundle_source.bundle_sha` from the run itself. Using the run's own sha would close the gap that round-1 #7 was about.

### Side effect of the golden run (housekeeping)
Running the golden check rewrote a *tracked* bytecode file in the worktree: `evolve/policies/systems/__pycache__/mooncake_min_ttft.cpython-310.pyc` now shows as `M`. Any import of that policy does the same, the fixer's own golden run included. It is harmless, but the `.pyc` should be untracked (`git rm --cached` plus a `.gitignore` entry) so that golden runs leave the tree clean. I did not revert it, because the checkout is read-only for this review.
