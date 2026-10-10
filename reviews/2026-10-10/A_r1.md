# Chunk A (simulator v4 changes), round 1

**Verdict: changes requested.** The default-path changes (probe, diagnostics skip, observe hook, zero-delay and zero-refresh paths) are behaviour-preserving as claimed.
The stuck-request path that 0272b19 advertises crashes before it writes anything. The two v4 knobs (`route_delay_s`, `view_refresh_s`) do not model what their help text says on the bursty Tool/Conv traces.
No finding changes default-path numbers or invalidates cached baselines. Every fix touches `vidur/` (hashed), so it bumps `code_version`.

Scope: `git diff 1823e99..HEAD -- vidur tests/test_v4_robustness.py tests/conftest.py`, excluding `learned_global_scheduler.py`.
`tests/test_v4_robustness.py` passes (7 passed, 59 s). Repro scripts are under the session scratchpad; the commands and their output are quoted below.

## Focus questions, answered

- **`get_cached_prefill_length` side-effect-free: yes, fully.** `KVCacheManager.get_computed_blocks` changes only two things: `req_to_block_hashes` (a defaultdict insertion) and `prefix_cache_stats`. `BlockPool.get_cached_block` only reads: no touch, no LRU or free-queue change, and `in` on the defaultdict inserts nothing. The new code pops the inserted entry when `had` is false and restores a copy of the stats. When the request's own replica probes it again before scheduling, the hash list is recomputed from `block_hash_ids`, which is deterministic and gives the same content. `prefix_cache_stats` is never read for outputs: the hit ratio comes from the request histograms in `replica_metrics_store.get_prefix_cache_stats`. Removing the top-K diagnostics scan (`_collect_topk_lookahead_budget_stats`) touches only the same two structures, so it is also behaviour-preserving. The disk variant was left unfixed (minor, below).
- **`route_delay_s = 0`**: the branch is taken only when `delay > 0`. No event is created and no id is consumed. `ROUTE_DELIVERY = 10` is appended to the enum, so no existing priority shifts. Byte-identical as claimed.
- **`view_refresh_s = 0`**: it calls `_poll()` on every view build and reads the same three fields as before. Exact as claimed.
- **observe timing**: PREFILL_END (5) sorts before REQUEST_END (6) at the same time, so `first_token` precedes `end` even when both fall in one batch. `first_token_at` is the first prefill completion (`request.py:365` keeps it across restarts). `num_cached_tokens` is frozen at the first schedule. The `_prefilled` guard fires `first_token` once after a preemption restart. The hook is a no-op when the policy has no `observe`, and no policy in `evolve/` defines one.
- **`write_output()`**: it is idempotent and is called once on the normal exit path, before the checkpoint save and before exit 3. If `run()` raises, nothing is written; the comment says this is deliberate. Not idempotent-safe on failure: the flag is set before writing (nit). See findings 1 and 5.

---

### [major] Stuck-request path crashes: `_collect_stuck_requests` iterates a non-iterable queue, then the JSON dump fails on `ReplicaId`
`vidur/simulator.py:154` and `vidur/simulator.py:178-180`

**Failure scenario.** A run ends with requests still pending, which is the exact case 0272b19 adds `stuck_requests.json` and exit 3 for. `run()` calls `_collect_stuck_requests()`, which does `for r in rs._waiting_queue`. `FCFSRequestQueue` and `EDFRequestQueue` define neither `__iter__` nor `__getitem__`; they expose `to_list()`. That raises `TypeError` for every replica, even one with an empty waiting queue. The process exits 1 with a traceback. No metrics, no `predictor_info.json` and no `stuck_requests.json` are written. For a learned scheduler, `save_final_checkpoint` never runs either; in v3 an atexit writer plus the checkpoint save would both have run. With that fixed, the second bug appears: `json.dump(self.stuck_requests)` fails because `"replica": rid` is a frozen-dataclass `ReplicaId`. By then `_output_written` is already true and `stuck_requests.json` has been opened for writing, so a truncated file is left behind.

The harness still fails such a run (`returncode != 0`), so no wrong number reaches a score. But the advertised end-state file and exit code 3 never happen, and the diagnosis is lost.

**Evidence.** Repro: wedge one replica with `can_schedule = lambda: False` on a 6-request Synth slice.
```
  File ".../vidur/simulator.py", line 154, in _collect_stuck_requests
    out += [{"where": "waiting", "replica": rid, "request_id": r.id} for r in rs._waiting_queue]
TypeError: 'FCFSRequestQueue' object is not iterable
output written flag: False | stuck file: []
```
After monkeypatching the collector to use `to_list()`, `sim.write_output()` fails with:
```
TypeError: Object of type ReplicaId is not JSON serializable
```
**Not caught by tests.** `test_write_output_records_predictor_provenance` only asserts that the file is absent.

**Fix.** Use `rs._waiting_queue.to_list()` and `"replica": rid.id`. Add a test that wedges a replica as above, then asserts that `stuck_requests.json` parses and that `vidur.main` exits 3 (or check `sim.stuck_requests` plus the file). Hashed: **yes** (`vidur/simulator.py`).

### [major] `route_delay_s > 0` un-batches same-timestamp bursts: not a pure hop delay, and Tool/Conv arrive only in bursts
`vidur/events/route_delivery_event.py:21-26`, `vidur/events/global_schedule_event.py:34-39`, `vidur/types/event_type.py:16`

**Failure scenario.** With zero delay, a burst routed in one `GlobalScheduleEvent` is added to the replicas first, then one `ReplicaScheduleEvent` runs per replica, so the burst can share a batch. With a delay, there is one `RouteDeliveryEvent` per request, and each returns its own `ReplicaScheduleEvent` at the same time. `REPLICA_SCHEDULE = 8` sorts before `ROUTE_DELIVERY = 10`, so the replica schedules after the first delivery alone. It forms a batch containing only that request, and the rest of the burst waits for that batch to end. The extra latency mixes a batching artifact into the hop latency.

How common this is: in the v4 traces, **100 %** of Tool and Conv requests share their timestamp with another request (median burst 20 and 10 requests; bursts every ~3 s). For Synth it is 0.6 %.

**Evidence.** 4 requests at t=1.0, 64-token prompts, 1 replica, round_robin.
```
prefill_e2e_time delay0:  [0.0188, 0.0188, 0.0188, 0.0188]      # one batch
prefill_e2e_time delay.5: [0.5103, 0.5286, 0.5286, 0.5286]
diff:                     [0.4915, 0.5097, 0.5097, 0.5097]      # pure shift would be 0.5 each
request_e2e_time diff:    [0.5001, 0.5096, 0.5096, 0.5096]
```
`test_route_delay_shifts_every_request` misses this for two reasons. It uses the Synth slice, where there are almost no bursts. It also asserts only a lower bound (`late >= base + 0.5`).

**Fix.** Deliver a whole mapping with one event: `RouteDeliveryEvent(t + delay, mapping)` adds every request and then returns one `ReplicaScheduleEvent` per distinct replica, mirroring the zero-delay path. Test it with a same-timestamp burst and assert an exact shift (`late == base + delay` within 1e-9) at light load. The zero-delay path stays untouched, so the golden set is unaffected. No campaign passes `route_delay_s` today (grep of `evolve/` and `scripts/`). Hashed: **yes**.

### [major] `view_refresh_s` polls lazily at decision times, so a period shorter than the burst gap gives no staleness at all
`vidur/scheduler/global_scheduler/evolved_global_scheduler.py:206-212`

**Failure scenario.** The help text says the metrics "are polled every this many seconds". The code re-polls at the first decision at least `view_refresh_s` after the previous poll, and it uses `ctx.now`, which is that decision's arrival time. So a refreshed view is always perfectly fresh at the decision that triggers it. A real periodic poller on a fixed grid would show an age of `t mod R`, uniform in [0, R). Tool and Conv arrive in bursts every 3 s × scale (≥ 1.5 s at scale 0.5). For any `R` ≤ the gap, every burst re-polls and sees exact state, so the knob silently does nothing. For larger `R`, staleness is still understated, because every refresh resets the age to zero exactly at a decision.

**Evidence.** Conv v4 slice: 120 requests, 16 bursts, gaps 1.5 s after scale 0.5, 4 replicas. The policy routes on `waiting + running` and `kv_usage`.
```
refresh 0  : d26af095d407   (sha of request_metrics.csv)
refresh 1.0: d26af095d407
```
`test_view_refresh_freezes_polled_metrics` tests only the degenerate `1e9` (one poll ever), so it cannot see this.

**Fix.** Poll on a fixed grid. On a view build at time `t`, return a snapshot taken at `floor(t / R) * R`. That needs a snapshot event, created only when `R > 0`, so zero-refresh runs keep their event sequence. Alternatively, a cheap approximation is to keep the last snapshot taken at or before the grid point. At minimum, rename the knob and reword its help text to "minimum re-poll interval at decision times", and say in the paper that it understates staleness. Add a test with `0 < R <` burst gap that expects outputs to differ from `R = 0`. Hashed: **yes**.

### [major] `test_trace_block_size_must_match_cache` passes with the validator deleted
`tests/test_v4_robustness.py:16-22`

**Failure scenario.** The test overwrites `block_size` with 32 but leaves hash ids that were computed at 16. `Request.__init__` (`request.py:49-56`) already asserts `last_block_size in [0, block_size)` when the first request is generated in `_init_event_queue`. The test accepts `AssertionError`, so it passes whether or not `_validate_block_size` exists. The case KV-10 is about is a trace that is internally consistent at 32 tokens replayed on 16-token cache blocks. That case passes the Request assertion and is not tested.

**Evidence.** With `Simulator._validate_block_size = lambda self: None`:
```
raised AssertionError -40442 is not in the range [0, 32)
```
**Fix.** Build a trace hashed consistently at 32: halve the id list, or keep `len(ids) == ceil(total / 32)` and run with `--cache_config_block_size 16`. Then require `ValueError` with "does not match" in the message. Hashed: no (tests only).

### [minor] `main.py` now writes outputs before saving the RL checkpoint, so an output failure loses the checkpoint
`vidur/main.py:20-25`

In v3 the order was `run()`, then `save_final_checkpoint()`, then the atexit writer. Now `write_output()` comes first and raises on failure, for example a plotting or disk error, or the `ReplicaId` JSON error in finding 1. The learned network of a long training run is then never saved. **Fix:** call `save()` before `write_output()`, or wrap each in try/finally so both run and the process still exits non-zero. Hashed: **yes** (`vidur/main.py`).

### [minor] `clamp_hits` counts only lookups beyond the prediction grid, not the profiled range its comment names
`vidur/execution_time_predictor/sklearn_execution_time_predictor.py:45`, `:835`

The comment says "Lookups outside the profiled table, clamped to its edge". The table is the forest prediction grid, which reaches `prediction_max_tokens_per_request` (262,144 by default). The profiling data can end far below that: `a100/meta-llama/Llama-2-7b-hf/attention.csv` stops at kv_cache_size 4032. A Llama-2 run with 30k-token prompts reports `clamp_hits == {}` even though every long-context attention time is a flat forest extrapolation. Reading `clamp_hits == {}` as "no out-of-profile predictions" would be wrong. **Fix:** reword the comment and docstring to "outside the prediction grid". Optionally, also record per-table profiled maxima in `provenance()` so a reader can compare. Hashed: **yes**.

### [minor] `provenance()["bundle_key"]` does not say whether the bundle was actually loaded
`sklearn_execution_time_predictor.py:835-848`

The key is a hash of the *inputs*. In `IGNORE_CACHE` mode, or when the bundle is missing at load time, the predictions are recomputed but the same key is reported. The harness (`evolve/evaluate.py:bundle_fingerprint`) then hashes whatever file sits in `cache/` at scoring time. In `IGNORE_CACHE` mode it attributes a table the run never used. The harness uses `USE_CACHE`, so this is not live. **Fix:** in `_predict_from_models`, record `{"source": "bundle" | "computed", "bundle_sha": <sha of the loaded or written file>}` and report it from `provenance()`. Hashed: **yes**.

### [minor] GS-15 not applied to the disk variant
`vidur/scheduler/replica_scheduler/vllm_v1_disk_replica_scheduler.py:61-63`

The disk variant still calls `get_computed_blocks` directly, so it leaks one hash list per probed but unchosen replica and bumps the stats. This does not affect outcomes, for the same reasons as on the main path; it is only a memory leak in long disk-cache runs. **Fix:** move the probe into a shared helper. Hashed: **yes**.

### [minor] Matplotlib Agg is forced only in `python -m vidur.main`; in-process users still get the GUI backend
`vidur/main.py:6`

`evolve/sim_fork.py`, `evolve/rl_plan/headroom.py` and the tests import `vidur.simulator` directly, and plots are produced from `ClusterMetricsStore.plot`. The AppKit hang that cafde65 fixes can still happen when one of these runs as a background process on macOS. **Fix:** set the backend at the top of `vidur/metrics/cluster_metrics_store.py`, or wherever pyplot is first imported. Hashed: **yes**.

Related, in chunk B's area: `SimHandle.write_output` (`evolve/sim_fork.py:259`) calls the private `_write_output`. Forked rollouts therefore never write `predictor_info.json` or `stuck_requests.json`. With deepcopy forks the predictor is shared, so `clamp_hits` accumulates across forks in one process.

### [minor] Remaining test gaps
`tests/test_v4_robustness.py`

- `test_write_output_records_predictor_provenance`: the second `write_output()` call ("idempotent") asserts nothing. A non-idempotent writer that rewrites the files would pass. Compare mtimes or contents before and after the second call.
- `test_observe_hook_reports_first_token_then_end`: it counts events but checks neither per-request order (`first_token` before `end`) nor timing. A hook fired late would still pass, because `first_token_at` is read from the request. Also record `sim._time` (or the event time) when each outcome is observed, and assert it equals `first_token_at` or `completed_at`.
- `test_router_probe_leaves_no_state` checks only the hash dict and the stats. That is sufficient today, since the block pool is read-only on this path, but a cheap guard against future regressions is a snapshot of `block_pool.free_block_queue` order and `cached_block_hash_to_block` keys.
- Nothing tests the stuck path, the burst case for route delay, or a non-degenerate refresh (findings 1-3).

### [nit] Observe-side details
`evolved_global_scheduler.py:319-331`

During `observe`, `ctx.now` is still the last *routing* time, not the time of the event being observed. A learner that time-stamps with `ctx.now` will be wrong; document that, or set `ctx.now` to the event time before the call and restore it after. `observe` calls are counted in `policy_calls` and `policy_seconds`, which inflates per-decision cost figures (used in `evolve/spikes/a11`). Negative `route_delay_s` or `view_refresh_s` values are silently treated as 0. Hashed: yes if changed.
