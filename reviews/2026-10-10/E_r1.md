# E r1: simulator fork (`evolve/sim_fork.py`, `tests/test_sim_fork.py`, `evolve/spikes/a11/`)

**Verdict: changes requested.** The central claim holds on the path that was tested. A fork of an evolved-policy run, continued without intervention, gives byte-identical `request_metrics.csv` and `token_gaps.npz`. This holds for sequential and interleaved forks, with single-request and multi-request bursts, and with prefix caching hitting on 196 of 200 requests. My repros confirm it.
Four ways of using the API break exactness or independence: forcing a decision and then forking; forking from inside a `run(until=...)` callback; forking a learned (RL) run; and id counters for classes that first appear after `build_simulator`. A forced decision also leaves the policy's own memory (`ctx.state`) recording the replica the policy picked, not the forced one, which confounds the lookahead numbers in the spike.
None of the fixes needs to touch a `code_version`-hashed file: `evolve/` is not hashed, apart from four functions in `evaluate.py` that are not involved here.

Verified OK (no finding):
- `tests/test_sim_fork.py` passes (64 s).
- Interleaved forks taken at decisions 30 and 120, stepped alternately, and the parent all match the unforked run byte for byte.
- Bursty window (`Conv 4R s1.5 w6000+300 p0`, 33 decisions for 300 requests, bursts of 5 and 11 at the fork points): forks at 1/4 and 1/2 of the run plus the parent all match byte for byte.
- `_step` mirrors `Simulator.run`, including the arrival-vs-event `<=` tie rule. Event ids and heap order are copied, so tie-breaking is preserved.
- Per-object RNGs are deep-copied. The global `random`/`np.random` state is saved per handle and restored for the parent.
- BlockPool copy: every block's `__dict__`, the free-list prev/next links (re-pointed by `block_id`), the queue head, tail and count, and `cached_block_hash_to_block` with both its key order and its inner `{block_id: block}` order. `ref_cnt` and `_block_hash` are carried in the block dict. Nothing is missing.
- Trace arrays and per-request `block_hash_ids` are never mutated, so sharing them is safe. The policy modules of the evolved finalists keep no module-level state.

---

### [major] Forcing a decision and then forking makes the fork write into the parent's scheduler
`evolve/sim_fork.py:266-280`

**Failure scenario:** `override_next_decision(h, c)` puts the closure `once` on `h.sim._scheduler` as an instance attribute. The closure captures the parent's `gs` and its bound `orig`. `copy.deepcopy` treats functions as atomic, so a fork made after the override carries the same closure. When the fork reaches its decision, three things happen:
1. `once` deletes the patch from the *parent*.
2. It books the fork's forced assignment into the *parent's* `_assigned`, `_outstanding`, `_inflight` and session map.
3. The fork keeps its own patch, so its next assignment is forced again and crashes on `del gs._assign`.

If the next assignment came later, the parent would be silently corrupted. A planner that forces a move at a node and then expands children from that node hits this directly.

**Evidence** (`Conv 4R w200 p1`, `run_to_decision(50)`, `override_next_decision(h, 0)`, `g = h.fork()`, `g.run(until decisions>=51)`):
```
fork shares parent's patched _assign: True
fork raised: AttributeError _assign
parent _assigned has fork's request: [True]      # request 49, which the parent has not routed yet
parent _assigned before/after running fork one decision: 19 20 | parent still patched: False | fork still patched: True
```

**Suggested fix:** do not capture the scheduler in a closure. Store the pending choice as data, for example `gs._forced_next = c`, and have `once` (or a class-level wrapper installed by sim_fork) read `self`. Alternatively, make `fork()` refuse when `"_assign" in gs.__dict__`, or re-create the patch on the copy. The same applies to `capture_requests` in the spikes, which patches `g.get_next_request` with a closure over the parent's generator.

**Hashed file:** no.

### [major] A fork taken inside a `run(until=...)` callback gets stale id counters and RNG state, which duplicates request ids
`evolve/sim_fork.py:150-163, 242-253`

**Failure scenario:** `active()` copies the live class counters and global RNG state back into `self.counters` and `self.rng_state` only when the handle exits. `fork()` copies `self.counters` and `self.rng_state`. Called from inside `until`, while the handle is active, those values are the ones from when `run()` was entered.

The fork then re-issues request, event and batch ids that are already in use. This breaks event tie-breaking `(time, type, id)` and collides keys in `_assigned` and `_inflight_req`. With outputs, the run asserts at write time. A planner that reads latencies from `Request` objects instead (as the lookahead spike does) gets silently wrong results.

A planner that forks every time `until` fires is a natural way to write MCTS against this API.

**Evidence** (fork taken when `decisions == 49` inside the callback, then run to completion):
```
callback fork counters ...: {'Request': 0, 'BaseEvent': 1}     # build-time values, not those at decision 49
fork.write_output() -> AssertionError in replica_metrics_store.get_merged_df (Request Id not unique)
```

**Suggested fix:** keep a flag that says the handle is active. In `fork()`, when the flag is set, snapshot `_read_counters()` and the live RNG state instead of the cached fields. Alternatively, raise an error if `fork()` is called while any handle is active.

**Hashed file:** no.

### [major] Learned (RL) runs cannot be forked, and the RL design modules keep module-level state that a naive fix would share between forks
`evolve/sim_fork.py:217-240`; `vidur/scheduler/global_scheduler/learned_global_scheduler.py:194,208`; `evolve/finalists/rl_loto_r2_e413fe9e193c.py:90-92`, `rl_loto_synth_r1_d2cb78118623.py:88-90`

**Failure scenario:** `LearnedGlobalScheduler` holds `self._torch` (a module) and `self._teacher` (a module). Only `_policy` is pre-seeded in the memo, so `deepcopy` raises.

The module docstring and the A11 framing say this underpins RL planning, so the RL scheduler is an expected target. Three problems sit underneath:
- **Module-level state in design modules.** The `rl_loto_*` design modules keep `_PEND`, `_ORDER` and `_BASE`, which feed rewards. Sharing `_policy` across forks, as `fork()` already does for evolved policies ("stateless by contract", line 229), would let forks overwrite each other's reward bookkeeping.
- **torch RNG is not saved.** `SimHandle.rng_state` does not include torch's global RNG, which `set_seeds` seeds.
- **Learner records the wrong action.** Under `override_next_decision`, the learner stores its own action `a` in its transition, not the forced one.

**Evidence:**
```
LearnedGlobalScheduler
fork FAILED: TypeError cannot pickle 'module' object
```

**Suggested fix:**
- Pre-seed `_torch` and `_teacher` in the memo as well.
- Treat any policy module that has mutable module globals (dict, list, set or deque in `vars(mod)`) as not forkable. Either refuse to fork, or re-load the module per fork and deep-copy those globals.
- Add the torch RNG state to `rng_state` when torch is loaded.
- Document that the evolved contract (state only in `ctx.state`) is assumed, and enforce it with that check.

**Hashed file:** no. A change to the learned scheduler would be hashed, but none is needed.

### [major] After a forced decision, the policy's memory still records the replica it picked, so counterfactual rollouts are confounded (this affects the A11 lookahead numbers)
`evolve/sim_fork.py:266-280`; `evolve/spikes/a11/spike_a11_lookahead.py`

**Failure scenario:** the docstring discloses that the policy is still called, so its `ctx.state` updates are those of the unforced decision. The consequence for planning is not disclosed.

Both finalists used here keep a per-replica FIFO of prompt lengths in `ctx.state` and append to it the replica they picked (`queues[pick].append(n)`: `tail4_ni.py:189`, `l3_48a067a7df63.py:234`). In a candidate rollout forced to replica `c`, the policy thinks it sent the request to `pick`. Its attended-context estimate is too low for `c` and too high for `pick` for the request's lifetime. Every later decision in the rollout is made by this inconsistent policy.

Only the non-policy candidates are affected, so `gain_vs_policy`, `policy_rank` and "policy is best" in `look_*.json` mix the effect of the routing choice with this corruption of the policy's memory.

**Evidence:** `l3_48a067a7df63`, `Conv 4R s1.5 w6000+300 p0`, decision 12 (burst of 4, policy choice 1). Each candidate is run to the end twice: once naively, and once with the FIFO entry moved from `pick` to `c` after the forced decision.
```
forced 0: naive mean/p99 E2E 10.7842/27.9411  state-consistent 10.7842/27.9411
forced 2: naive mean/p99 E2E 10.8249/27.6913  state-consistent 10.8037/30.8554
forced 3: naive mean/p99 E2E 10.8659/27.2430  state-consistent 10.8659/27.2430
```
For candidate 2, the p99 moves by about 11% and the ranking against the other candidates changes.

**Suggested fix:**
- Give policies an optional hook that commits the decision, for example `commit(request, choice, ctx)`, separate from scoring, so that bookkeeping follows the executed choice. Alternatively, have the harness pass the executed assignment back through `observe` and require policies to keep their state from it.
- Until then, restrict lookahead to policies whose state does not record their own choices, and treat the A11 lookahead gain and rank figures as provisional.

**Hashed file:** a hook in `evolved_global_scheduler.py` would be hashed (vidur/). Restricting which policies are used would not.

### [major] Id counters for classes first used after `build_simulator` leak between handles, so Batch, BatchStage and ExecutionTime ids differ from `vidur.main`
`evolve/sim_fork.py:46-61, 150-163`

**Failure scenario:** a `BaseEntity` subclass owns a counter only after its first `generate_id()` call. `Batch`, `BatchStage` and `ExecutionTime` first appear during the run, so they are absent from the handle's counters at build time. `_write_counters` only writes the keys it has and never deletes the others. A second handle in the same process therefore continues numbering from wherever the previous handle left those classes.

This contradicts the module docstring ("a fork numbers its new objects exactly as the unforked run would", and `build_simulator` "numbers ... batches as vidur.main would"). It also applies to a fork taken before the first batch, which shares the counter with its parent.

No output we score is affected today: batch ids appear only in batch metrics and traces, which `sim_command` disables. They would appear in any run with `--metrics_config_store_batch_metrics` or with chrome or JSON traces enabled. The test cannot catch this because it compares only request metrics and token gaps.

**Evidence:**
```
A counters at build: {'Batch': None, 'BatchStage': None, 'ExecutionTime': None}
A at decision 20: {'Batch': 2399, ...}
B counters at build: {'Batch': None, ...}
B at decision 20: {'Batch': 4799, ...}   (a fresh process gives 2399)
```

**Suggested fix:** record every counter class, and mark the absent ones with a sentinel (`c.__dict__.get("_id", MISSING)`). In `_write_counters`, `delattr` the classes marked MISSING. Alternatively, in `build_simulator`, give every `BaseEntity` subclass its own `_id = -1`.

**Hashed file:** no.

### [minor] The test checks exactness at one fork time and checks the counterfactual only loosely
`tests/test_sim_fork.py:22-47`

**What is checked:** one fork point (decision 100). Forks are run one after another, never interleaved. The window is jittered (`p1`, jitter 3 s), so every burst has size 1 (200 decisions for 200 requests). The multi-request burst path and `at_decision`'s wait for all arrivals at the same instant are never exercised. The spike windows are also all bursts of size 1 (`decisions_total == rows`).

**Weak counterfactual assertion:** the only check is `G != A`. It does not check that:
- request `first` landed on the forced replica;
- requests that finished before the fork are unchanged;
- the parent is unaffected by the override. `o` finishes before `g` is forced, so the bug in the first finding cannot show up.

**Not compared:** batch metrics and end-of-run state.

**Evidence:** my interleaved and bursty repros, listed under "Verified OK", pass. The gaps are in coverage, not hidden failures on these paths.

**Suggested fix:**
- Add a `p0` window with bursts.
- Fork at two points and step the forks alternately.
- Assert the replica of `first` in G and the rows of G that finished before the fork.
- Add regression cases for the findings above: override then fork, a fork inside a callback, two handles in one process with batch metrics on, and a learned run.

**Hashed file:** no.

### [minor] `SimHandle` does not detect stuck requests or write `predictor_info.json`, and the shared predictor's `clamp_hits` adds up across forks
`evolve/sim_fork.py:198-205, 257-259, 226-228`; `vidur/execution_time_predictor/sklearn_execution_time_predictor.py:953,981`

**Failure scenario:**
- `SimHandle.write_output` calls `Simulator._write_output`, not `write_output`. Fork outputs therefore have no `predictor_info.json`, which carries the bundle key that `score()` uses to refuse mixed bundles.
- `run()` never fills `sim.stuck_requests`. A wedged fork produces metrics with no stuck signal, unlike `vidur.main`, which exits with code 3. `evolve/rl_plan/headroom.py` writes outputs through this path.
- The predictor is shared through the memo, but it is not a pure table: `clamp_hits` is incremented on every clamped lookup. Its counts therefore add up across the parent and every fork. If the provenance file is added, it would report wrong counts.

Simulated outcomes are not affected.

**Suggested fix:** in `SimHandle.write_output`, call `_collect_stuck_requests()` when the scheduler is non-empty and then `sim.write_output()`. Save and restore a per-handle copy of each predictor's `clamp_hits` in `active()`, in the same way as the counters.

**Hashed file:** no (all in sim_fork.py).

### [nit] The commit message and docstrings promise things the module does not have
- 7a322d5 says `evolve/sim_fork.py` contains "os.fork rollouts". `os.fork` appears only in the spike scripts and in `rl_plan/headroom.py`.
- The docstring calls the predictor "a pure lookup table". See `clamp_hits` above.

**Hashed file:** no.

### [nit] `fork()` without `output_dir` keeps the parent's output dir, and the forced replica is not range-checked
`evolve/sim_fork.py:242-253, 266-280`

- **Output dir:** calling `write_output()` on a fork made without `output_dir` overwrites the parent's files.
- **Range check:** `choice_for_first=-1` silently selects the last replica, because `cached[-1]` and `_replica_ids[-1]` index from the end. Use `gs._check(choice_for_first)`.

**Hashed file:** no.

### Spike scripts (`evolve/spikes/a11/`): only issues that could invalidate the reported numbers
- **Lookahead gains and ranks:** confounded by the policy-memory problem in the fourth finding.
- **Exactness and cost numbers:** these have no problem that would invalidate them. The child processes in `in_child` and in the os.fork test of the cost spike are reaped with `waitpid` after the pipe reaches EOF. A child that raises returns `{"error": ...}`, which makes the parent fail loudly with a KeyError; a child that crashes gives empty data, which makes it fail loudly with a JSONDecodeError. Nothing fails silently.
- **macOS fork safety (unverified):** the children run only pure-Python simulation. `set_seeds` imports torch, and matplotlib or seaborn print an NSXPC warning at output time, so a child that touched BLAS, OpenMP or Objective-C could deadlock or crash on macOS. Nothing in the current child path does. If this grows into a pool of rollouts, prefer in-process `fork()` (exact, as verified) or a `forkserver` start method.
- **Determinism of os.fork rollouts:** each child installs the handle's own counters and RNG through `active()`, and the parent is never stepped between candidates, so rollouts are deterministic.
