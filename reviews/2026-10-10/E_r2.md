# E r2: simulator fork (`evolve/sim_fork.py`, `tests/test_sim_fork.py`), fix bc0b369

**Verdict: changes requested (two majors, both small, both in the non-hashed `evolve/sim_fork.py`).**
Every round-1 finding is fixed or now raises, and exactness holds on two new policy/trace pairs against a `vidur.main` reference. The forced decision is exact for the l3/tail4 FIFO finalists.
Two of the new guards can still pass silently. First, "ctx.state unchanged by the call, so the forced decision is exact" is wrong for the SGLang and AIBrix system policies; this is verified on a real run. Second, the stateful-module check sees only top-level dict/list/set/deque/bytearray globals.

## Round-1 findings, re-checked against bc0b369

| r1 finding | Status | Evidence (my repro, `Conv 4R s1.5 w6000+200 p1` unless stated) |
|---|---|---|
| override then fork corrupts the parent | **fixed** | fork's `_call` is bound to the fork's scheduler (`__self__ is parent gs`: False). After the fork's forced decision, the parent's `_assigned` is unchanged, the parent's override is still pending and the fork's is consumed. The fork runs 2 more decisions without a crash. The parent later applies its own override. |
| fork inside a `run` callback | **fixed** | callback fork at decision 49 has counters `Request 49, BaseEvent 24820`, not the build-time values. Parent and fork outputs are byte-identical to an unforked run (request_metrics, token_gaps, predictor_info). |
| learned (RL) runs | **now raises** | a real `LearnedGlobalScheduler` (rl_loto_r2): `fork()` and `override_next_decision()` both raise `NotImplementedError` with a clear message. |
| policy memory after a forced decision | **fixed for the FIFO finalists** | see "Verified OK" (exact against a self-chosen counterfactual). My mean/p99 numbers for r1's decision-12 case (forced 2: 10.7725/28.4006) are not comparable with r1's (the v4 trace changed in between), so I relied on the byte-for-byte check against the self-chosen run instead. |
| lazy counters leak between handles | **fixed** | two handles with batch metrics on: `Batch` counter 2103 / 2103 at decision 20, and no `_id` left on the class afterwards. |
| test coverage (minor) | **fixed** | bursty p0 window, two fork times, alternating steps, callback fork, and forced vs self-chosen with ledger, state and rng compared. `pytest -q tests/test_sim_fork.py`: 5 passed, 83 s. |
| stuck requests, predictor_info, clamp_hits (minor) | **fixed** | fork output contains `predictor_info.json`. `run()` fills `stuck_requests` the way `Simulator.run` does. clamp_hits is per handle (the test probes it; my windows had 0 clamps). |
| output dir and range check (nit) | **fixed** | `write_output` on a fork without `output_dir` raises `RuntimeError`. A choice of -1 or 4 raises `PolicyError`. |

## Verified OK (no finding)
- **Exactness on new pairs, with `python -m vidur.main` as the reference** (a stronger reference than the test's in-process run). Compared files: `request_metrics.csv`, `token_gaps.npz`, `predictor_info.json`.
  - `systems/mooncake_min_ttft` on `Tool 4R s1.0 w2000+300 p0` (21 decisions, bursts of 19 and 6 at the forks). Forks at decisions 5 and 12, stepped alternately, plus a third fork taken inside a callback in the middle of the event stream (decision 15, not at a decision). O, F1, F2 and F3 are all byte-identical.
  - `completion_r2_c8177f73cf22` (uses `ctx.rng.choice`) on `Synth 4R s1.0 w1000+300 p1`. Forks at decisions 60 and 150, plus a callback fork. All byte-identical.
- **Forced vs self-chosen, tail4_ni** (a finalist the test does not use), `Tool w2000+300 p0`, decision 2. The burst has 12 requests; the policy picked 0 and I forced 2. The call trimmed the FIFO of **both** replicas. After the call, `ctx.state`, the ledger (`_assigned`, `_outstanding`, `_pending_prefill`, `_inflight`, `_session_last`) and `ctx.rng` equal those of the run where `_select` itself returned 2. The final outputs are byte-identical.
- **Why the FIFO commit is exact for the finalists:** every l3/tail4 finalist trims with `while len(q) > r.outstanding_requests: popleft()` before it picks, so the trim does not depend on the pick. It then appends exactly `n` at the pick. All 182 files with `queues[pick].append(n)` share this shape. The 40 candidates that append `(u, f)` tuples carry extra state keys (`pf_queues`, …), so they raise.
- **The `policy_state="independent"` claims are true** (I read the code):
  - claude_r1 stores `sessions[sid] = (arrived_at, n)`; its pruning is time-based.
  - completion_r2 stores `session_prompt[sid] = n`, and `scarcity_ewma` comes from the views' waits.
  - t4_70b_8b92 updates `prompt_ewma` from `n`.
  - t4_70b_81e keeps no state.
- A forced decision on a `choose()` policy with a multi-request burst is exact: the rest of the burst sees views that already include the forced assignment.
- `learned_global_scheduler.py` is in `_VERSION_EXCLUDE` (`evolve/evaluate.py:338-345`), so it is **not** hashed (see the last finding).

---

### [major] "ctx.state unchanged by the call" is taken as exact, but the SGLang and AIBrix policies leave it unchanged at their own pick and would change it at the forced one
`evolve/sim_fork.py:464-465` (`if after == before: return`), and the docstring at 407.

**Failure scenario:**
- `systems/sglang_cache_aware.py` and `systems/aibrix_prefix_load.py` keep a route-time prefix index and run `idx[pick].update(request.prefix_hashes)`.
- When every hash of the request is already in `idx[chosen]`, the call changes nothing. This happens when the same prefix was routed there before, and that replica is exactly where these policies send such a request.
- `_reconcile` then returns "exact" and the forced replica is booked.
- Had the policy picked the forced replica `a` itself, `idx[a]` would contain those hashes.
- Every later decision in the rollout is made with an index that matches neither world. Nothing raises, which breaks the contract ("anything it cannot guarantee must raise").

**Evidence** (`Synth 4R s1.0 w150+300 p0`, SGLang, first decision where the first request's hashes ⊆ `idx[chosen]`):
```
decision (253, chosen 1, forced 0, request 252)
forced call raised nothing; routed to 0 | hashes in idx[forced] after: False | state unchanged by call: True
forced-as-implemented vs state-consistent: requests on a different replica: 0 | p99 E2E 46.8955 46.8955
```
- **Frequency:** requests whose whole hash set was already seen earlier number 10 in 6000 (Tool), 6 in 6000 (Conv) and 23 in 3993 (Synth).
- **Effect on outcomes:** in this window none, because no later request reused the prefix. The effect is rare and usually small, but silent.
- **Other policies:** Mooncake and Dynamo start a fresh per-instant dict, so their first call always changes state and raises (verified for Mooncake: keys are not FIFO, so `NotImplementedError`).

**Suggested fix:** accept "unchanged" as exact only when the state is empty both before and after the call (lor, deepseek_r1, t4_70b_81e and the floor policies). Otherwise require `on_forced` or `policy_state="independent"`. An `on_forced` for SGLang/AIBrix is a single line (`idx[forced].update(request.prefix_hashes)`), if those policies should be plannable.

**Hashed file:** no.

### [major] The stateful-module refusal sees only top-level dict/list/set/deque/bytearray, and misses the usual ways LLM-written code keeps state
`evolve/sim_fork.py:192-210`

**Failure scenario:** the policy module is shared by every fork (memo, line 338). Any state the module keeps outside `ctx.state` is shared, so forks overwrite each other's state and the parent's.

`isinstance(v, _MUTABLE)` on `vars(module)` does not catch:
- an int or float rebound with `global` (`_CALLS += 1`);
- an `np.ndarray`;
- a module-level `random.Random`;
- a `SimpleNamespace` or instance of a custom class;
- a mutable class attribute;
- a closure cell;
- a `functools.lru_cache` (its cache);
- a mutable default argument;
- function attributes (`choose.cache = {}`).

The docstring claims "a module with mutable globals is refused by `fork`". In practice, `fork` raises only for the five container types. Evolved candidates are written by an LLM and evaluated in a fresh process each time, so nothing in the pipeline discourages these idioms. Today none of them appear: I grepped for `global`, `lru_cache` and `random.Random(` across finalists, policies, candidates, probes and runs, and found 0 hits.

**Evidence** (toy module with all of the above, mutated on every call):
```
toy module refused? None
```

**Suggested fix:** make the check conservative.
- **Option 1, AST pass at fork time:** refuse when the module has `global`/`nonlocal` statements, cache decorators, mutable defaults, or module-level assignments whose value is not a literal constant or tuple.
- **Option 2, fingerprint:** at `build_simulator`, fingerprint the module's reachable non-code state: globals other than modules; functions' `__defaults__`, `__kwdefaults__`, closure cells, `__dict__` and `cache_info()`; classes' non-callable `__dict__` entries; `ndarray` bytes; `Random.getstate()`. Refuse in `fork()` if the fingerprint changed.

The false positive (refusing a constant dict) is safe.

**Hashed file:** no.

### [minor] The FIFO commit infers its semantics from the two key names, and its shape check accepts three changes it cannot reconcile
`evolve/sim_fork.py:466, 474-498`

**Failure scenario:** any policy whose `ctx.state` keys are exactly `{attn_queues, attn_sums}` gets the move. `_fifo_move` checks only that each queue is a suffix of its old contents and that `n` was appended at `chosen`. It does not check how much was trimmed, or that the entry is the prompt length.

**Evidence** (unit repros on `_fifo_move`):
```
uncached-entry FIFO accepted: True -> [[500], [700, 1000]]  | counterfactual would append 200 at replica 1
no-append call (q[chosen] already ends with n) accepted: True [[], [1000]]   (an outstanding request's entry moved)
pick-dependent trim accepted: True [[400], [600, 700, 1000]]
tuple FIFO raised TypeError unsupported operand type(s) for +: 'int' and 'tuple'
```
- **Entry depends on the replica.** If the entry is the uncached length (`n - cached[pick]`) and `cached[pick] == 0`, the entry equals `n`. The move puts `n` at the forced replica instead of `n - cached[forced]`.
- **Conditional append.** A policy that appends only sometimes is misread when `q[chosen]` already ends with `n`.
- **Pick-dependent trim.** Trimming that depends on the pick is accepted as if it came from the views.
- **Tuple entries.** A tuple FIFO under these exact keys raises `TypeError` instead of the clean `NotImplementedError`.

None of the current finalists reach these paths (see "Verified OK").

**Suggested fix:**
- Check each kept length against the views passed to the call: `len(trimmed_i) == min(len(old_i), replicas[i].outstanding_requests)`, using `args[1]`.
- Prefer an explicit opt-in declared by the policy (for example a module constant `FORCED_COMMIT = "attn_fifo"`) over inferring from key names.
- Wrap the sum comparison so that it returns False on a type error.

**Hashed file:** no.

### [minor] `route()` policies: a forced first request leaves the rest of the burst routed for the policy's own first choice, and this does not raise
`evolve/sim_fork.py:418-419, 444-454`

**Failure scenario:** with a burst of more than one request and `chosen != forced`, the policy picked requests 2..k with its own first choice in mind. The result is a mixed action, not the counterfactual of (c). The behaviour is disclosed, but the contract asks for a raise. No `route()` policy exists in the repo today (0 in finalists, policies, candidates and runs).

**Suggested fix:** raise when `is_route and len(args[0]) > 1 and chosen != forced` (unless an `on_forced` hook returns the remaining choices).

**Hashed file:** no.

### [nit] `policy_state="independent"` is not checked
`evolve/sim_fork.py:413-415, 450`

**Failure scenario:** passing `"independent"` for an l3/tail4 FIFO policy or an index policy silently skips reconciliation.

**Suggested fix:** refuse `"independent"` when the state contains a key that sim_fork knows depends on the pick (`attn_queues`, `*_index`, `*_burst`).

**Hashed file:** no.

### [nit] A comment calls `learned_global_scheduler.py` hashed, and what an exact RL fork would need
`evolve/sim_fork.py:198-201`

**The comment:** "The last needs a hook in learned_global_scheduler.py (hashed)" is wrong. The file is in `_VERSION_EXCLUDE`, so editing it invalidates no cached baseline. RL manifests do record the learner's sha (review F_r1), so an edit is still traceable.

**What an exact RL fork would need** (from reading the scheduler):
1. **A forced-action hook in `LearnedGlobalScheduler.schedule`,** applied after `_act` and before `_record`/`_assign`. `schedule` does not go through `_call`, so the current `_call` patch would never fire for it.
   - The hook must record the forced `a` in the `_Decision` and in `_routed` (which drives `cached[a]` and the reward).
   - It must also update `_stats["actions"]` and `_who`.
   - It must decide whether the learner's `_rng` draws (teacher mix, epsilon) are consumed as they are now. "Policy chose a" has no meaning for an epsilon draw, so define the forced action as replacing the executed action after all draws.
2. **Design-module state** (`rl_loto_*`: `_PEND`, `_ORDER`, `_BASE`) must become per handle: move it into the scheduler or ctx, or re-exec the module per fork and deep-copy its globals. `CONFIG` is a constant dict and trips the current check as a false positive.
3. **Inference mode** (`not _training`): the network is read-only, so it can be shared. Copy `_routed`, `_ctx`, the teacher ctx and its rng, and `_prev`.
4. **Training mode:**
   - deep-copy the network, target network, Adam state, replay buffer (`_buffer`, `_buf_pos`, `_buf_sid`), `_n_decisions` and `_n_steps` per fork;
   - make the torch global RNG per handle (done already);
   - redirect or disable `checkpoint_out`, the logs and the trace writes per fork, or every fork writes the same checkpoint files.
5. **Memo:** the memo must pre-seed `_torch` and `_teacher` (modules).

**Hashed file:** no (comment). The RL hook would not be hashed either.

### Carry-over, outside this diff (no action required for E)
- **A11 lookahead outputs are stale.** `evolve/spikes/a11/out/look_*.json` (02:39) predate the fix and still carry the policy-memory confound from r1 finding 4. They were not regenerated. Treat their gain and rank figures as provisional until they are re-run.
- **The lookahead spike picks the burst's first request without sorting.** `spike_a11_lookahead.child_rollout` takes `_request_queue[0]` as the burst's first request, but `schedule()` sorts the queue only later. This is safe only for bursts of one (p ≥ 1).
- **`evolve/rl_plan/headroom.py` keeps its own forcing.** It patches `_assign` and calls `commit_policy_state` instead of `override_next_decision`. Its commit checks only `q[pick][-1] == n`, without the trim-shape check, and its closure binds the scheduler (its docstring discloses this). It should migrate to `override_next_decision` so that there is one audited mechanism.
