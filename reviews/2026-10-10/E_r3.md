# E r3: simulator fork (`evolve/sim_fork.py`, `tests/test_sim_fork.py`), fix 1ad201a

**Verdict: changes requested (one major, small, in the non-hashed `evolve/sim_fork.py`).**
Every E_r2 failure now raises or is exact. Exactness holds on a new pair (claude_r1, Conv p1), and a forced decision under `policy_state="independent"` is byte-identical to the self-chosen counterfactual.
The major: the per-replica check behind `"independent"` looks only at top-level values. It accepts the repo's own Mooncake/Dynamo burst state and any `{session: replica}` map. The minors are holes outside `ctx.state`: ctx attributes, a window in the fingerprint check, and `os.fork` inside a callback.

## E_r2 findings, re-checked against 1ad201a

| E_r2 finding | Status | Evidence (my repro) |
|---|---|---|
| SGLang/AIBrix "unchanged state = exact" | **now raises** | Real run, `Synth 4R s1.0 w150+300 p0`. For both policies, decision 253 (policy chose 1, request 252, hashes already in `idx[1]`) is the same point E_r2 found. Forcing 2 gives `NotImplementedError: cannot force replica 2 (policy chose 1) exactly: ctx.state (keys ['sglang_index'])…` (and `['aibrix_index']` for AIBrix). |
| module state beyond the five container types | **fixed for the 10 toy cases** | Each of the 10 cases is refused by the static scan, and its fingerprint changes after one call. `pytest -q tests/test_sim_fork.py -k "refused or fingerprint or reconcile or fifo or module_scan"`: 37 passed. Remaining holes are in minor 3. |
| FIFO commit inferred from key names; three wrong commits accepted | **fixed** | uncached-entry, conditional-append and pick-dependent-trim variants of tail4_ni are not recognised by the source check and raise. `_fifo_move` now checks each kept length against `min(len, outstanding)` from the views. Tuple entries return False instead of `TypeError`. |
| `route()` bursts accepted partial forcing | **fixed** | Raises when the override is placed with a burst of more than 1 request. It also raises at call time, because `_forcing_call` re-checks `len(args[0]) > 1`, which covers arrivals that join the burst after the override was placed. `route_first_only=True` books `[forced] + rest` (test). |
| `"independent"` not checked | **partly**: see the major | |
| RL comment "hashed" | **fixed** | `sim_fork.py:418` now says it is not hashed. |

## Verified OK (no finding)
- **Exactness on a new pair, with `python -m vidur.main` as the reference.** claude_r1_76be082eb0f5 on `Conv 4R s1.0 w3000+300 p1` (300 decisions).
  - Forks at decisions 60 and 180, stepped alternately with the original, one decision each. `request_metrics.csv`, `token_gaps.npz` and `predictor_info.json` of O, F1 and F2 are all byte-identical to the `vidur.main` run.
  - **Forced vs self-chosen with `policy_state="independent"`** at decision 60: the policy chose 0 and I forced 1. The comparison run patched `choose` once to return 1 after its full computation.
  - After the decision, `ctx.state`, `_assigned`, `_outstanding`, `_pending_prefill`, `_inflight`, `_active`, `_session_last` and `ctx.rng` are equal in both runs. The final outputs are byte-identical, and they differ from the reference run, so the forced decision did take effect.
- **claude_r1 `sessions` really is independent of the pick.** It stores `(arrived_at, n)` per session. Pruning is by time. The pick-dependent fact (where the session went) is `request.session_last_replica`, which the scheduler ledger (`_session_last`) books with the executed replica. The byte-identical forced-vs-self result above confirms this end to end.
  - completion_r2 (`session_prompt[sid] = n`, `scarcity_ewma` from the views' waits) and t4_70b_8b92 (`prompt_ewma` from `n`) pass the rule and are independent of the pick. t4_70b_81e keeps no state (empty-state path). An early-decision false refusal is noted in nit 9.
- **Record-only API with `os.fork`, called outside `run()`** (l3_48a, `Conv 4R s1.5 w6000+300 p0`, decision 10).
  - Four children each forced a replica in turn, one child alive at a time. Each child's `record` gives the same `policy_choice` (1) as the parent's later record-only decision, and each child booked its forced replica.
  - The child forced to 1 ends the decision with exactly the parent's state: `ctx.state`, ledger, `ctx.rng`, id counters and global RNG.
  - Placing the override right before the decision in the parent works.
- **Module scan on the repo's policies and archived candidates.**
  - 302 router files (finalists, policies, `runs/*/programs`, `candidates_*`, probes, excluding RL designs): 300 pass the static scan.
  - The 2 refusals (`runs/l3_t4_r1/programs/29ef4e990072.py`, `398d265b4648.py`) are harmless false positives: `REF_FACTOR = _work_factor(0.0, PROMPT_REF)`.
  - Forking is therefore usable for essentially every candidate (forcing coverage: nit 6).
- **Fingerprint cost:** 0.06–0.09 ms per fork (fingerprint plus the cached scan, which includes a file read and sha256) on claude_r1, l3_48a and completion_r2. This is negligible next to the deepcopy.
- **Source recognition of the l3/tail4 shape.** Anything that changes the queues beyond trim-then-append-`n`-at-the-pick is caught by `_fifo_move`. That includes aliases (`q = queues[pick]; q.append(…)`), passing `queues` to a helper, or rebinding `_attended_context`/`choose` with a module-level name assignment. Any other `ctx.state` key is caught by `set(after) == _FIFO_KEYS`. The only false accepts are outside `ctx.state` (minor 2) and the `FORCED_COMMIT` declaration (nit 7).

---

### [major] The `"independent"` check is shallow: it accepts the repo's Mooncake/Dynamo burst state and every map whose value is the pick
`evolve/sim_fork.py:736-741` (`_per_replica`), `755-762`, and the docstring at 668-671.

**Failure scenario:** the docstring presents the check as the safeguard: "refused when the state holds anything per replica …, which is where a pick-dependent update lives". `_per_replica` looks only at **top-level** values, and only for a sequence of exactly `num_replicas` entries or a dict whose keys are all replica indices.
- **Nested structures pass.** `mooncake_min_ttft` and `dynamo_kv_cost` (10 files in all, counting the tuned variants and copies) store `(now, [set() per replica])`, a 2-tuple, and run `idx[pick].update(...)`. With N ≠ 2, `"independent"` is accepted.
  - The rest of the burst then sees the hashes on `chosen` instead of `forced`, the same silent error as E_r2's SGLang case.
- **Maps keyed by request or session pass.** A session-affinity map `{sid: pick}` has the pick as its value, not its key, so it passes. So do a `last_pick` scalar, a pick-history deque, a global FIFO `[(replica, n), …]`, and `{"q": [per replica]}`.

No caller passes `"independent"` today. But 132 of 302 candidates keep `ctx.state` that neither the FIFO commit nor the empty-state path covers (nit 6). A planner that wants to cover them will pass `"independent"` wholesale, and then this check is the only guard.

**Evidence** (`_reconcile(..., chosen=1, forced=0, mode="independent")`, N=4):
```
mooncake_burst (now,[sets]) N=4          -> ACCEPTED
dynamo_burst N=4                         -> ACCEPTED
session affinity {sid: pick}             -> ACCEPTED
last_pick scalar                         -> ACCEPTED
nested per-replica {'q': [..N..]}        -> ACCEPTED
global FIFO [(replica, n)]               -> ACCEPTED
pick history deque(maxlen=16)            -> ACCEPTED
```

**Suggested fix:**
1. **Make the check recursive** (tuples, lists, dicts, sets, depth-limited) so that a length-N sequence or a replica-keyed dict at any depth refuses. This catches Mooncake, Dynamo and nested FIFOs.
2. **Do not rely on structure for maps whose value is the pick.** Structure cannot catch them. Make `"independent"` an allowlist instead of a free assertion: a dict in sim_fork from source sha256 to name, filled only with the reviewed policies (claude_r1, completion_r2, t4_70b_8b92). Anything else raises with "define `on_forced()`".
3. **Narrow the docstring claim** to what is actually checked.

**Hashed file:** no.

### [minor] State kept on `ctx` attributes (not `ctx.state`) bypasses all three automatic paths
`evolve/sim_fork.py:712` (only `ctx.state` is snapshotted), `763-766`, `_fifo_source_ok:833-835`.

**Failure scenario:** `PolicyContext` is a plain object. A policy can write `ctx.last_pick = pick`. That attribute is deep-copied per fork, which is fine for (a) and (b), but no check sees it under forcing:
- `_reconcile` compares only `ctx.state`;
- the static scan treats `ctx` as a local parameter;
- the FIFO recognition rejects only the attribute `.state`.

So a pick-dependent ctx attribute passes the empty-state path, the FIFO path and the `"independent"` path. After the forced decision it holds the policy's choice, not the executed replica.

**Evidence:**
```
round-robin via ctx.last_pick: static issues [] ; empty-state path: ACCEPTED (ctx.last_pick = chosen 1, executed 0)
tail4_ni + "ctx.last_pick = pick" before the tail: fifo-recognised = True ; static issues = []
```
Today 0 of 302 candidates write a ctx attribute. They all use `ctx.state`, `ctx.rng` and `ctx.now` only.

**Suggested fix:** in `_forcing_call`, snapshot `{k: v for k, v in vars(ctx).items() if k not in ("rng", "state")}` together with `before`. In `_reconcile`, refuse when `vars(ctx)` holds any key outside `{rng, num_replicas, now, state}`. This costs one dict comparison.

**Hashed file:** no.

### [minor] The static scan misses stores through `cls`/aliases, and the fingerprint is compared only at `fork()`, so forks taken before a lazily-started mutation are silently contaminated
`evolve/sim_fork.py:269-285` (stores are flagged only when the root name is module-level and not local), `430-433` (fingerprint compared only in `_unforkable_reason`).

**Failure scenario:**
- `class _T: n = 0` with `@classmethod def bump(cls): cls.n += 1` passes the static scan, because `cls` is a parameter. So do `c = _C; c.n += 1` and `f = choose; f.__dict__[...]`. The class body is a constant, so the scan is satisfied.
- The fingerprint catches the change, but only at the next `fork()`. If the module starts mutating after the fork point (lazy, threshold-triggered or session-triggered), the fork is allowed. Parent and fork then share and advance the same class attribute, and nothing raises until someone forks again.
- So the rollouts of the last fork point before the mutation are wrong silently. A one-shot planner (hindsight oracle, a single forced-vs-self comparison) never sees an error.

**Evidence** (toy policy `pols/lazy_cls.py`: lowest pending prefill, plus a deviation on the first 5 `cls` bumps after t > 60; `Conv 4R s1.5 w6000+300 p0`; fork at decision 5, t = 18):
```
static scan issues: []
O vs unforked reference identical: True
F vs unforked reference identical: False      <- no raise
later fork: policy module ... changed its module-level state since the handle was built ...
```
0 of 302 candidates use these idioms (no `@classmethod` state, no aliases). The classmethod-counter idiom is plausible in LLM-written code.

**Suggested fix:**
- **Close the window.** Compare the fingerprint again when `run()` returns and in `write_output()`, for every handle of a build that has been forked (keep a shared `forked` flag in the family). On a mismatch, raise: "module state changed during a run; this handle and every handle forked from the same build are invalid". At 0.07 ms per call this is free.
- **Flag the idiom statically.** In the scan, also flag stores into attributes of the first parameter of methods of module-level classes, and stores into attributes of local names bound to a module-level name.

**Hashed file:** no.

### [minor] `os.fork` taken inside a `run()` callback: CPython reseeds the global `random` in the child, so the child's global RNG differs from the unforked run
`evolve/sim_fork.py:684-685` (docstring: "works unchanged in an os.fork child"), `469-473` (a live handle uses the live global state).

**Failure scenario:**
- `random` registers `os.register_at_fork(after_in_child=_inst.seed)`, so every `os.fork` child gets a fresh, urandom-seeded global `random`.
- Outside `run()` this is harmless: the child's `active()` reinstalls `h.rng_state`.
- Inside a `run(until=...)` callback the handle is live, so `active()` is re-entrant and nothing reinstalls the state.

**Evidence** (l3_48a, os.fork inside the callback at decision 20; the child runs the decision record-only; the parent runs the same decision; component by component):
```
state True  assigned True  outstanding True  pp True  sess True  ctxrng True  counters True  grng False  time True
```
Outside `run()`, the same comparison is fully equal (Verified OK).

**Impact today: none.**
- In a v4 evolved run, no code draws from global `random`: the schedulers use their own `random.Random` instances, and 0 of 302 candidates import `random` or `np.random`.
- `np.random` and torch are not reseeded at fork.
- `evolve/rl_plan/headroom.py` forks outside `run()`.

**Suggested fix:** in sim_fork, `os.register_at_fork(before=..., after_in_child=...)`: if `_LIVE` is non-empty, save `random.getstate()` before the fork and restore it in the child. Alternatively, document "os.fork only outside run()".

**Hashed file:** no.

### [nit] A pending override cannot be cancelled; a record-only override placed before `os.fork` blocks every child
`evolve/sim_fork.py:691-692`.

**Evidence:** after `override_next_decision(h, None, record={})` and then `os.fork`, the child's own override raises `RuntimeError: an override is already pending on this handle`. Callers must place the parent's record-only override after spawning the children (headroom does).

**Suggested fix:** state the required order in the docstring, or add `cancel_override(h)`.

**Hashed file:** no.

### [nit] Forcing coverage of existing candidates (usability, not a bug)
The categories are mutually exclusive. Across 302 router files:
- 118 are FIFO-recognised (exact);
- 50 keep no `ctx.state` (exact);
- 132 keep other `ctx.state` and raise unless they define `on_forced` or pass `"independent"`;
- 2 are refused by the static scan.

Among the 132:
- 44 are FIFO policies with pick-independent extras (`band_ema`/`band_t`, `cong_avg`, `prompt_ref`) or a second FIFO (`pf_queues`). `"independent"` refuses them too, because `attn_queues` is per replica.
- 10 are Mooncake/Dynamo copies (see the major).

63 files contain `queues[pick].append(n)` but are not recognised, mostly because of a different tail or extra state. This is a safe failure. If the planner needs these candidates, one route is a FIFO commit that tolerates extra keys left unchanged by the move. It would have to be combined with the allowlist from the major for those extras.

**Hashed file:** no.

### [nit] `FORCED_COMMIT = "attn_fifo"` trusts the entry semantics
`evolve/sim_fork.py:779-780`.

**Failure scenario:** a policy that declares it but appends the uncached length passes `_fifo_move` whenever `cached[pick] == 0`. The entry `n` is then moved where `n - cached[forced]` belongs.

**Evidence:**
```
accepted; queues after move: [[500], [700, 1000]]   counterfactual: 400 at replica 1
```
No policy declares `FORCED_COMMIT`.

**Suggested fix:** document that the declaration asserts the entry is `num_prefill_tokens`.

**Hashed file:** no.

### [nit] `"independent"` false refusal at the first decisions of full-trace runs
`evolve/sim_fork.py:740`.

**Failure scenario:** session ids start at 0 in all three v4 traces, and the full traces are used by STAGE1 and held-out. While claude_r1's `sessions` (or completion_r2's `session_prompt`) holds only ids < N, it looks like a replica-keyed dict, and forcing raises. For example `{0: …, 1: …}` with N=4 gives `raised`. This is a safe failure. The allowlist from the major removes it.

**Hashed file:** no.

### [nit] The module source is re-read from disk at every fork and every forced decision
`evolve/sim_fork.py:339-346`.

**Failure scenario:** if the policy file is replaced after the build, the scan and FIFO recognition examine code that is not the loaded module. If it is deleted, they refuse.

**Suggested fix:** read the source and its sha once in `SimHandle.__init__` (it is shared through the memo), and refuse if the file's sha later differs.

**Hashed file:** no.

## Reproduction scripts
Scratchpad of this review (not committed):
- `unit.py`: the `"independent"` cases, ctx attributes, static-scan holes, cost, `FORCED_COMMIT`;
- `partA.py`: SGLang/AIBrix on a real run;
- `partB.py`: exactness and forced vs self-chosen for claude_r1;
- `partC.py` and `partC2.py`: `os.fork` with record-only and forced overrides;
- `partD.py` with `pols/lazy_cls.py`: the fingerprint window.

At most two simulator processes ran at a time (parent plus one `os.fork` child, or in-process plus one `vidur.main`).
