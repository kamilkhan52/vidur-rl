# E r4: simulator fork (`evolve/sim_fork.py`, `tests/test_sim_fork.py`), fix 5c36f7b

**Verdict: approve with minors (no blocking or major finding).**
Every E_r3 finding is fixed or now raises. The planned rollout policy l3_48a passes everything on a new real run: forks at two points are byte-identical to `vidur.main`, forced decisions equal self-chosen ones, and a 120-point forcing sweep had 0 raises.
Two minors remain, both in the non-hashed `evolve/sim_fork.py`. First, module state the fingerprint cannot see (a nested class mutated through `self` or a parameter) still lets a fork diverge silently. Second, the new alias rule refuses harmless policies, e.g. `ctx.state["k"] = CONST`.

## E_r3 findings, re-checked against 5c36f7b

| E_r3 finding | Status | Evidence (my repro, `e4/unit4.py` unless noted) |
|---|---|---|
| [major] shallow `"independent"` check | **fixed** | All 6 E_r3 shapes now raise: Mooncake `(now,[sets])`, `{sid: pick}`, last-pick scalar, nested `{'q': [N]}`, global FIFO, pick-history deque.<br>- With an unlisted policy: "accepted only for reviewed policies".<br>- With claude_r1 (listed) and the same state change: "the call's ctx.state change involves the pick". |
| [minor] ctx-attribute state | **fixed** | Round-robin via `ctx.last_pick` on the empty-state path: `raised: … keeps state on ctx attributes`. |
| [minor] cls/alias stores; fork before a lazy mutation silently diverged | **fixed for fingerprint-visible state** | - `classmethod cls.x`, `c = _C; c.n += 1` and `os.environ` alias are now refused by the static scan.<br>- `f = choose; f.__dict__.setdefault(...)` and `_bump(_C)` (store through a helper parameter) still pass the scan. Their fingerprint changes, so the run end, `write_output`, the next `fork` and forcing all raise. `test_module_state_change_after_fork_is_caught` passes.<br>- State the fingerprint does not cover still diverges silently: see minor 1. |
| [minor] `os.fork` inside `run()` reseeds global `random` | **fixed** | E_r3's `partC2` rerun (`e4/partC2.py`, l3_48a, `Conv 4R s1.5 w6000+300 p0`, child inside the callback at decision 20, no explicit helper call): see the os.fork section below. `test_os_fork_inside_run_callback_keeps_global_rng` passes. numpy/torch are not reseeded by CPython at fork; the hook restores all three (a no-op for those two). |
| [nit] no cancel | **fixed** | `cancel_override` returns `[True, False]` on two calls (test). |
| [nit] source re-read from disk | **fixed** | `_module_source` pins the text at the first read, which `SimHandle.__init__` triggers right after the build. If the file is edited afterwards, the scan still sees the loaded code (`source pinned: True`). |
| [nit] `"independent"` false refusal on early session ids | **fixed** | Replaced by the allowlist (test covers `{0..3: …}`). |

## Verified OK (no finding)
- **Exactness on a new pair, l3_48a (the planned π_e) on `Tool 8R s0.75 w0+600 p1`, l3_8b** (`e4/partE.py`, 600 decisions, reference `python -m vidur.main`):
  - Forks at decisions 150 and 420, stepped alternately one decision each with O and four forced/self-chosen handles: `request_metrics.csv`, `token_gaps.npz` and `predictor_info.json` of O, F1 and F2 are byte-identical to `vidur.main`.
  - **Forced vs self-chosen** (auto mode, so the FIFO commit is used), at both points (decision 150: chose 4, forced 5; decision 420: chose 5, forced 6). The counterfactual is a separate module instance whose `_select` returns the forced replica after running in full.
    - After the decision, `ctx.state`, the ledger (`_assigned`, `_outstanding`, `_pending_prefill`, `_inflight`, `_active`, `_session_last`) and `ctx.rng` are equal.
    - The final outputs are byte-identical, and they differ from `vidur.main`, so the forced decision did take effect.
  - **Forcing sweep:** a fork at every 5th decision (120 points) forced a rotating replica. That gave 104 real reconciliations and 16 points where the forced replica was already the policy's pick, with **0 raises**. The family was marked forked throughout, and `_check_module_state` never fired on l3_48a.
  - So **sim_fork is usable for the planned rollouts with π_e = l3_48a** (persistent forks and forced root decisions). headroom's `os.fork` path does not set `family["forked"]`, so only the forcing-time fingerprint check applies there; l3_48a passes it.
- **Allowlist by source sha:**
  - The three entries equal the sha256 of `evolve/finalists/{claude_r1_76be082eb0f5, completion_r2_c8177f73cf22, t4_70b_8b92be1e362d}.py`.
  - Every copy under `evolve/runs/*/programs/` has the same sha. The program id is the sha prefix.
  - `*_archive.jsonl` records hold no code, so there is no "loaded from the archive" path.
  - A CRLF copy is accepted, because `read_text` normalises newlines and the code is the same. A copy with one extra trailing newline is refused, which is a safe failure.
  - No false refusal for the intended three; see the full-trace check below.
- **Guard false refusals on the listed policies across full traces** (`e4/partG.py`). At every routing call, `_pick_dependent_change(before, after, pick)` and the ctx-extras check ran on the real state change:
  ```
  claude_r1      7b-4R-Synth-s1.0-full-p0       calls=3982 guard refusals=0 ctx-extras=0
  completion_r2  7b-4R-Synth-s1.0-full-p0       calls=3982 guard refusals=0 ctx-extras=0
  t4_70b_8b92    7b-4R-Synth-s1.0-full-p0       calls=3982 guard refusals=0 ctx-extras=0
  completion_r2  l3_8b-8R-Conv-s1.0-w0+3000-p1  calls=3000 guard refusals=0 ctx-extras=0
  claude_r1      l3_8b-8R-Tool-s1.0-w0+3000-p1  calls=3000 guard refusals=0 ctx-extras=0
  ```
  The only pick-equal ints these policies write are prompt lengths (`session_prompt[sid] = n`, `sessions[sid] = (t, n)`). The v4 minimum prompt is 13 tokens (Synth) and 891 (Tool/Conv), so a false refusal needs ≥ 14 replicas.
- **Fingerprint re-check timing.** Each run segment starts with the build fingerprint: `fork()` compares it, and every `run()` of a forked family ends with a compare. A run's output therefore depends on module state only through a deterministic function of the build state and its own inputs.
  - A mutation that is visible to the fingerprint either raises at the end of the segment or is undone within it. Undoing it within the segment cannot leak, because the unforked run would undo it the same way.
  - So with full fingerprint coverage, the check timing is sufficient. The residual hole is coverage (minor 1), not timing.
  - Reads made inside an `until` callback happen before the end-of-run check, but the run then raises, so the result is not silent.
- **Module scan on 304 router files** (finalists, policies, `runs/*/programs`, candidates, probes; RL designs excluded): 300 pass.
  - Refused: E_r3's two harmless `REF_FACTOR = _work_factor(...)` cases, plus `evolve/proposer.py` (not a router).
  - One new false refusal comes from the alias rule (minor 2).
- **Targeted tests:** `pytest -q tests/test_sim_fork.py -k "pick_dependent or reconcile or refused or fingerprint or module_scan or fifo"`: 42 passed. With `-k "module_state_change or os_fork or sglang or forced_decisions"`: 5 passed.

---

### [minor] 1. Module state invisible to the fingerprint (nested class, method `__dict__`, callable class attribute), mutated through `self` or a parameter: forks diverge silently, nothing ever raises
`evolve/sim_fork.py:408-411` (classes: only non-callable attributes are fingerprinted, so nested classes and methods are skipped), `:285-302` (stores are flagged only through a module-level name, an alias, `cls` or `type(...)`/`__class__`).

**Failure scenario:**
- A store rooted at `self` or at a function parameter passes the static scan.
- If the object it reaches is a nested class, the `__dict__` of a method, or a callable instance installed on a class, the fingerprint never changes.
- So fork, run end, `write_output` and forcing all accept, and the fork shares and advances the state with its parent. This breaks contract (b), and "anything not guaranteed must raise".

**Evidence:**
- `e4/unit4.py`: three calls each, static scan result and whether the fingerprint changed:

```
nested class via self (instance per call)      static=passes   fp changes=False  outs=[1, 2, 3]
nested class via helper parameter              static=passes   fp changes=False  outs=[1, 2, 3]
method __dict__ via self                       static=passes   fp changes=False  outs=[1, 2, 3]
callable class attr installed via param        static=passes   fp changes=False  outs=[1, 2, 3]
```
- Real run (`e4/partF.py`, `e4/pols/lazy_nested.py`): the E_r3 `lazy_cls` toy with `self.Count.n += 1` on a nested class, on `Conv 4R s1.5 w6000+300 p0`, forked at decision 5 (t = 18):

```
static scan issues: []
no raise                                   <- O.run, O.write_output, F.run, F.write_output
O vs unforked reference identical: True
F vs unforked reference identical: False
later fork allowed: True
```

**Impact today: none.** Of 304 router files, 1 defines a module-level class and 0 define a nested one. The idiom is less plausible for LLM-written code than the `cls` counter E_r3 found, but the failure is silent.

**Suggested fix:** make the fingerprint walk recursively everything the module defines. That means every type or function reachable from module globals whose `__module__` is the policy's name, including nested classes, and for each:
- `vars()` of classes, including callable non-function attributes, via `vars()` or pickle;
- `__dict__`, `__defaults__` and closure cells of methods, as is already done for module-level functions.

This keeps the "raise at the next check" guarantee for anything reachable. Optionally the scan could also flag attribute stores of depth ≥ 2 rooted at `self` or a parameter (`self.X.y = …`).

**Hashed file:** no.

### [minor] 2. The new alias rule marks the root of attribute/item targets as shared: harmless policies are refused, with a misleading message
`evolve/sim_fork.py:277-282`.

**Failure scenario:** `shared |= {x.id for t in tg for x in ast.walk(t) if isinstance(x, ast.Name)}` collects every name inside the target, not only plain-name targets. Any store of a bare module-level name or attribute into an item or attribute therefore marks the container's root as "shared", and then every store through that root is flagged. Examples:
- `costs[i] = INELIGIBLE`;
- `ctx.state["ewma"] = INIT`;
- `d["k"] = math.pi`.

The policy is then unforkable (`fork` raises) and unforceable (`_reconcile` raises), with "stores into module-level 'ctx'".

**Evidence:**
- `e4/unit4.py`:

```
local list item = module const   static issues: ["choose: stores into module-level 'costs' (costs[0])"]
ctx.state[k] = module const      static issues: ["choose: stores into module-level 'ctx' (ctx.state['n'])", "choose: stores into module-level 'ctx' (ctx.state['ewma'])"]
local dict d['k'] = math.pi      static issues: ["choose: stores into module-level 'd' (d['k'])", "choose: stores into module-level 'd' (d['j'])"]
```
- Real candidate newly refused by 5c36f7b: `evolve/runs/t4_local_r3/programs/33c06df4220e.py` (`costs[i] = INELIGIBLE`, line 190).
- `ctx.state[k] = CONST` is a natural way for an LLM to initialise state. Today 1 of 304 files is hit, and l3_48a is not.

**Suggested fix:** collect only plain-name targets. For an `Assign`, take the targets that are `ast.Name`, and the `Name`s of `Tuple`/`List`/`Starred` targets, but not names nested in `Attribute`/`Subscript` targets.

**Hashed file:** no.

### [nit] 3. The pick-recording guard checks Python `int` values only
`evolve/sim_fork.py:821-851`, docstring at 736-738.

The allowlist is the real guard, and the three listed policies were reviewed. Each case below is a single call on claude_r1 with pick = 1:
- **Accepted (pick-dependent, missed):**
  - `np.int64(1)`, `"1"` or `1.0` as a value;
  - a pick-keyed dict (`{1: 4} -> {1: 5}`) or a replica-keyed dict changed at the pick;
  - an immutable per-replica tuple rebuilt (`(0,0,0,0) -> (0,7,0,0)`);
  - `pick*1000+n`.
- **Refused (pick-independent):** a call counter going `0 -> 1` when the pick is 1.

This matters only if a listed policy is edited, which changes its sha, so it is refused anyway, or if the review was wrong.

**Suggested fix:** document it as best effort: "Python ints and mutable sequences only; dict keys are not checked". Optionally also treat `numbers.Integral` values, and dicts whose keys are all replica indices, like sequences.

**Hashed file:** no.

### [nit] 4. `_SOURCES` pins every loaded policy module for the life of the process
`evolve/sim_fork.py:356-373`.

Measured: 26.6 KiB retained per build of l3_48a (500 builds). This is harmless for planners that build per scenario. It is worth a `weakref.WeakKeyDictionary` (or storing the source on the handle) if an RL loop builds a simulator per episode in one process (10^4 builds ≈ 260 MB).

**Hashed file:** no.

### [nit] 5. Usability for planned rollouts beyond π_e = l3_48a
- **A different rollout base on a fork has no supported path.** Rebinding `gs._policy` on a fork (e.g. root decision under π_e, then a heuristic base) makes the next run end raise "changed its module-level state during a run" for the whole family, because `policy_fp` is the build module's.
  - The updated `test_forced_decisions` has to reset `c.family` and `c.policy_fp` by hand to do it (`tests/test_sim_fork.py:150-152`).
  - A small `set_policy(h, path)` that loads a module and rebinds `policy_fp` and `family` would make the plan's "parallel rollout over Mooncake/Dynamo/LOR/π_e bases" explicit.
- **Mooncake/Dynamo as the simulator's own policy cannot take a forced root decision.** Their burst index raises, which is correct. LOR as the built-in `lor` scheduler is not an evolved scheduler, so `override_next_decision` refuses it.
  - An `on_forced` in `evolve/policies/systems/mooncake_min_ttft.py` and `dynamo_kv_cost.py` would cover them. It must move the request's hashes only if they were new at `chosen`.
  - Those files are not hashed, and the baseline keys use the scheduler name.
- **Coverage for future evolved π_e is unchanged from E_r3:** about 44 % of candidates (pick-independent extras) raise on forcing until reviewed into the allowlist or given `on_forced`. This is a safe failure, but each new π_e needs a review step.

**Hashed file:** no.

## os.fork re-check (E_r3 partC2, rerun)
The child is `os.fork`ed inside the `run()` callback at decision 20. It does not call `after_os_fork`; only the `register_at_fork` hook applies. It runs the decision record-only, and the parent runs the same decision. Each component compared between the two:
```
state True  assigned True  outstanding True  pp True  sess True  ctxrng True  counters True  grng True  time True  decisions True  queue_len True
```
In E_r3 `grng` was False; it is now True. (The counter "diff" line the script prints is `MISSING` vs `MISSING`. These are distinct unpickled sentinels, not a difference.)

Semantics of `after_os_fork`: `_AT_FORK[0]` persists in the child. If a child that has already run a few steps calls the helper again, the global RNG is rewound to the fork-time state. The docstring says "first thing in the child", so this is fine as documented. No v4 code draws from the global `random`.

## Reproduction scripts
In the scratchpad of this review (`…/scratchpad/e4/`, not committed):
- `unit4.py`: E_r3 shapes, guard false accepts and refusals, ctx attributes, scan holes, alias false refusals, source pin, sha variants;
- `scan.py`: the 304-file scan and allowlist copies;
- `partE.py`: l3_48a exactness, forced vs self-chosen and the forcing sweep;
- `partF.py` with `pols/lazy_nested.py`: the silent nested-class leak;
- `partG.py`: the guard on full traces;
- `partC2.py`: global RNG after `os.fork` inside a callback.

At most two simulator processes ran at a time: in-process plus `vidur.main`, or parent plus one `os.fork` child.
