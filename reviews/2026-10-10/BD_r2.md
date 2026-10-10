# Chunks B and D, round 2 (fixes 068fd0b, 8dab76b; reviewed at HEAD 7435890)

**B (harness/traces): approve with minors.** Every accepted B_r1 finding is fixed. `code_version` is still `c16266a8ef49` in both modes, and baseline keys are unchanged against `1f9fde4`. Two minors remain: a non-JSON table value makes `code_version` silently `"unknown"`, and the bundle-fingerprint gaps (parts 3 and 4) are still open.
**D (baselines, tuner, head_to_head, launcher): changes requested (1 major).** The driver budget, head_to_head and launcher fixes are correct. The new tuner rule "retry invalid records on resume" splits the ES trajectory when the failure happened mid-run. The old trajectory's records then stay eligible for `best`, so the tuned control is picked from more than its budget, and `afterany` chain links trigger this automatically.
None of the findings below touches a `code_version`-hashed file.

Later commits (29ca349, 7435890) do not touch the chunk files. Targeted tests pass: `test_head_to_head`, `test_tune_baselines`, `test_driver_resume` and `test_v4_harness` (31 passed). Repro scripts are in this session's scratchpad (`tuner_repro*.py`, `driver_legacy.py`, `keys.py`, `cv_stub.py`).

## Round-1 findings: verification

| R1 finding | Status | How verified |
|---|---|---|
| B#1 table values outside `code_version` | **fixed** | Mutating each of KV_CAPACITY, KV_CAPACITY_DEVICE, MAX_CONTEXT, MODEL_NAME, TP, SEED_FLAG and BUCKETED changes `code_version`; restoring them gives back `c16266a8ef49`. |
| B#2 profile test was a substring check | **fixed** | The test is now behavioural (temp REPO; editing or adding a CSV changes the version). |
| B#3 v3-data provenance / keep path | **fixed** | `provenance()` now has `v3_data`, trace shas and `tables`; the keep path goes to `EVOLVE_KEEP_DIR/v3data` (nit N-B2). |
| B#4 bundle check | swap deferred per LOOP; **parts 3–4 not fixed** | See m-B2. |
| B#5 converter guard | deferred | — |
| B#6 defaults / prompt | **fixed** | Missing `--bar`/`--objective` now warn; `adapt_prompt` is tested; SELECTION.md renamed the bar. Residual: `--bar` is still silently ignored without `--protocol v2` (N-B3). |
| B#7 split test | **fixed** (disjointness) | The unit test of `expand_block_hashes_unpadded` was not added (N-B3). |
| B nits (comment, mix- `device`) | **fixed** | — |
| D#1 iterations restart per link | **fixed for new run dirs** | `test_driver_resume` (total budget, kill mid-iteration, refill of a missing slot) passes. Legacy dirs: m-D2. |
| D#2 head_to_head wrong bar | **fixed** | The bar comes from the JSON, and mismatched bars are refused. The test is the round-1 mock (true 0.8, previously reported as 1.6). |
| D#3 code_version / trace mismatch | **fixed** | Real run: `head_to_head --json evolve/analysis/out/l3_ablate_search3b.json ...` now exits with "scored under code_version c6fceae7bfa8 (current c16266a8ef49)". |
| D#4 invalid records permanent | **fixed, but introduces M-D1** | — |
| D#5 file-name reuse after a kill | **fixed** | Round-1 scenario re-run (budget 40, parallel 2, slow even candidates, kill at evaluation 10, resume): **0 file/params mismatches**, and the same 40 param sets as the uninterrupted run. |
| D#6 resume provenance / truncated line | **partly** | `meta.json` guard works. The truncated-line handling fails on the second resume (m-D1). |
| D#7 defaults-reproduce test | **fixed** | Shared ctx with bursts; asserts > 300 tie-breaks and > 1000 same-burst matches. |
| D#8 launcher | **fixed** | DRY run under fake `ssh`/`rsync`/`scp`/`sbatch` on PATH made **zero** calls. The CV check now runs before rsync; duplicate-chain and archive guards and the tuned-header check (`head -3 ... "code_version $CV"`) are in place; the comment is corrected. |
| D#9 tuner bar vs campaign bar | SELECTION.md **fixed**; tuner docstring not | N-D1 |
| D#11 install header path | **fixed** (`run.name`) | — |

## Specific checks requested

- **`code_version`** is `c16266a8ef49` with and without `EVOLVE_V3_DATA=1`. The two table digests are `6d3f26ccb7e3c8a9` / `692c66d3f1b80f93` and match `_TABLES_AT_C16266`.
- **Digest canonical across interpreters:**
  - Identical `code_version` and digest in both modes under Python 3.9.6, 3.10.20 (venv), 3.11.15 (Alpine runs 3.11.14) and 3.13.15. Tested with numpy/pandas stubbed (`cv_stub.py`); the digest uses only `json` and `inspect`.
  - Unaffected by `PYTHONHASHSEED`: `BUCKETED` is sorted, and `sort_keys` handles the dicts.
  - A CRLF copy of `evaluate.py` gives the same digest, because `inspect` reads source through universal newlines.
  - No Path objects or floats in the tables today.
  - Numpy values break it: m-B1.
- **Coverage.** Nothing material that shapes a run is outside both `code_version` and the digest:
  - Bar members are keyed by file sha, the trace by sha, bundles by fingerprint.
  - Python and library versions are recorded in provenance only, by design.
  - The one leftover is the `Scenario.warmup_frac` default (N-B1).
- **Baseline keys are unchanged.** I extracted `1f9fde4` and HEAD with `git archive` into the scratchpad, pointed TRACES at the same files, and computed `baseline_key` for 240 (scenario, BAR_V4 member) pairs across search3b, heldout3, test3b, search2 and stage 1. Output was byte-identical in v4 mode and in v3-data mode.
- **Driver `run.json` on pre-8dab76b dirs (e.g. the l3 runs).** These do not break: with no `run.json`, the first resume writes one and proceeds. Budget counting is wrong for them, though (m-D2).
- **Driver and `--max-cost-usd`.** Cumulative across links; checked only for iterations with missing slots, so completed iterations are skipped. Correct.
- **Running Alpine tuner chains.** The next link finds no `meta.json`, writes one from its own arguments plus `c16266a8ef49`, and proceeds. The 8dab76b `evaluate.py` gives the same `code_version` on 3.11, so syncing it later does not trip the guard. **However, the first link on the new code retries every invalid record left by the old code: see M-D1 before letting it run.**

---

## D findings

### [major] M-D1: retrying invalid tuner records on resume splits the ES trajectory, and `best` is then chosen from more than the budget
`evolve/analysis/tune_baselines.py:103-118` (`load_history` drops invalid records), `:185` (`best = max(done.values())`)

**Failure scenario:**
1. A candidate fails mid-run, from a 3600 s timeout, an OOM kill of one simulation, or a transient error. The running process stores -1e9 in memory and the ES carries on, ranking that point last. Later generations are recorded.
2. The next resume drops the invalid record and re-evaluates the point. Its real fitness can place it in the top mu, which changes that generation's elite, mean and sigma.
3. From there the ES samples new points and spends up to the full budget again along the new path.
4. The orphaned records of the original later generations stay in `done`, so `best` is the max over both trajectories.

The tuning chains are 4 links joined with `afterany`, so a link that starts after the budget was spent still runs `main()`. A single mid-run failure anywhere in the history is therefore enough. The time-limit case that motivated the fix is safe, because the kill happens in the last generation and nothing after it was recorded.

**Evidence** (`tuner_repro.py`; stub evaluator, budget 25, evaluation 3 fails with a timeout, link 1 completes, link 2 resumes):
```
A: link1 evals 25 | link2 new evals 19 | valid records now 43 (budget 25) | best changed: True
```
So the "equal-budget" control (125 vs 123) could silently become about 125 + (125 - g·λ) evaluations, and it would not reproduce from the seed. A control picked from more candidates is stronger, which is conservative for the paper's claim, but the stated budget would be false. `install`'s header counts history lines (44 here), so the evidence would be recorded but easy to miss.

**Action now (Alpine).** Before the next link of `tune_dynamo` / `tune_mooncake`, run `grep -n '"valid": false' history.jsonl` and compare with `wc -l`. An invalid line followed by at least λ (4 for dynamo, 5 for mooncake) later lines triggers this. If there is one, either set the record's fitness permanently, or accept the retry and report the real evaluation count.

**Suggested fix:**
- Retry an invalid evaluation immediately, in-process, up to N times, and keep invalid records as cache hits on resume. The exception is records whose failures are all signal exits (`exit -15` / `-9`), which can only come from the kill that ended a link.
- Compute `best` only over the keys the replayed trajectory actually visited.
- Stop at `budget` distinct evaluated keys.

**Hashed file:** no.

### [minor] m-D1: the tuner's truncated-line tolerance survives one resume, then crashes the next
`evolve/analysis/tune_baselines.py:103-118, 172`

**Failure scenario:** a link is killed mid-write, leaving a last line with no newline. The next link ignores that line but does not rewrite the file. Its first `log.open("a").write(...)` lands on the same physical line, so the line is now corrupt and in the middle of the file. The link after that raises `JSONDecodeError`, and so does every later link. The driver's `Archive` does rewrite the file, so this affects only the tuner.

**Evidence** (`tuner_repro2.py`):
```
link1 killed            # then last record cut mid-write
link2 killed lines 11
line 8 now: {"params": {"PREFILL_SCAL{"params": {"PREFILL_SCALE": 0.295648, "TEMPERATURE": 0.481938}, ...
link3 raises JSONDecodeError Expecting ':' delimiter
```

**Suggested fix:** on load, if the last line is truncated, rewrite the history without it atomically (`ev._atomic_write`), the way `Archive` does.

**Hashed file:** no.

### [minor] m-D2: resuming a pre-8dab76b driver run dir miscounts the budget and skips any code_version check
`evolve/driver.py:169-173` (`missing_slots`), `:183-194` (`check_run_config`)

**Failure scenario:**
- Old archives have no `slot`. Every old link restarted at iteration 0, and no-code rows carried no `iteration`.
- `missing_slots` therefore treats an iteration index as done once it holds `batch` rows. Proposals from repeated links are not counted, and old no-code proposals are re-proposed.
- `run.json` is written from the current settings without being compared with the archive: a v3-era archive (rows at `c6fceae7bfa8`) would continue under `c16266a8ef49` without a word.

**Evidence** (`driver_legacy.py`): a legacy archive of 2 links × 3 iterations × batch 2, plus 1 no-code row, resumed with `--iterations 5`:
```
legacy proposals before: 13 | added by --iterations 5 (budget 10 proposals): 4 | total 17 | run.json written: True
```

It does not crash, and the v4 campaign dirs will be fresh, so this matters only if an old pilot dir (`l3_t4_r1`, `l3b_t4_r4`, `rl_loto_r1` with 16 iteration-less no-code rows) is resumed.

**Suggested fix:** when the archive exists but `run.json` does not:
- refuse unless every row's `result.code_version` equals the current one;
- count non-seed rows against `iterations × batch`, or refuse and say the dir predates totals.

**Hashed file:** no.

### [minor] m-D3: head_to_head identifies bar members by path, not content
`evolve/analysis/head_to_head.py:52-87`; `evolve/analysis/bar_eval.py` (stores `BAR` as paths only)

**Failure scenario:** `evolve/policies/tuned/*_best.py` is re-installed under the same `code_version`, for example after M-D1 changes `best.json`. An earlier bar_eval JSON then maps to the same path. `check_provenance` passes, and `baseline_key` picks up the new file's sha, so the JSON's ratios (scored against the old tuned member) are multiplied by the new member's cached P99s. If the new member is not cached, a `KeyError` results; if it is cached, the numbers are silently wrong.

**Suggested fix:** in bar_eval, record `{member: file_sha[:12]}`, as the driver's `_member` does. head_to_head then refuses a JSON whose member shas differ from the current files.

**Hashed file:** no.

### [nit] N-D1: the tuner docstring still claims "same stage, bar and objective"
`evolve/analysis/tune_baselines.py:2`. Round-1 #9 asked for a rewording; SELECTION.md got it, the docstring did not. The tuner scores against `BAR_V4`, the campaign against `v4t`.

### [nit] N-D2: residual provenance gaps in the tuner
- `meta.json` lacks the BAR_V4 member shas (the driver's `run.json` has them), so an edit to a floor file mid-run is not caught.
- `install` checks only `best.json`'s `code_version`, not the rest of the history.

### [nit] N-D3: `Archive` rewrites a truncated archive non-atomically
`evolve/driver.py:59`. `path.write_text` truncates first; a kill during the rewrite loses the archive. Use `ev._atomic_write`.

---

## B findings

### [minor] m-B1: a non-JSON value in any digested table turns `code_version` into a silent `"unknown"`
`evolve/evaluate.py:376-383` (`json.dumps(tables)`), `:426-427` (`except Exception: "unknown"`)

**Failure scenario:**
- `KV_CAPACITY_DEVICE` is "filled in as deployments are added", from Vidur's MemoryPlanner. An `np.int64` there (or a Path, or a set) makes `json.dumps` raise.
- The catch-all then sets `code_version = "unknown"`. Nothing refuses that value (the only other `"unknown"` in the repo is unrelated), so every baseline key, run.json and meta.json uses `unknown|...`. Any later table state also maps to `unknown`, and stale baselines are reused, which is exactly what B#1 set out to prevent.
- `provenance()` raises outright.

**Evidence:**
```
ev.KV_CAPACITY['l3_8b'] = np.int64(483328)   # same value
numpy int same value -> unknown
provenance raises TypeError Object of type int64 is not JSON serializable
```

**Suggested fix:** canonicalise before dumping (`default=` that maps numpy scalars to `int()`/`float()` and Paths to `str`), and make `code_version` raise instead of returning `"unknown"`, or at least refuse `"unknown"` in `baseline_key`. Neither change alters today's digest or `c16266a8ef49`.

**Hashed file:** no.

### [minor] m-B2: bundle-check parts 3–4 from B_r1 are still open
`evolve/evaluate.py:599-602`. LOOP defers only the mid-run swap, but nothing changed in `score()`.

**Evidence:**
```
fingerprinted + unfingerprinted accepted: -0.0
missing==missing accepted: -0.0
```

**Fix** (harness only): treat a missing or `:missing` fingerprint as a mismatch when any other run of the scenario has one.

**Hashed file:** no.

### [nit] N-B1: `Scenario.warmup_frac`'s default is outside `code_version`, the digest and the tag
`evolve/evaluate.py:111`, used in `run_one` (`:505-508`). Changing the default changes the cached summaries of windowed scenarios under the same key. Only SMOKE is windowed today. Add `Scenario.__dataclass_fields__` defaults (or `warmup_frac`) to `table_digest`.

### [nit] N-B2: the `EVOLVE_KEEP_DIR` rewrite at import nests on re-import
`evolve/evaluate.py:70-71`. `importlib.reload`, or a child Python that inherits the mutated environment and imports `evolve.evaluate` with `EVOLVE_V3_DATA=1`, gives `.../v3data/v3data`. No current path does this (`golden` refuses KEEP_DIR, and no subprocess imports `evolve`). A guard such as `if not path.endswith("v3data")` would close it.

### [nit] N-B3: B_r1 residue
- `--bar` is still silently ignored unless `--task router --protocol v2`.
- No 3-row unit test of `expand_block_hashes_unpadded` (B#7's second bullet).
