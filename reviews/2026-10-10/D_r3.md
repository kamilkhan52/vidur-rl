# Chunk D, round 3: tuner resume logic (fix 7cb131c, reviewed at HEAD 16ffa55)

**Verdict: changes requested (1 major, 2 minor, nits).** Both BD_r2 findings are fixed: the trajectory no longer splits, the budget stays equal, and a truncated line no longer crashes the second resume. Moving from link-1 records (1f9fde4, `--parallel 2`, no `gen`, no `meta.json`) works exactly: 80 of 80 randomized chains matched the uninterrupted run.
**Remaining gap.** The retry rule ("a failed record in the last recorded generation was not consumed") picks the wrong records. A real kill leaves no failed record, so the records it retries are genuine failures that the uninterrupted run also saw. A failure that does not recur then changes the trajectory, and a link that starts after the run finished can change `best.json`, even after `--install`.
None of the fixes below touches a `code_version`-hashed file. `tests/test_tune_baselines.py` passes (7 passed); `code_version` is `c16266a8ef49`.

Repro scripts are in this session's scratchpad (`r3/harness.py`, `s1`–`s7.py`). They run the real `main()` of HEAD, and of 1f9fde4 for link 1, with a stub `evaluate_policy`:
- Fitness is a smooth function of the params.
- Random per-candidate latency, so with `--parallel 2` records land in random order.
- Failure models: transient (fails on the first attempt only), deterministic, or none.
- A "kill" after K new records:
  - SIGKILL-like: the candidates in flight and in the queue write nothing.
  - SIGTERM-race: the candidates in flight write an `exit -15` record first.

Budgets 125 (lam = 6 dynamo, 7 mooncake), plus 25/29 (exactly on a generation boundary), 31 and 40.

## Requested checks

1. **BD_r2 major: fixed.** `s1.py`: a mid-run transient failure, link 1 completes, then link 2 and link 3 run. Link 1 ran on either code version, budgets 25 and 125. Link 2 makes 0 new evaluations whenever the final generation has no failure, so no trajectory split and no orphan records. The one exception is the final-generation case, which is part (b) of M1. The chain always ends with exactly `budget` distinct keys, all on the uninterrupted run's trajectory.
   **BD_r2 minor: fixed.** `s2.py`: link 1 is killed and its last record cut mid-write; then link 2 is killed, link 3 is killed, and link 4 completes. The truncated line is removed atomically, and the result equals the uninterrupted run. The missing-newline variant still fails: m1.
2. **Link-1 records.**
   - *Generations are contiguous in the file.* `evaluate` is `list(ex.map(...))`, which blocks until every future of generation g is done, and only then does `ask_tell` compute the update and sample g+1. If a candidate raises, the `with ThreadPoolExecutor` exit waits for the remaining g futures; the exception then ends `main`, so no g+1 record follows.
   - *Order within a generation does not matter.* Only completion order varies, and the inference uses block membership, not order.
   - *Defaults come first.* `score(defaults)` runs synchronously before the pool exists.
   - *Inference holds while each candidate writes exactly one line.* A duplicate key would break that: a cache hit writes no line, and two concurrent identical keys write two. With 6-digit rounding of continuous samples (sigma >= 0.02), duplicates are practically impossible.
   - *Evidence.* `s3e.py none`: 80 randomized chains, each with link 1 on 1f9fde4 and `--parallel 2`, killed at random points (including during the defaults and exactly at generation boundaries), then 1–5 HEAD links killed at random points, then one post-finish link. All 80 match the uninterrupted run (same keys, same valid set, same best). Every link-1 trial needed at least one HEAD link; the dir lacked `meta.json` and no guard tripped.
   - *Kill at the 4 h limit.* Slurm sends SIGTERM to every process of the job. Nothing in `evolve/` installs a handler (`grep signal` finds none), so the tuner process dies immediately with its threads.
     - The file then holds complete lines for every finished candidate: the earlier generations in full, and part of the current one. In-flight candidates leave no line.
     - Each line is a single small `write` at `close`, so a torn line is very unlikely.
     - An `exit -15` record is possible only in the race where a simulation child dies and the parent writes before its own signal lands (milliseconds against microseconds).
     - The resume then re-evaluates the missing candidates; they are cache misses, so this is correct. It also retries every *failed* record of the last generation, and those are almost always genuine failures (M1).
3. **`meta.json`.**
   - *Link 2* writes it from its own arguments, `code_version` and `bar`. *Link 3* compares and passes.
   - *`bar` is path-independent*: basename plus file sha (`'mooncake_min_ttft.py:40204e7b7d0d'`, ...). Nothing in `meta` depends on the run-dir path, cwd or REPO; a link started from another cwd with a relative `--run-dir` passes (`s6.py`).
   - *Another budget is refused.* The commit after 7cb131c (16ffa55) does not change the value.
   - *The guard is not spurious.* It trips only on real changes: a floor file edit, a hashed-file sync, or other arguments.
   - *Open gaps.* Link 2 cannot check that link 1 ran with the same arguments, and `meta` does not cover the template or the tuner itself (m2).
   - *Precondition.* LOOP says 068fd0b was synced to Alpine before 7cb131c. If any chain link already ran 068fd0b code, the dir has a `meta.json` without `bar`, so every later link exits ("run dir was started with ..."). That link would also have appended out-of-slot retries, which breaks the position inference.
   - *Timing.* Link 1 started after 04:20 (1f9fde4), so no link has run since then, unless link 1 died early. **Before link 2: confirm `meta.json` does not exist in either run dir.**
4. **Edge cases.**
   - *Multiple consecutive resumes* are exact (above).
   - *A defaults failure* followed by later generations is kept, which matches the uninterrupted run. A defaults failure followed by a kill before any generation-0 line is retried; that is the M1 class, but harmless to the ES because the defaults never feed it (`s6.py`).
   - *Budget exactly on a boundary* (25 → 4×6, 29 → 4×7) is exact when nothing fails.
   - *A resume after the run finished, with no failure in the final generation* makes 0 new evaluations, and `best.json` keeps the same params. When link 1 ran on old code, the record gains an inferred `"gen"` key, which is harmless.
   - *A resume after the run finished, with a failure in the final generation* re-evaluates it and can change `best.json`: M1(b).
5. **Other risks to the chain.** See m2: an edit between links to the template, `FAMILIES`, `ask_tell` or `params_of` is not detected. Nothing else found. RNG streams (`default_rng`, PCG64) and JSON float round-trips are exact, and fitness ties sort stably.

---

### [major] M1: the resume retries genuine failures, not kill artifacts, so a chain can diverge from the uninterrupted run, and a post-finish link can change `best.json`
`evolve/analysis/tune_baselines.py:138-141` (`if r.get("valid") or r["gen"] < last`); docstring `:112-117`

**Premise.** The docstring says a failed record in the last recorded generation "was not consumed (the job died before the generation finished, usually a time-limit or out-of-memory kill)". A whole-job kill does not produce failed records, though: the tuner process dies with its children (point 2 above). Whatever failed record sits in the last generation is a genuine failure from earlier in that link, the same first attempt the uninterrupted run would have made.

Retrying it changes nothing when the failure is deterministic. When it is not, the retry differs from the uninterrupted run. Non-deterministic failures include:
- a 3600 s timeout under node load;
- a cgroup OOM kill of one simulation (`exit -9`), which with `--parallel 2` depends on what the other candidate is doing;
- a transient I/O error.

**(a) Mid-run.** A link is killed while generation g is in progress, and g already holds a genuine failure. The next link retries it; if it now succeeds, g's elite changes and every later generation differs. The budget stays equal (exactly `budget` distinct keys, no orphans), so the BD_r2 major is fixed. But the commit's claim that "a resumed run replays the original trajectory" does not hold, and the result no longer reproduces from the seed.

**(b) After the run finished.** The chain links are `afterany`, so links that start after completion still run `main()`. The final generation (4 candidates for dynamo, 5 for mooncake at budget 125; a full lam on a boundary) is the last recorded generation, so its failures are retried in *every* later link, and `best.json` is rewritten.
- If `--install` already ran, `best.json` and `evolve/policies/tuned/*_best.py` then disagree, and the history gains lines, which inflates the install header's count.
- A deterministic failure is re-run in every remaining link, which wastes an evaluation's wall time each time.

**Evidence:**
```
s3b.py  120 chains, transient failures (first attempt fails), random kills, link 1 on 1f9fde4:
  exact match 10 | diverged 110, and every divergent chain retried a genuine (non-kill) failure | unexplained 0
s3e.py  same with deterministic failures: 80/80 exact; with no failures: 80/80 exact
s4.py   run finished, the final generation's best candidate had failed once, one post-finish link:
  dynamo 125:   new evals 1, best before -0.0000 after +1.0000, history lines 126
  mooncake 125: new evals 1, best before -0.0003 after +1.0000, history lines 126
  dynamo 25 / mooncake 29 (boundary): same
```
The failure rate in s3b is exaggerated (8–20 %). At a realistic rate p, the chance per kill is about 1-(1-p)^(records in the last generation): roughly 5–10 % at p = 2 %. Part (b) needs a non-recurring failure among the final 4–5 candidates.

**Suggested fix:**
- Keep every record on resume (this is exact under a SIGKILL or SIGTERM kill).
- Retry only records in the last recorded generation whose failure reasons are all signal exits (`exit -15`, `exit -9`): the parent-survives race. Legacy link-1 records carry no `failures`, so keep them (the race is far rarer than an OOM or timeout).
- Tests:
  - A post-finish link with a failure in the final generation makes 0 new evaluations and leaves `best.json` unchanged.
  - A genuine failure in an interrupted last generation is kept.
  - The current e2e test builds a kill-made invalid record by hand, a state a real kill does not produce. Repurpose it for the signal-exit case.
- **Alpine, before link 2 (~08:20 or later):** list the `"valid": false` lines of each `history.jsonl` and their positions. A failed line inside the last block (index >= 1 + lam·floor((N-2)/lam), lam = 6 or 7) will be retried by 7cb131c. Older failed lines are safe.

**Hashed file:** no.

### [minor] m1: a last record complete except for its newline still crashes the chain on the second resume
`evolve/analysis/tune_baselines.py:121-132`

**Failure scenario:** the kill cuts the write after the JSON object but before `"\n"`. The line parses, so it is kept, but the file does not end in a newline. The next link's first append joins it on one physical line, and the link after that raises `JSONDecodeError: Extra data`, as does every later link. This is the same failure class as BD_r2 m-D1, just narrower.

**Evidence** (`s2.py`):
```
only the newline missing | link1 (old code) killed | link2 killed lines 15 | link3 JSONDecodeError: Extra data: line 1 column 219 | link4 JSONDecodeError ...
```

**Suggested fix:** when `not text.endswith("\n")`, rewrite the file atomically with the lines that parse, each newline-terminated. As now, drop the last line only if it does not parse.

**Hashed file:** no.

### [minor] m2: nothing checks that the loaded records lie on the replayed trajectory, and `meta` does not cover the ES or the template
`evolve/analysis/tune_baselines.py:169-177, 215-216`

**Failure scenario:** a record off its positional slot, or a change between links to `FAMILIES` (ranges), `params_of` (rounding), `ask_tell` (sigma, clip, lam) or the template `evolve/policies/tunable/*.py`. Each one silently changes either the keys the replay asks for or what an evaluation means.
- Off-slot record: a link that ran 068fd0b code (out-of-slot retries) or a duplicate key. It shifts the inferred `gen` of every later legacy record, and a mislabelled earlier-generation failure is then retried, which recreates the orphan and over-budget problem.
- Changed ES or template: the replay misses the cache and spends new evaluations, the old records stay in `done`, and `best` is the max over all of them.
- None of these files is in `code_version`, and none is in `meta`.

**Evidence** (`s7.py`): a legacy history with one 068fd0b-style retry appended. Link 2 completes without a word, and the valid set differs from the uninterrupted run.
```
link2 done new evals 85 | lines 126 (ref 125) | valid-set equal False
```

**Suggested fix:**
- After `ask_tell`, refuse unless every loaded key was visited and `len(done) == a.budget`. Compute `best` over the visited keys only.
- Add to `meta`: the template's sha, and a hash of `inspect.getsource` of `ask_tell`, `params_of`/`to_value` and `FAMILIES`.

This would also have caught the BD_r2 major directly.

**Hashed file:** no.

### [nit] N1: leftovers
- `:176`: `meta.json` is written with a plain `write_text`. A kill during that write leaves a file that makes every later link raise; use `ev._atomic_write`. The window is tiny, because kills come at the time limit, not at startup.
- `:216`: the comment "valid evaluations only" is false: `done` holds consumed failures. If every candidate failed, `best.json` is an invalid record, and `install` would freeze it (its `code_version` is set from `meta`). Filter on `valid` and refuse when none exists.
- `install` header: "after {n} evaluations" counts lines, so retries are counted twice. Count distinct keys instead.
- BD_r2's Alpine action note gives λ as 4 and 5; the real values are 6 (dynamo, dim 2) and 7 (mooncake, dim 3). It is superseded by the M1 note above.
