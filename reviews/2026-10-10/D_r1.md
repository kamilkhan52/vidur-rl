# Chunk D, round 1: baselines, tuning control, head-to-head, campaign launcher

**Verdict: changes requested.**
The tuning ES (populations, budget, resume replay) is correct. Two things need attention before results are used:
(a) the campaign's equal-budget claim is false as written (the driver restarts its iteration count on every chain
link), and (b) `head_to_head` silently produces wrong numbers (wrong bar for v4 JSONs, and no code_version guard).
For the running tuning jobs: check the Alpine `history.jsonl` files for `"valid": false` lines before each resume
(finding 4).

None of the suggested fixes touch a `code_version`-hashed file. Editing a floor or tuned policy file does change that
policy's own baseline key, because the key includes the file's sha.

---

### [blocking] Campaign budget is not 60 iterations per run: every chain link restarts the driver's iteration count
`scripts/launch_v4_campaign.sh:4,33-38`; `evolve/driver.py:169` (`for it in range(iterations)`)

**Failure scenario.** Each of the 4 links runs `evolve.driver --iterations 60`. On resume the driver reloads the
archive (seeds are skipped), but it then runs `range(iterations)` from 0 again, so each link may run another 60
iterations. Links are chained with `afterany`, so all 4 always run. The real budget is therefore set by wall-clock
time (4 x 4 h), not by `ITERATIONS=60`. Only the DeepSeek runs can be cut short, by the cumulative `MAX_COST_USD=8`;
the relay run's cap is 1000.

**Evidence.** In a prior search3b run (`evolve/runs/l3b_t4_r4/archive.jsonl`, 30 candidates), `wall_s_total`
averaged about 2,000 s per candidate across 10 scenarios. With `WORKERS=10`, that is roughly 200-400 s per candidate
plus proposal time. So one batch-2 iteration takes about 6-10 min, and one 4 h link covers about 25-40 iterations.
Over 4 links that is about 100-160 iterations, or 200-320 proposals, against the tuning control's 125 evaluations.
This estimate is from a v3 run and was not measured on v4/Alpine. Even in the best case, where link 1 finishes 60
iterations, links 2-4 add up to 180 more.

The script header ("Budget per run: 60 iterations x batch 2 = 120 proposals ... about the tuning control's 125") and
the paper's equal-budget control therefore do not hold. A related effect: `random.Random(seed)` is re-created on every
link, so parent selection depends on where the links were cut. Runs are then not reproducible from the seed alone.

**Suggested fix.** Make `--iterations` a total. Derive `start = 1 + max(meta.iteration)` over the archive rows, or
count non-seed proposal rows and stop at `iterations * batch`. Restore or advance the RNG deterministically, or save
its state. Alternatively, cap the chain externally, though that is harder to make exact.
**Hashed file:** no (`evolve/driver.py`, launcher).

### [major] `head_to_head.main()` always reconstructs against BAR_V2, whatever bar the JSON was scored against
`evolve/analysis/head_to_head.py:23,60`

**Failure scenario.** `bar_eval --bar v4` (or `v4t`) stores ratios against min over BAR_V4. `main()` calls
`head_to_head(..., cache)` without `bar`, so it uses the default `ev.BAR_V2`. It then rebuilds the policy P99 as
ratio_v4 x min_V2. That P99 is inflated whenever a floor or tuned member is the per-metric best. The output also omits
the floor and tuned members. Every bar_eval JSON from the v4 campaign will have this problem. The JSON already records
`"bar"` (`bar_eval.py:72`).

**Evidence** (mock cache; the third member `jsq` is the best):
```
policy/sticky_lor reported 1.6 true 0.8
members reported ['lor', 'sticky_lor']
```
`tests/test_head_to_head.py` passes `bar` explicitly, so it cannot catch this.

**Suggested fix.** Read the bar from the JSON (`json["bar"]`, falling back to BAR_V2 only if the key is absent) and
pass it through. Refuse to merge JSONs whose bars differ. Better still, read the policy's own P99 from the cache:
bar_eval puts `policy:<path>` runs into `baselines.json`, which avoids both the bar algebra and the 4-dp rounding.
**Hashed file:** no.

### [major] `head_to_head` mixes a JSON's ratios with baselines from a different code_version and traces, silently
`evolve/analysis/head_to_head.py:36-37,51-55`

**Failure scenario.** The ratios come from the JSON, scored under its own `code_version`. The baselines come from the
cache under the *current* code_version and trace sha. When the current cache has those scenarios, the two are combined
with no error. search3b now has all BAR_V2 baselines under v4 (`c16266a8ef49`), and heldout3 is running.

**Evidence.** This command ran without error and printed a full table that multiplies v3 ratios (padded traces) by v4
baselines (unpadded traces):
```
.venv/bin/python -m evolve.analysis.head_to_head --json evolve/analysis/out/l3_ablate_search3b.json \
    --stage search3b --policy l3_48a067a7df63.py      # JSON code_version c6fceae7bfa8, cache c16266a8ef49
  vs sticky_lor  e2e 0.784 [0.720,0.835] 10 | ttft 0.549 ... | tbt 0.881 ...
  vs mooncake_min_ttft  e2e 0.716 ... (all 7 rows printed)
```
Every committed bar_eval JSON in `evolve/analysis/out/` is at `c6fceae7bfa8` or older, so none of them can be
re-headed correctly today. The committed `test2_head_to_head.txt` was produced before v4 and is fine.

**Suggested fix.** `assert json["code_version"] == ev.code_version()`. Also record and compare trace hashes, or key
off the JSON's own provenance.
**Hashed file:** no.

### [major] Tuner records infrastructure failures as permanent fitness -1e9; resume never retries them
`evolve/analysis/tune_baselines.py:143-150`; `evolve/evaluate.py:590-593`

**Failure scenario.** A candidate fails if any one of its 10 sims fails. `evaluate_policy` then returns
`combined_score=-1e9, valid=False`, and `score()` appends that to `history.jsonl`. On resume the record is a cache hit
and is never re-run. The ES ranks the point last and spends a budget slot on it. The record also stores no failure
reason, so it cannot be diagnosed later.

Plausible triggers on the running 4 h chains:
- the time-limit SIGTERM reaching the `vidur.main` children before the Python parent;
- a sim OOM-killed in the 88 G cgroup;
- `run_one`'s 3600 s timeout on a contended node;
- `scancel`.

**Evidence** (partly verified). In a local test, a parent thread waiting on `subprocess.run` recorded the failed child
when the child got SIGTERM 1 ms before the parent (2/2 lines written). With 0 ms between them, nothing was written.
Whether slurmstepd signals children before the batch shell and Python on Alpine was not verified; the order is not
guaranteed. The tuning sbatch command itself is not in the repo (presumably `submit_python_job.sbatch` with
`eval "$CMD"`).

**Action now.** On Alpine, run `grep -c '"valid": false' evolve/runs/tune_*/history.jsonl`. Delete any such lines
before the next link resumes; the replay stays deterministic.

**Suggested fix.** Do not persist invalid results whose failures are signal exits, timeouts or OOM, or retry invalid
records on resume. Store `res.get("failures")` reasons in the record.
**Hashed file:** no.

### [major] Resume after a kill under `--parallel 2` reuses a candidate number and overwrites a recorded candidate's file
`evolve/analysis/tune_baselines.py:132,139-142`

**Failure scenario.** Candidates n and n+1 run in parallel, and n+1 finishes first and is logged. The job is then
killed while n is still in flight. On resume, `counter = len(done)` equals n+1. The replayed in-flight params get
number n+1 and overwrite `candidates/<fam>_{n+1}.py`. History still says that file holds the earlier params. Both
`history.jsonl` and `best.json`'s `"file"` can then point at code with different constants. Params and fitness are
correct, and `--install` renders from params, so it is unaffected. But anyone re-evaluating `best["file"]` (held-out,
test3b) would run the wrong policy.

**Evidence.** I drove `main()` with a stub evaluator: budget 40, parallel 2, even-numbered candidates slow, killed
with `os._exit` after 0.33 s, then resumed. The killed-and-resumed run produced the same 40 parameter sets as an
uninterrupted run, so replay determinism holds. But:
```
MISMATCH B/candidates/dynamo_0005.py {'PREFILL_SCALE': 0.282261, 'TEMPERATURE': 0.120374} file has 10.421396
dup file names: 1 records with wrong file: 1
```
This will happen on the real runs whenever a link is cut after out-of-order completion.

**Suggested fix.** Name files by a hash of the params (or the rendered source), as the driver does with `save_program`.
Or set `counter = 1 + max(index in existing record file names / files on disk)`.
**Hashed file:** no.

### [minor] Resume does not check provenance; a truncated last line crashes it
`evolve/analysis/tune_baselines.py:125-128,87-93`

Three gaps:
- History records carry `code_version` (valid records only), but resume accepts records from any code_version.
  Resume also ignores `--seed`, `--stage`, `--objective` and bar, none of which are recorded. A link resumed after
  the tree under `$R` was updated mixes simulator versions in one ES run, silently. Records from an earlier run in the
  same dir with another `--seed` count toward `best` and add to the evaluation count.
- `install` checks only `best.json`'s code_version, not the rest of the history.
- A partially written last line (kill mid-write, quota) makes `json.loads` raise, so the next link dies at once.
  Unlikely, but the chain then loses 3 links.

A related theoretical race: two in-flight candidates with identical params are both evaluated and both logged. The
duplicate line then shifts `len(done)` and feeds the collision above. With continuous sampling this is practically
impossible.

**Fix.** Write a `run.json` with family, stage, objective, bar, seed, budget and code_version. Refuse to resume on a
mismatch. Skip or refuse records with a different code_version. Tolerate a malformed final line.
**Hashed file:** no.

### [minor] The defaults-reproduce test never reaches the tie-break or same-burst paths
`tests/test_tune_baselines.py:13-32`

A fresh `PolicyContext` per call means the burst index is always empty, so `_match` always returns 0. With random
`pending` (0-5000) and float `kv_usage`, I counted the RNG tie-break taken in 0/200 cases for both families. So the
test would pass even if the template's tie-break or same-burst code were broken.

The property itself holds. I ran a stronger check: a shared ctx with bursts of 4 at one `now`, quantised loads so ties
are frequent, 8 replicas. It found 0/6000 mismatches for each family. That also confirms Mooncake's float
`1.0*pending + (n-hit)*1.0 + 0.0*kv` orders exactly like the int expression. Adding that harness to the test would
make the test check what its name says.
**Hashed file:** no.

### [minor] Launcher details: DRY=1 still sshes; rsync into a live shared tree runs before the check; misleading comment
`scripts/launch_v4_campaign.sh:10,23-28,30-44`

- Line 26 (`REMOTE_CV=$(ssh alpine ...)`) is not wrapped in `run`, so `DRY=1` opens an ssh connection (Duo). This
  contradicts "print the sbatch commands only". I did **not** run the script. DRY also exits at line 16 until the tuned
  files exist, so it cannot preview anything yet.
- The rsyncs (lines 24-25) push `vidur/` and `evolve/` into `$R` *before* the code_version check. If the Mac has
  hashed changes Alpine lacks, the push makes both sides equal, the check passes, and any job still running from `$R`
  changes code mid-run. Per the plan, golden, bars and gen70 run there. Each sim is a fresh `python -m vidur.main`,
  while the parent's code_version is cached at start, so results land under the old key. Compare the CVs first, and
  refuse to push when they differ and jobs are active, or use a fresh directory per release.
- The comment on line 23 says the rsync carries "the Mac's v4 baseline cache", but `.cache*` and `evcache*` are
  excluded. The runs are actually seeded from Alpine's `evcache_tune_dynamo/baselines.json`. That holds BAR_V4 only,
  so each of the 3 runs recomputes the 2 tuned members on search3b (20 sims each). This is correct but duplicated.
- There is no guard against submitting twice. Re-running the launcher starts a second chain on the same `RUN_DIR`,
  and two drivers then append to one archive.
- The launcher does not compare the installed `*_tuned_best.py` header code_version against `$CV`. If hashed code
  changed between `--install` and launch, v4t is a bar tuned under an older simulator.

Verified fine:
- `SEED_POLICY` quoting. Reproduced the remote parse: `--export=...,SEED_POLICY=a b c,SEED=1,...`, which the sbatch
  script word-splits correctly.
- Backslash-newline joins inside the double-quoted `cmd`.
- macOS openrsync honours the trailing `--exclude`s, tested locally: `runs`, `.cache_v4`, `evcache_*` and
  `__pycache__` were skipped.
- No `--delete`, so remote run and cache dirs are not removed or overwritten.
- `--relative` lands `vidur/` and `scripts/` at `$R/`.
- An `afterany` chain is right for timeouts.

**Hashed file:** no.

### [minor] The tuning control is not scored against the same bar as the campaign; fitness values are not comparable
`evolve/analysis/tune_baselines.py:1-5,143`; launcher `BAR=v4t`

The docstring says candidates are "scored exactly like an evolve candidate (same stage, bar and objective)". The tuner
scores against BAR_V4, while the campaign scores against v4t (BAR_V4 plus the tuned winners). That is unavoidable,
but it means a tuner history fitness and a campaign archive fitness must never be compared directly. The comparison
has to be `bar_eval`/`head_to_head` of both finalists under one bar. Budgets are 125 vs 123 (60 x 2 + 3 seeds);
negligible next to the blocking finding. Reword the docstring.
**Hashed file:** no.

### [minor] Power-of-two floors crash for a single replica
`evolve/policies/floor/p2c_outstanding.py:9`, `p2c_kv.py:14`

With n = 1, `ctx.rng.sample(range(1), 2)` raises `ValueError: Sample larger than population` (reproduced). No stage
has fewer than 8 replicas, so this is latent. Do **not** fix it now: the floor file's sha is in its baseline key, so
any edit invalidates the BAR_V4 cache and changes the bar the tuning runs are scored against. Note it for v5. There
are no masks in the interface; all replicas are always offered. Tie-breaking is deterministic (first sample, and jsq
falls back to the seeded `ctx.rng`). The KV load (`kv_usage x capacity + pending uncached prefill`) matches its
docstring and the Dynamo decode term.
**Hashed file:** no (but see the baseline-key caveat).

### [nit] Install header makes the tuned file's sha, and so its baseline key, depend on the install path
`evolve/analysis/tune_baselines.py:94-98`

The header embeds `{run}` as typed. Installing the same `best.json` via a relative path on one machine and an absolute
path on another gives two v4t bars with different cache keys. The results are identical, but caches cannot be shared.
The launcher rsyncs the Mac's copy, so this is harmless as long as installs happen only on the Mac.

---

**Checked and fine:**
- `render()`: anchored `^NAME = ` with no prefix collisions, docstring mentions are not at line start, `repr` of
  `1e-05` and `0.0` is valid source, count=1.
- `reflect` for any real value.
- The (mu/mu_w, lambda) mean update and sigma clipping.
- Budget accounting: defaults counted once, the ES gets `budget-1`; the stub run logged exactly 40/40.
- The lock around counter and log.
- `ensure_baselines` runs before the parallel candidates.
- Invalid candidates sort last.
- `install` refuses a stale or None code_version.
- The ratio algebra in `head_to_head` for a correct bar.
- The cluster bootstrap uses the same sorted-tag order for `d` and `clusters`.
