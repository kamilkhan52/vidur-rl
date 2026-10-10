# Chunk C (golden-set tool), round 1

**Verdict: changes requested (small, all in non-hashed files). Today's claim holds: the v3 Mac reference is pre-v4 code and HEAD's hashed code reproduces it 30/30.**
The hashing covers everything scoring reads. But `golden` accepts a run with no hashes (EVOLVE_KEEP_DIR set) and `compare()` counts None == None as identical.
It also ignores cells that failed in both runs and cells the reference lacks. No cell I could check reaches preemption or restart.

## What I verified (no findings, context for the reader)
- **The v3 reference is truly pre-v4 code.** `code_version` of `git archive 1823e99`, computed by that commit's own
  `evolve/evaluate.py`, is `c6fceae7bfa8`. That is the `code_version` in `evolve/golden/v3_Kamils-Mac-mini.json`.
  (0272b19 gives `c6466da0ce2b`, af87600 gives `c7194082bc5e`, HEAD's archive gives `c16266a8ef49` = the working tree.)
  So A1-A2 are covered too: the reference was built before them.
- **HEAD's hashed code reproduces it on v3 data.** `golden check --manifest v3_Kamils-Mac-mini.json --against v3_alpine.json`
  gives `all 30 cells byte-identical`. The env diffs are code_version c6fceae7bfa8 vs c16266a8ef49, python 3.10 vs 3.11
  and pandas 2.2.3 vs 3.0.0. `v3_alpine` has HEAD's `code_version`, and 7fb6d38 (after it) left `code_version` unchanged.
  This one comparison covers every hashed change from 1823e99 to HEAD on these 30 cells, across two machines.
- **The Mac reproduces itself.** I re-ran 3 cells at HEAD on v4 data: `l3_8b-16R-Tool…p2/lop_uncached`, `l3_8b-8R-Tool-s0.75…/lor` and
  `70b-8R-Tool-s2.0…/lor`. Both hashes match `v4_Kamils-Mac-mini.json`. So the one v4 Mac/Alpine difference (16R
  lop_uncached, token_gaps only) is a cross-machine difference, not run-to-run nondeterminism on the Mac.
- **What is hashed covers what scoring reads.** `run_one` derives every score input from `request_metrics.csv`:
  `summarize_rows` reads e2e, prefill_e2e, scheduling_delay, prefill/cached/decode tokens and arrival time. It derives
  TBT from `token_gaps.npz` (`gap`, `request_id`). Both are hashed in full, warm-up rows included. The CSV also has a `replica`
  column, so routing changes show up directly. Unhashed outputs: `replica_prefix_cache_metrics.json` (the probe's
  `prefix_cache_stats` reaches only this file, and nothing in `evolve/` or `tail_metrics` reads it), `predictor_info.json`
  (bundle and clamp hits, bookkeeping) and `stuck_requests.json` (any stuck request already fails the completion gate).
- **Thread isolation of `_LOCAL` / `_TempfileShim` is sound.** `_LOCAL.keep` is set and cleared in the worker thread
  that calls `run_one` synchronously. Other threads see the real `TemporaryDirectory`, and `mkstemp` passes through.
  The keep dir is rmtree'd before each cell. The 6 tags x 5 names are distinct. A fake-subprocess run of 10 cells on
  4 workers gave 10 distinct hashes, each from its own cell. `run_one` has no result cache (the cache is in
  `ensure_baselines`), so a "cached, nothing simulated" path does not exist.
- `tests/test_golden.py`: 3 passed.

Repro harness for the findings below:
`/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/repro_golden.py`.
It replaces `ev.subprocess.run` with a fake simulator that writes a timestamped `request_metrics.csv` and
`token_gaps.npz` under `--metrics_config_output_dir`, then drives `golden.run_all` and `golden.compare`.

---

### [major] With EVOLVE_KEEP_DIR set, cells get no hashes, the build accepts them, and two such runs compare "byte-identical"
`evolve/analysis/golden.py:178-183` (and `compare` at 213); the cause is in `evolve/evaluate.py:441-443`

**Failure scenario.** `run_one` writes to `$EVOLVE_KEEP_DIR/<code_version>/<tag>/<who>` when that variable is set. It
writes there instead of the `TemporaryDirectory` that golden's shim hands out. `run_cell` then searches its own empty
keep dir, finds nothing, and records `request_metrics_sha256 = token_gaps_sha256 = None`. The cell still has `ok: True`
and a full summary, because `run_one` read the files from the other directory. `build` checks only `ok`, so it writes the
manifest and exits 0. `compare` asks only `r.get(f) != n.get(f)`, so None vs None is "unchanged". A reference and a check
both run in a shell that exports EVOLVE_KEEP_DIR therefore print `all 30 cells byte-identical` for any simulator change.
This is plausible on Alpine: `submit_python_job.sbatch` is called with `--export=ALL`, and EVOLVE_KEEP_DIR is the
documented v4 switch for keeping per-request outputs. The committed manifests are safe: `test_committed_manifests_have_schema` asserts 64-char hashes. A check
against them also flags `str != None`. The hole is any new reference built this way, such as a fresh Alpine manifest.

**Evidence (fake simulator, 10 cells):**
```
--- 2. EVOLVE_KEEP_DIR set: outputs go elsewhere, hashes None, ok True
[(True, None, None), (True, None, None)]
compare(None-run, None-run) -> []
compare(real, None-run)     -> ['l3_8b-8R-Tool-s0.75-w4000+2000-p1 / lor: request_metrics_sha256, token_gaps_sha256 differ; ...']
```

**Suggested fix.**
- In `main`/`run_all`, refuse to run when `EVOLVE_KEEP_DIR` is set, or `os.environ.pop` it.
- In `run_cell`, require exactly one match: `len(rm) == 1 and len(tg) == 1`, else `ok=False, reason="outputs not found / ambiguous"`.
  This also closes the stale-file case below.
- In `compare`, report a None hash on either side as a difference.

Hashed file: no (golden.py only). Do not fix it in `run_one`, which is hashed.

### [major] The golden set probably never reaches preemption or restart, so it cannot vouch for that path
`evolve/analysis/golden.py:44-51` (SCENARIOS)

**Failure scenario.** The windows are 2,000-3,000 requests and start from empty queues. In the 3 cells I re-ran,
including both "knee" cells, no request has `request_num_restarts > 0` or `request_preemption_time > 0`:
```
l3_8b-8R-Tool-s0.75-w4000+2000-p1__lor  restarts>0: 0 preempt>0: 0
70b-8R-Tool-s2.0-w2000+2000-p1__lor     restarts>0: 0 preempt>0: 0
l3_8b-16R-Tool-s0.5-w8000+3000-p2__lop_uncached  request_num_restarts: 0, request_preemption_time: 0
```
The other 27 cells are unverified, but they have similar or lighter loads. The preemption/restart path is known to
matter in this codebase (the earlier "restart fix"; `EvolvedGlobalScheduler.on_prefill_end` counts a prefill once
after a restart; the vllm_v1 `req_to_block_hashes` reuse after preemption, which the new probe code reasons about). A
change that breaks only that path passes `golden check`. For today's v4 changes the risk is low: none of them is
specific to preemption, and the probe restores `req_to_block_hashes` exactly. But the docstring prescribes the tool
"before and after every simulator change", and there it is a real blind spot. Other untested paths: `route_delay_s > 0`,
`view_refresh_s > 0`, a policy that defines `observe` (no golden policy does, so `_observe` returns at its first
line), the H100 device, the 128K cap (off under EVOLVE_V3_DATA), transforms, and the learned/RL scheduler. The RL
scheduler is in `_VERSION_EXCLUDE`, so changes there (af87600, 7fb6d38) are invisible to both the golden set and the
cache key. The defaults are what the "behaviour-preserving" claim is about, so those gaps are acceptable when stated.

**Suggested fix.** Add one or two cells that reach KV pressure: a longer window at the knee, or fewer KV blocks via a
cache-config flag. Assert in the build that at least one cell has restarts > 0 (record `restarts`/`preempted` counts
in the cell summary). Optionally add one cell with `route_delay_s > 0` and a toy `observe` policy, so the new paths
at least have a frozen reference. State the coverage in the module docstring.
Hashed file: no.

### [minor] compare() ignores cells that failed in both runs and cells the reference lacks; the message overstates the count
`evolve/analysis/golden.py:200-221, 255, 283`

**Failure scenario.**
- If a cell has `ok: False` in the reference and in the new run, nothing is reported. `build` exits 1 on failed cells
  but still writes the manifest (lines 274-277), so such a reference can exist.
- `compare` iterates only `ref`. Cells in `new` that the reference lacks are skipped, for example after a scenario is
  added to `SCENARIOS` or with a truncated manifest.
- `check` then prints `all {len(cells)} cells byte-identical`, using the *new* count. A reference with 0 matching
  cells prints "all 30 cells byte-identical".

**Evidence:**
```
compare([], new) -> []
compare(fail, fail) -> []
```
**Suggested fix.**
- Report `not r["ok"] or not n["ok"]` as a difference.
- Report keys in `new` that are missing from `ref`.
- Print the number of cells actually compared and identical.
Hashed file: no.

### [minor] Stale or ambiguous output files: rmtree errors are ignored and the first rglob match is hashed
`evolve/analysis/golden.py:168, 178-181`

`shutil.rmtree(keep, ignore_errors=True)` can fail silently, for example on Alpine scratch with a busy file or a
permission error. A stale `<old-timestamp>/request_metrics.csv` would then sit next to the new one. Vidur writes into
`output_dir/<timestamp>/`, so the two never collide by name. `rglob(...)[0]` then picks one in directory-listing order,
which can be the stale file. `run_one` has the same `files[0]` pattern for scoring. Not reproduced, because it needs
an rmtree failure. Fix: the `len == 1` check from the first finding, and `rmtree` without `ignore_errors` (or check
`not keep.exists()` afterwards). Hashed file: no.

### [minor] `request_metrics.csv` is written at `%.6f`, so its hash cannot see changes below 1 µs; token_gaps is the only full-precision witness, and SUMMARY_KEYS omits the token-gap statistics that would size such a change
`vidur/metrics/cluster_metrics_store.py:106` (context); `evolve/analysis/golden.py:59-60`

Scores are computed from the same rounded CSV, so equal CSV bytes do give equal E2E/TTFT/TPOT scores on these cells.
But "byte-identical outputs" is weaker than "bit-identical simulation". A sub-µs timing change appears only in
`token_gaps.npz`, and requests with no decode gaps have no full-precision witness. The v4 Mac/Alpine cell is probably
this case: `request_metrics` is identical, `token_gaps` differs, and all 10 summary keys are equal to the last bit, so
compare prints "summary metrics equal". The tool cannot say how large the gap difference is. This is inferred:
I cannot see the Alpine npz. `run_one` already returns `mean_tbt`, `p999_tbt`, `n_tbt` and `bundle`, but
`SUMMARY_KEYS` drops them. Fix: add `mean_tbt`, `p999_tbt`, `n_tbt` (and `bundle`, `clamp_hits` for diagnosis) to the
cell. Optionally store `max |Δgap|` when a check finds a token_gaps-only difference. This matters before pooling Mac
and Alpine v4 results. Hashed file: no.

### [minor] Default `--out` is `v3_<host>.json` whatever the data; the Alpine script builds "v3" without EVOLVE_V3_DATA
`evolve/analysis/golden.py:270`; `scripts/submit_golden_alpine.sh:13-17`

- `golden build` without `--out` now runs on v4 traces (the default since af87600) and overwrites
  `evolve/golden/v3_Kamils-Mac-mini.json`. That file is the only pre-v4 reference, and rebuilding it means rebuilding
  1823e99. Git can recover it, but a silent overwrite of the comparison baseline is a trap.
- `submit_golden_alpine.sh` names the job `golden_v3` and writes `v3_alpine.json`, but does not pass
  `EVOLVE_V3_DATA=1`. Run as committed today, it builds v4 data under the v3 name.
- The header says "Sync evolve/analysis/golden.py and the policies", but the script rsyncs only golden.py and one
  bundle, not the simulator or the policy files. Policy drift would show up as an env diff and differing hashes, so
  this is not a false "identical".
- The committed `v3_alpine.json` has v3 trace hashes, so it was evidently built some other way.

Fix:
- Derive the prefix from `ev.V3_DATA` (`v3_`/`v4_`) and refuse to overwrite an existing manifest without `--force`.
- Record `"v3_data": ev.V3_DATA` in `environment`.
- Update the script (EVOLVE_V3_DATA, sync list), or mark it superseded.

Hashed file: no.

### [minor] The tests check the helpers, not the plumbing or the holes above
`tests/test_golden.py`

The three tests do what their names say. `sha_npz` ignores the container and key order and is sensitive to dtype;
`compare` flags a changed hash, a missing cell and an ok flip; the committed manifests have full hashes. Nothing
exercises `run_cell`/`run_all`, the part with real risk: the shim, which directory gets hashed, and EVOLVE_KEEP_DIR.
`compare` is not tested on None hashes, on failures in both runs or on extra cells, all of which pass today. The
fake-subprocess harness above runs in under a second and can serve as the plumbing test. Assert three things: each
cell hashes its own output; a changed output byte changes the cell's hash; EVOLVE_KEEP_DIR set yields a failure, not None.
Hashed file: no.
