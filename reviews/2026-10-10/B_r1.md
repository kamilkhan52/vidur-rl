# Chunk B (harness and v4 traces), round 1

**Verdict: approve with minors** (one major latent cache-key gap, one major test weakness; no wrong numbers found).
The v4 traces are correct: they rebuild byte-identically, pass lint, and match v3 and the raw Mooncake files row by row. The 128K cap gives the same row count in the window filter and in the generator.
The open issue is reproducibility: `code_version` and the baseline key ignore the *values* of the cap and capacity tables, so editing one of them would silently reuse stale baselines.

Reviewed: `git diff 1823e99..HEAD` on the chunk paths at HEAD 7fb6d38 (`code_version` = `c16266a8ef49`).

## What I verified (no finding)

- **v4 traces vs raw and v3**, row by row for all three traces (`scratchpad/align.py`):
  - `num_prefill_tokens == input_length` and `num_decode_tokens == output_length`.
  - v3 decode lengths are equal row by row, and v3 pads are 0–511 / exactly 512 / 0–511 (means 287.3 / 512 / 302.1, as in TRACES.md).
  - Sessions are equal to v3 in every row.
  - For Conv (upstream's file), the full-prompt-block ids of v3 and v4 agree in all 12,031 rows, so the session carry-over landed on the right rows.
  - No row has 0 prefill or 0 decode tokens, so the generator's `clip(lower=1)` never changes P+D.
- **Hash-id construction:**
  - Every row has `len(hash_ids) == ceil(P/512)`, so the sub-ids always cover the P//16 full prompt blocks.
  - The largest real sub-id is 5.87M, below the 5e9 unique base.
  - The largest row has 11,961 blocks, below the 1e6 per-row stride, so private ids cannot collide across rows.
  - Mooncake's partial-last-block ids are never reused as full blocks or at another partial length (0 cases in all three traces), so the docstring's "equal ids mean equal tokens" holds.
  - The `Request.__init__` assert (`len == (P+D)//bs`) is the same rule as lint R1.
- **Rebuilds:**
  - Synth `--v4` rebuilt in scratch matches the TRACES.sha256 entry (`40db850a…`).
  - Synth v3 rebuilds byte-identically.
  - A 300-row Conv slice through `convert_trace(..., unpadded=True, sessions_from=v3)` is byte-identical to the first 300 rows of the committed v4 file.
  - `shasum -c TRACES.sha256`: all 10 OK.
- **Lint:**
  - All three v4 files pass.
  - Negative tests: dropping one id, reusing one decode id in another row, repeating an id inside a row, and a second block_size each trip exactly R1, R2, R3 and R4.
- **Total-token drops** 3.17 / 3.97 / 1.92 % and **10 Synth requests > 131,072** match TRACES.md.
- **Cap consistency:** `_window_csv` and `TraceRequestGenerator` both filter `P + D <= max_request_tokens(sc)` on the same columns, so `n_expected` equals the generator's count. No row sits exactly at 131,072.
- **Splits:**
  - Tag overlap is 0 for every pair of search3b / heldout3 / test3 / test3b.
  - test3b has 45 scenarios, 5 draws in each of 9 configurations: Tool/Conv p9–p13, Synth p8–p12. This matches SELECTION.md §1.
- **`EVOLVE_V3_DATA=1`** restores the v3 trace paths and the v3 cap (`MAX_CONTEXT = {}` gives `KV_CAPACITY`). `sim_command` is otherwise unchanged for `device="a100"`.
- **Profiles in `code_version`:** a behavioural check (temp REPO, edit one profiling CSV) changes `code_version`. The walk finds 54 CSVs, the same as `git ls-files data/profiling`.
- **Bar wiring:** `BARS` is `{"v2", "v4"}` today (`evolve/policies/tuned/` does not exist yet), so `--bar v4t` fails at argparse rather than silently. Bar members are keyed by file sha.
- **Fork policy:** the `base-upstream-25e0082` tag exists and is an ancestor of HEAD; the upstream push URL is disabled.
- `tests/test_v4_harness.py`: 4 passed, and 3 passed + 1 skipped with `EVOLVE_V3_DATA=1`.

---

### [major] `code_version` and `baseline_key` ignore the values of `MAX_CONTEXT`, `KV_CAPACITY`, `KV_CAPACITY_DEVICE`, `MODEL_NAME` and `TP`

`evolve/evaluate.py:389` (hashes only `inspect.getsource` of four functions) and `:488-491` (`baseline_key`)

**Failure scenario:** `max_request_tokens` and `sim_command` refer to these tables by name, so only their source text is hashed, not the table contents. The cached-baseline key is `code_version|trace_sha|tag|scheduler`, and the tag contains neither the cap nor the model path. Plausible edits:
- set `MAX_CONTEXT["l3_8b"]` to 128000;
- fill in `KV_CAPACITY_DEVICE` (its comment says it will be "filled in as deployments are added") and later correct a value;
- change `TP`.

After any of these, cached baselines run under the old cap are reused. `_window_csv` does hash `max_request_tokens(sc)`, so the candidate runs on the new window while the bar members come from the old one. The completion gate cannot catch this, because each result was gated against its own window. A `MODEL_NAME` change is partly caught: the bundle key changes, so `score()` raises.

**Evidence:**
```
sc = Scenario('Synth',8,1.0,-1,0,1,model='l3_8b'); v=code_version(); k=baseline_key('lor',sc)
MAX_CONTEXT['l3_8b']=32768; KV_CAPACITY['l3_8b']=1; _VERSION.clear()
-> c16266a8ef49 c16266a8ef49   max_request_tokens 131072 -> 1   baseline_key unchanged: True
```
The same gap covers `_transformed_csv` and `window_csv`: neither is hashed, and the `|v2` literal in the transform key never reaches `baseline_key`. This does not affect v4 stages, which use no transforms.

**Suggested fix:** in `code_version`, also hash `repr(sorted(...))` of `KV_CAPACITY`, `KV_CAPACITY_DEVICE`, `MAX_CONTEXT`, `MODEL_NAME`, `TP` and `SEED_FLAG`, plus the source of `_transformed_csv`. Alternatively, add `max_request_tokens(sc)` and `MODEL_NAME[sc.model]` to `baseline_key`. No current number is wrong, because the tables have not changed since `c16266a8ef49` was cut, so this can wait for the next planned bump.

**Hashed file:** not a `vidur/` file, but either fix changes the `code_version` value or every baseline key, so it **invalidates cached baselines**. Do it between campaigns.

### [major] `test_code_version_covers_profiles` checks a source substring, not the property

`tests/test_v4_harness.py:30-32`

**Failure scenario:** the test only asserts that `'data" / "profiling'` appears in the source of `code_version`. It would still pass in each of these cases:
- the walk result is computed but never added to `files`;
- the walk points at a directory that does not exist (`os.walk` silently yields nothing);
- `file_sha` memoisation hides a changed file.

The `monkeypatch` and `tmp_path` fixtures are requested but unused. The property itself holds today (verified behaviourally above).

**Suggested fix:** a behavioural test. Monkeypatch `ev.REPO` to a temp tree holding `vidur/a.py`, `scripts/tail_metrics.py` and `data/profiling/m/x.csv`. Clear `ev._VERSION` and `ev._SHA`, compute the version, edit the CSV, recompute, and assert the two differ. I ran exactly this and got `c3d9d8b516b0 -> c047d357b3a5`.

**Hashed file:** no.

### [minor] `EVOLVE_V3_DATA` runs leave no trace in outputs and share v4's cache dir and keep paths

`evolve/evaluate.py:60-66, 93, 355-361, 444`

**Failure scenario:**
- `code_version` is identical in both modes, and `CACHE_DIR` stays `.cache_v4` rather than v3's `.cache`. On Alpine, `submit_evolve_driver.sbatch:47` defaults to `$SCRATCH/cache` for both releases. Baseline keys still separate the two modes through the trace sha, so no numbers mix.
- Outputs keyed only by `code_version` cannot tell the modes apart:
  - `evaluate_policy()` results;
  - bar_eval's JSON (`code_version` plus `provenance()`, which records neither trace shas nor `V3_DATA`);
  - the `EVOLVE_KEEP_DIR/<code_version>/<tag>/<who>` path. A v3-data run `rmtree`s and overwrites the kept v4 outputs of the same scenario and router.
- The `CACHE_DIR` comment "so v3 and v4 numbers never mix" is enforced by the keys, not by the directory.

**Suggested fix:**
- Add `"v3_data": V3_DATA` and `{t: file_sha(p)[:16] for t, p in TRACES.items()}` to `provenance()`.
- Put `file_sha(TRACES[sc.trace])[:12]` into the keep path.
- Optionally default `CACHE_DIR` to `.cache` when `V3_DATA` is set.

**Hashed file:** `provenance()` no. The keep path sits in `run_one`, which is hashed, so editing it changes `code_version`. To avoid that, compute the path in a helper that is not hashed.

### [minor] Bundle check: fingerprint is read from disk at harness time and memoised per process, and runs without a fingerprint pass the check

`evolve/evaluate.py:346-352, 566-568`

**Failure scenario:**
1. `_BUNDLE_SHA` is memoised for the life of the process. If `cache/prediction_bundle_<key>.pkl` is replaced during a long driver run (for example by shipping the Alpine bundle, as the Vidur memory note prescribes), later simulations load the new table but are recorded with the old sha. `score()` then accepts a mixture of old and new tables.
2. The sha describes the file on disk when the harness hashes it, not the table the simulation used. With N parallel first runs, each process trains and `os.replace`s its own bundle, and all runs report the sha of whichever write landed last. This only matters if training is not deterministic within one machine (unverified).
3. `"<key>:missing"` compares equal everywhere.
4. `score()` skips results without a `bundle`, so one fingerprinted run mixed with unfingerprinted ones passes.

**Suggested fix:**
- Have the simulator record the sha of the bundle it actually loaded or wrote in `predictor_info.json`. This is in `vidur/`, so it is hashed and invalidates baselines.
- Harness-only alternative: drop the memo, and treat a missing fingerprint or `missing` as a mismatch when any other run in the scenario has one.

**Hashed file:** simulator fix yes; harness-only fix no (`score` and `bundle_fingerprint` are not hashed).

### [minor] Converter session carry-over checks only timestamps, and 90–95 % of Tool/Conv timestamps are tied

`scripts/convert_mooncake_traces.py:185-188`

**Failure scenario:** Tool and Conv timestamps fall in 3 s buckets. The tied fraction of consecutive rows is 0.950 for Tool and 0.902 for Conv. If the `sessions_from` file had its rows reordered within a bucket (upstream's Conv file is not our converter's output), the assert would pass and sessions would land on the wrong rows. The current files are correct (verified via decode lengths and hash ids above); the guard is just weak.

**Suggested fix:** also read `num_decode_tokens` and assert it equals `output_length` row by row. Optionally assert that the first `min(P_v3, P_v4)//16` hash ids agree.

**Hashed file:** no.

### [minor] `--bar` / `--objective` defaults and the proposer prompt lag the v4 protocol

`evolve/driver.py:251`, `evolve/analysis/bar_eval.py:39-40`, `scripts/submit_evolve_driver.sbatch:34`

**Failure scenario:**
- Driver and bar_eval default to `--bar v2`, bar_eval to `--objective balanced`, and the sbatch leaves out `--bar` when `BAR` is empty. A v4 search or held-out evaluation launched without the flags is silently scored against the 7-member v2 bar, with balanced weights in bar_eval. SELECTION.md fixes the v4 bar plus tuned members and tail4. `launch_v4_campaign.sh` passes `BAR=v4t` correctly.
- `--bar` is silently ignored unless `--task router --protocol v2`.
- With `--bar v4` or `v4t`, the proposer prompt still describes the v2 bar (five heuristics plus Mooncake and Dynamo, with no power-of-two, JSQ-KV or tuned members).
- For search3b, the prompt describes only "conversation and tool/agent" workloads, though Synth is in the set.
- SELECTION.md §2 calls the pre-search bar "`BAR_V2`" while listing the v4t members.

**Suggested fix:**
- When `--search-stage` is a v4 stage, require `--bar` explicitly, or default it to `v4t` once that bar exists.
- Default bar_eval's `--objective` to `tail4`.
- Derive the prompt's bar paragraph from `BASELINES`.
- Fix the bar name in SELECTION.md.

**Hashed file:** no.

### [minor] Test gaps: no test of v4-split disjointness, the converter or lint

`tests/test_evolve_evaluate.py:55` checks only STAGE1 against HELDOUT.

**Failure scenario:** a later edit that shifts a test3b draw into heldout3 or search3b, or an off-by-one in `expand_block_hashes_unpadded`, would go unnoticed. Today both are correct: tag overlap is 0, and the slice rebuild is byte-identical.

**Suggested fix:**
- Assert that the (trace, perturbation) sets of search3b, heldout3, test3 and test3b are pairwise disjoint.
- Add a 3-row unit test of `expand_block_hashes_unpadded`, covering P multiple of 16, P not a multiple of 16, and D < 16 − P%16.
- Run lint on a corrupted copy, like my negative tests.

**Hashed file:** no.

### [nit] `MAX_CONTEXT` comment says "Llama-2 keeps v3 behaviour"

`evolve/evaluate.py:79`

Llama-2 runs now use the unpadded v4 traces; only the cap rule (KV capacity, not context) is unchanged.

**Suggested fix:** reword to "Llama-2: no context cap (as in v3); v4 traces".

**Hashed file:** no (comment outside the hashed functions).

### [nit] `code_version` hashes every profiling CSV, not only those a run uses

`evolve/evaluate.py:381-383`

Adding an H100 or 70B profile invalidates every cached A100 8B baseline. This is a deliberate safety-over-cost trade, noted only so it is not a surprise. The bundle key already identifies the exact profile files a run used.

**Hashed file:** changing it would change `code_version`.

### [nit] `mix-` transform drops `device`

`evolve/evaluate.py:267`

`Scenario(other, sc.replicas, sc.scale, -1, 0, 0, sc.model)` loses `device`, so the cap of an H100 mix scenario would use A100 capacity. This path is unused today.

**Hashed file:** no (`_transformed_csv` is not hashed, which is itself part of the first finding).
