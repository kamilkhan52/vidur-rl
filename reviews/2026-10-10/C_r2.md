# Chunk C (golden-set tool), round 2

**Verdict: approve with minors. All seven round-1 findings are fixed; each round-1 failure scenario now fails loudly.**
**`golden check` can no longer report byte-identical for a cell whose hashed outputs changed. A real re-run of the extended set reproduces `v4ext` 5/5.**
Remaining items are small and all in non-hashed files: `--force` can still replace the pre-v4 reference, plus a few coverage and naming nits.

Repro harness (fake simulator replacing `ev.subprocess.run`, no vidur runs):
`/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/repro_golden_r2.py`.
Observe-ablation run: `.../scratchpad/noobs_run.py`. `tests/test_golden.py`: 11 passed.

## Round-1 findings: verification

| Round-1 finding | Status | Evidence (new code) |
|---|---|---|
| [major] EVOLVE_KEEP_DIR gives None hashes, accepted, compared equal | **fixed** | `run_all` raises `RuntimeError`. `main build/check` returns 2 and writes no manifest. `compare(None-hash, None-hash)` reports `request_metrics_sha256 missing, token_gaps_sha256 missing`, and `report` returns 1 (`0 of 2 compared cells byte-identical`). `validate` rejects `missing hash`. An empty `EVOLVE_KEEP_DIR=` (what the Alpine script passes) is treated as unset, and the cells hash normally. |
| [major] no cell reaches preemption/restart | **fixed (opt-in)** | `PREEMPT` (7b 8R Tool s0.75) has 160-201 restarted requests in all 4 cells of `v4ext`. `validate` fails the build if any `EXPECT_RESTARTS` cell has 0 (or lacks the column). `request_num_restarts` is `Request._num_restarts`, incremented only in `Request.restart()`, which `vllm_v1_replica_scheduler.py:227` calls on preemption. So the check measures real restarts. |
| [minor] both-failed / one-sided cells ignored; count overstated | **fixed** | fail/fail: 10 diffs, rc 1. ref=[]: 10 × `not in the reference`, `0 of 0 compared`, rc 1. ref with 1 cell vs 10: `1 of 1 compared … (reference 1 cells, new 10)`, rc 1. `report([], [])` → rc 1. |
| [minor] stale/ambiguous outputs | **fixed** | rmtree raising: the cell fails with `cleanup … failed`. rmtree as a silent no-op: `cleanup … failed: still exists`. Two outputs: `found 2 and 2`, fail. A request_metrics.csv and a token_gaps.npz in different directories also fail. |
| [minor] `%.6f` CSV; TBT stats missing | **fixed / documented** | `mean_tbt`, `p999_tbt`, `n_tbt` are in `SUMMARY_KEYS`, and the docstring states the precision limit. A 1e-7 s perturbation is caught by `token_gaps_sha256 differ` alone. See the nit below on sizing such a diff. |
| [minor] default name `v3_*` on v4 data; Alpine script | **fixed** | The default name follows `ev.V3_DATA` (`v3_`/`v4_`, `ext` suffix). `v3_data` is recorded in `environment`. An existing manifest is refused without `--force`, both via the default name and via `--out` of the same path. The script now sets `EVOLVE_V3_DATA=1` and `EVOLVE_KEEP_DIR=`, and it syncs the test policies. |
| [minor] tests don't cover the plumbing | **fixed** | The fake-sim tests cover the following: each cell hashes its own output, a change is detected, KEEP_DIR is refused, missing outputs are never identical, ambiguous outputs fail, cleanup failure fails the cell, naming and `--force` work, and a failed build goes to `.failed.json`. The committed-manifest test now runs `validate` and checks the cell set against `cells_for(set, include_pending)`. |

## Other checks (no finding)
- **Original manifests still compare under the new code.**
  - `check --manifest v3_Kamils-Mac-mini.json --against v3_alpine.json` gives `all 30 compared cells byte-identical`, rc 0. The env diffs are code_version c6fceae7bfa8 vs c16266a8ef49, python and pandas. The reverse direction is also 30/30.
  - v4: `1 difference(s); 29 of 30 compared cells byte-identical`, rc 1. The difference is the known `l3_8b-16R…p2 / lop_uncached: token_gaps_sha256 differ`, with every summary metric Δ0.
  - Old manifests lack `set`, `include_pending` and `v3_data`. They fall back to `default` / False, and `env_diffs` skips `v3_data` when either side lacks it. Re-running an old v3 manifest without `EVOLVE_V3_DATA=1` still shows a `traces` env diff and 30 hash differences, so it cannot pass by accident.
- **One real check (allowed run):** `check --manifest v4ext_Kamils-Mac-mini.json --workers 2` gives `all 5 compared cells byte-identical (141 s)`, rc 0, with no env diffs. The working tree's code_version is still c16266a8ef49. The `feedback` policy and the preemption cells are deterministic run to run.
  So the commit's "v4ext manifest reproduces" holds.
- **The observe policy is valid, and observe changes the outputs.**
  - It routes through `choose`, and ties go to `ctx.rng`, a seeded `random.Random`, so `choice` returns an `int` that passes `_check`.
  - Invariant violations raise. `_call` wraps them in `PolicyError`, which propagates out of `on_prefill_end`/`on_request_end` and fails the run.
  - Ablation: the same file with `observe` renamed to `_unused_observe` changes both hashes on both cells. PREEMPT: p99_e2e 47.279 → 47.020, restarted 160 → 168. KNEE: 102.43 → 103.19.
  - So a regression that silently drops `observe` calls (e.g. `_observe` returning early) would be caught.
- **Pending cells are excluded by default.** `cells_for("default")` has no `EXT_POLICIES` names. `cells_for("extended")` has the 5 non-pending cells, and `--include-pending` gives 8. `check` takes `set`/`include_pending` from the manifest, so a v4ext check never runs the route-delay/view-refresh cells. The pending flags exist in `vidur/config/config.py` (725, 1023).
- **Central question.** A cell counts as identical only if it is `ok` on both sides and both hashes are valid 64-hex strings and equal. Missing, None, failed and one-sided cells are all reported, and zero compared cells is rc 1.
  - Scoring reads only `request_metrics.csv` and `token_gaps.npz`, and both are hashed. So equal hashes imply equal scores.
  - The simulator runs `python -m vidur.main` with `cwd=REPO`. vidur is not installed in `.venv` (`import vidur` fails outside the repo), so the repo's code is what runs.
  - I found no remaining path to a false "byte-identical" for changed outputs. One scope limit is noted below: predictor-training code.

---

### [minor] `--force` can still replace the pre-v4 reference `v3_Kamils-Mac-mini.json` with a HEAD-code build
`evolve/analysis/golden.py:418-421`

**Failure scenario.** `EVOLVE_V3_DATA=1 python -m evolve.analysis.golden build --force` on the Mac writes `v3_Kamils-Mac-mini.json`. It replaces the only reference built from pre-v4 code (code_version c6fceae7bfa8) with one built from the current code. After that, `check --against` of v3 Mac vs Alpine compares HEAD with HEAD, and the evidence that v4 preserved v3 behaviour is gone from the working tree.
`--force` is the documented way to rebuild (the Alpine script header suggests it). Someone could plausibly use it to "refresh the v3 reference" after a v4.1 change.

**Evidence:** `v3 build --force rc 0 v3 kept: False` (fake simulator, `OUT_DIR` redirected to scratch). The naming path itself is fine: without `--force`, both the default name and `--out <same path>` are refused (rc 2).

**Suggested fix.** With `--force`, refuse if the existing manifest's `environment.code_version` differs from `ev.code_version()`, unless `--out` names a new file. Alternatively, rename the frozen reference to something no default can produce (e.g. `v3_pre-v4_Kamils-Mac-mini.json`). The file is git-tracked and recoverable, hence minor. Hashed file: no.

### [nit] Prediction-training code is outside what the golden set can see
`evolve/analysis/golden.py:29-30` (coverage list)

`_bundle_key` hashes the predictor config and the profiling CSVs, not code. With the bundle present (which `main` requires unless `--allow-train`), a change to the forest training or featurisation code in `vidur/execution_time_predictor/` reuses the old table. The golden check then truthfully reports identical outputs, but a machine without the bundle would simulate differently. This is by design (shipped bundles), but the "Not covered" list should name it. Hashed file: no.

### [nit] A token_gaps-only difference still cannot be sized when every summary statistic is equal
`evolve/analysis/golden.py:347-350`

The v4 Mac/Alpine cell prints `token_gaps_sha256 differ` with every metric Δ0. Old manifests lack `mean_tbt`/`p999_tbt`/`n_tbt`, so those show as `None->x` against new runs. Even with them, a last-ulp gap difference that leaves all percentiles equal is reported only as "differ". If pooling Mac and Alpine results depends on that cell, a `max |Δgap|` needs the two npz files, which only `--keep` on both machines provides. Optional: note this in the docstring. Hashed file: no.

### [nit] Small CLI/edge-case rough spots
- `--include-pending` is silently ignored with `--set default` (`cells_for("default", True)` gives 30 cells). `build --set extended --include-pending` uses the same default name as the non-pending reference (`v4ext_<host>.json`), so with `--force` the non-pending reference is replaced by a superset. That is harmless for `check`, which reads `include_pending` from the manifest, but the file's meaning changes.
- `run_all(..., cells=[])` falls back to the default set (`cells = cells or …`, line 284). Use `if cells is None`.
- `test_committed_manifests_have_schema` globs `*.json`, so a local untracked `*.failed.json` makes it fail. Glob `v*_*.json` and skip `.failed`.
- `EXPECT_RESTARTS` is calibrated on v4 data. An extended build under `EVOLVE_V3_DATA=1` may fail validation (loudly, to `.failed.json`). Worth one docstring line.

Hashed file: no, for all of these.
