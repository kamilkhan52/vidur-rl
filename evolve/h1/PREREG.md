# h1: declared multi-seed RL selection and the oracle ceiling on H100, many replicas — pre-registration

Written 2026-10-26, before any h1 training, validation or test run. The only h1 simulation so far is the identity
check in §1. Vault notes: EXP31 (`Vidur_EXP31_RL_Method_Multiseed_H100`) and EXP32 (`Vidur_EXP32_RL_Oracle_Ceiling`).
Changes after the first training run are appended as dated amendments, never edited in place.

Kamil, 2026-10-26: "Let's do your clean version recommendation of declaring, training, selecting, and testing";
"use only the regime we identified is useful such as new hardware and more replicas"; port the oracle fix and
rerun the buggy result; run on Cashew (and the Mac mini when it is free), not on Alpine. Domain rule (Kamil,
2026-10-26, EXP30): newer and larger systems for all experiments.

## 1. Simulator

- Branch `h100` (repo vidur-rl), from `rtxpro6000` at 39f63b2: the r6k harness, machine-independent ES scoring,
  and the `--oracle` fix (7bc66fd + test 23a3d64).
- Device `h100df` (`evolve/evaluate.py`): Vidur's h100 profile, network `h100_pairwise_nvlink`, KV pinned to
  30,208 blocks, and the decode-attention fix (`data/profiling/compute_decodefix/h100/meta-llama/Meta-Llama-3-8B/
  attention.csv`, sha256 826d45ffe57a4c0d…, from vidur-rl-decodefix via sim6000 `vendor/MANIFEST.tsv`).
- Prediction table: sim6000's bundle `prediction_bundle_4534b960326a7810.pkl` (sha256 0c948f08d02491b3…), which
  this branch computes the same key for. Every h1 run, on any machine, uses this file.
- Identity check (2026-10-26, Mac): 4873 on the first 1000 requests of ToolHW, 8 replicas, scale 0.8973, draw 1 is
  request for request identical (max |diff| 0.0) to sim6000's EXP24 run `rv/h100-8b-r8-df/Tool_h1000_s0.8973_p1/4873`.
- `code_version` cd416352bffe (`evolve.evaluate.code_version()` at this commit; recorded by every run).

## 2. Regime (why H100, 8 and 16 replicas, high to heavy load)

From EXP24 (corrected tables) and EXP22, before this campaign:
- among the domain systems, the learned router's edge over the best tuned baseline is largest on H100 Llama-3-8B
  with 8 replicas of TP1 (−4.4 % mean E2E at high load, −7.7 % at heavy); fewer, larger replicas shrink it;
- mean-latency room for RL appears under heavy load and grows with replica count (A100 es −3.7 / −6.2 / −2.0 % at
  4 / 8 / 16R at overload; RTX ES at 4R hardware loads ≤ 0.3 %).

Loads: EXP24's h100-8b-r8-df knee H = 0.8973 (round robin's mean engine queue ≤ 1 s on the h1000 windows).
heavy = H / 1.5 = 0.5982, high = H = 0.8973, moderate = 1.4 H = 1.256 at 8 replicas; 16 replicas at half the scale
(same per-replica load): 0.2991, 0.4487, 0.628. Training also uses an overload level (8R 0.4, 16R 0.2).

## 3. Trace format, windows and hold-out

Hardware trace format (v3), traces `Tool3` (23,608 rows) and `Conv3` (12,031 rows), as r6k and EXP24.
2000-request windows, warm-up 20 %.

| use | Tool3 rows | Conv3 rows | draws |
|---|---|---|---|
| test `h1_test` (once) | 14000–15999, 16000–17999 | 6000–7999 | 1, 2, 3 |
| validation `h1_val` (selection only) | 20000–21999, 21600–23599 (overlap 400 rows) | 10000–11999 | 2, 3 |
| learning curve (checkpoint choice) | starts 0, 8000, 11000 | starts 2004, 4000, 8000 | rl_es EVAL_DRAWS |
| never used (hardware windows) | 4000–5999 | 0–2003 | — |

ES training windows may not touch the test, validation or hardware ranges (`--exclude`, asserted per window).

## 4. RL recipe (EXP31), four seeds

The b0 linear recipe that produced es_gen8div_b0_40 on A100, base = 4873:

```
--device h100df --arch linear --fitness gen8 --pairs 6 --sigma 0.03 --lr 0.05 --eval-every 10 --bound 0
--length 2000 --windows 16 --hours 15.5 --eval-len 2000
--configs Tool3:8:{0.4,0.5982,0.8973,1.256}, Conv3:8:{same}, Tool3:16:{0.2,0.2991,0.4487,0.628}, Conv3:16:{same}
--exclude Tool3:4000:2000,Tool3:20000:3608,Tool3:14000:4000,Conv3:0:2004,Conv3:10000:2031,Conv3:6000:2000
--eval-starts Tool3:0/8000/11000,Conv3:2004/4000/8000
```

- **Seeds 40, 41, 42, 43** (fresh). Script `evolve/h1/es_seed.sh`.
- **Checkpoint per seed:** the best gen8 on its learning curve (eval at iteration k scores `params_{k+1}`).
- **Validation per seed:** that checkpoint vs 4873 on `h1_val` (36 runs: 3 windows × {8R, 16R} × heavy/high/
  moderate × draws 2, 3), configuration-cluster bootstrap (draws together), 95 %. A seed passes iff gen8's upper
  bound < 1.
- **Selection (the method, declared now):** the passing seed with the lowest `h1_val` gen8. If no seed passes, the
  method finds no policy; that is the result, and `h1_test` is not run for selection (see the descriptive run below).
- **Test, once (`h1_test`, 36 runs: 3 windows × {8R, 16R} × heavy/high × draws 1–3; 12 clusters).** The selected
  policy vs 4873. **Pass iff all hold:**
  1. mean E2E: 95 % upper bound < 1 (Kamil's primary objective, mean latency);
  2. gen8 (four means + four P99s, equal weight): upper bound < 1;
  3. none of mean E2E/TTFT/TPOT/TBT, P99 E2E/TTFT/TPOT/TBT has a lower bound > 1 (P99 is a guard);
  4. P50 E2E and P50 TTFT lower bound ≤ 1.
- Reported with the test, not judged: `h1_test_mod` (moderate load, 18 runs), and the replication count (k of 4
  seeds pass validation).
- **Descriptive replication (declared now):** after the selection is frozen and written to `out/h1_selection.json`,
  every seed's checkpoint is also run on `h1_test`. These numbers describe how reliable the recipe is on fresh data;
  they never change the selection or the claim.

## 5. Oracle ceiling (EXP32)

The policy may read each request's true output length (`--oracle`; a diagnostic, never deployable).
- **(a) A100 rerun of the invalid run.** `es/lin4_oracle_s3` (2026-10-14, Alpine 34129012) never read true lengths
  (bug found 2026-10-19). Rerun on vidur-working (`openevolve-rl-search` 3957a30, A100 v4.1, shipped A100 tables,
  bundle 8e6e78e985b87b2a) with its exact arguments from its `meta.json` (linear, tails4, 4 configs Tool/Conv 8R
  0.75/1.0, seed 3, pairs 6, sigma 0.05, lr 0.05, 11.5 h). The buggy run used predicted lengths with everything else
  equal, so it is the matched non-oracle control. Readout: learning-curve tails4 and gen8 at iterations 10–80,
  oracle vs buggy. Descriptive.
- **(b) H100 arm.** Seed 40 of §4 with `--oracle`, everything else identical. Readout: best learning-curve
  checkpoint's gen8 and mean E2E (ratio to 4873 on the learning-curve windows), oracle vs non-oracle seed 40.
  Reading, fixed now:
  - oracle mean E2E ≥ 0.99: perfect lengths give < 1 % mean-latency room here, so the length information is not
    what limits RL in this regime;
  - oracle mean E2E ≤ 0.98 and at least 1 point below non-oracle seed 40: length prediction is the lever;
  - otherwise: inconclusive at this budget.
  The oracle policy is never validated or tested as a router.

## 6. Compute

Cashew (Slurm, `coe_cpu` or `all`, 32 CPUs per ES run, niced), and the Mac mini when sim6000 and decodefix
release cores. Not Alpine. Every h1 run must use the §1 bundle; a run whose recorded bundle fingerprint differs is
invalid.

## Files

- `evolve/evaluate.py` (`VIDUR_DEVICE`, `DEVICE_ARGS["h100df"]`, `H1_*`, stages `h1_val`, `h1_test`,
  `h1_test_mod`); tests `tests/test_h1.py`.
- `evolve/h1/es_seed.sh`, `evolve/h1/validate.py` (checkpoint choice, validation, selection, test, descriptive
  replication), `evolve/h1/cashew.sbatch`.

## Amendment 1 (2026-10-26, before any training run)

Kamil, 2026-10-26, relayed by the registry session: "Add heavy load test for RL everywhere", and report system
throughput next to the per-request metrics.
- **Heavy load:** already in every h1 stage. `h1_val` and `h1_test` both include heavy load (8R 0.5982, 16R 0.2991),
  the learning curve covers all 16 training configurations including heavy and overload, and the §5b oracle readout
  uses that learning curve. Nothing changes. §5a keeps the original run's configurations (A100 8R at 0.75 and 1.0;
  0.75 is the A100 knee), because it has to match the buggy run.
- **System throughput (reported, never judged; the §4 pass rule is unchanged).** The `h1_test` and `h1_test_mod`
  runs keep their request-level metrics (`EVOLVE_KEEP_DIR`). Per (window, replicas, draw), for the selected policy and
  for 4873, report **goodput capacity**: the highest load at which ≥ 90 % of requests meet both TTFT ≤ T and TPOT ≤ P.
  Two absolute SLO levels are used, as in EXP30: (2 s, 50 ms) and (5 s, 100 ms). Capacity is interpolated
  log-linearly in arrival rate between heavy, high and moderate load; a value outside that range is reported as a
  bound. Also reported: served tokens per second per GPU at each load. The script is `evolve/h1/goodput.py`, written
  before the test is run.

## Note (2026-10-27, after all h1 results): prior-work routers merged

Branch `priorwork` (EXP33, tip e34154e) was merged into `h100` at b925694, at Kamil's request relayed by
vidur-priorwork. The merge adds a router-side hold (`dispatch()` policies) to `vidur/`, and that changes
`code_version` to c18b7fc0cb62. For every router without `dispatch()` the behaviour is unchanged: the §1 identity
check, rerun after the merge, is again request for request identical (max |diff| 0.0). Every h1 result above was
produced at cd416352bffe; runs at the new code version recompute their 4873 baselines.
