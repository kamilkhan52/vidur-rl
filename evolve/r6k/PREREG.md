# r6k: routers re-selected on the RTX PRO 6000 profile — pre-registration

Written 2026-10-17, before any rtxpro6000 router result (the only rtxpro6000 run so far is a 1-replica round-robin
identity check on the C2 Conv window). Kamil's request via the realsys session: retrain/re-select the evolved router
family and the RL router on the RTX PRO 6000 profile, with the same objectives and pre-registered tests, at 4
replicas, time scales about 2–5.6, holding out the hardware windows. Changes to this file after the probe are
appended as dated amendments, never edited in place.

## 1. Simulator

- Branch `rtxpro6000` (repo vidur-rl). Device patch + profile from `~/vidur-rl-sim6000/handoff/rtxpro6000/`
  (SHA256SUMS verified), PCIe network device, KV pinned to 30,208 blocks (483,328 tokens) as on the hardware.
- Identity check: a 1-replica Conv C2 run is request-for-request identical (max |diff| 0.0) to sim6000's V-rtx run.
- `code_version` c4557954b77b. RL scoring uses correctly rounded sums (machine independent).

## 2. Trace format and hold-out

- The hardware replays windows cut from the **v3** traces (prompt lengths rounded up to whole 512-token hash blocks),
  so every r6k scenario uses that format: traces `Tool3`, `Conv3` (cap 131,072 tokens = the hardware's max length;
  no v3 request exceeds it, so row numbers equal the uncapped trace's).
- **Held out from everything except the final sim test:** `ToolHW` = realsys's `Tool_4000_2000_3281d502d2bbca0f`
  = Tool3 rows 4000–5999; `ConvHW` = `Conv_0_2000_2fe221fad356221c` = Conv3 rows 0–2003 less the 4 requests over
  122,752 tokens (it was cut with Llama-2-7B's cap). Verified byte-equal row by row.
- **Validation windows** (selection only; never trained or tuned on): Tool3 rows 20000–23607, Conv3 rows 10000–12030.
- Excluded from RL training and tuning: `Tool3:4000:2000, Tool3:20000:3608, Conv3:0:2004, Conv3:10000:2031`
  (asserted for every window by rl_es).

## 3. Loads

From the probe (`evolve/r6k/probe_load.py`: 4R, draw 1, 1000-request windows Tool3 w0 and Conv3 w3000, scales 1.5,
1.75, 2, 2.5, 3, 3.5, 4, 5.6), per trace, with the hardware pilot's rule applied to round robin's mean scheduling
delay in Vidur:
- **high** = smallest probe scale with rr mean queueing delay ≤ 1 s;
- **moderate** = smallest probe scale ≥ 1.4 × high;
- **light** = 5.6.

## 4. Stages (4 replicas, 1000-request windows, warm-up 20 %)

| stage | windows | loads | draws | use |
|---|---|---|---|---|
| `r6k_search` | Tool3 w0, w10000; Conv3 w3000, w7000 | high, moderate, light | 1, 4 | tuning (24 runs) |
| `r6k_val` | Tool3 w20000, w22000; Conv3 w10000, w11000 | high, moderate, light | 2, 3 | selection (24 runs) |
| `r6k_test` | ToolHW w0, ConvHW w0 (the hardware's 1000 requests) | 2, 2.5, 3, 4, 5.6 | 1, 2, 3 | final, once (30 runs) |

The `r6k_test` headline cells are the hardware pilot's two loads per trace, declared here as an amendment before the
test is run; the other scales are reported as they come.

## 5. Tuned baselines and the evolved router's constants (equal budget, same objective as on A100)

`evolve.analysis.tune_baselines`, stage `r6k_search`, objective `tail4`, budget 125, seed 0, bar `BAR_V4` on r6k:
- `mooncake` (QUEUE_W, LEN_QUAD, KV_W) and `dynamo` (PREFILL_SCALE, TEMPERATURE), as on A100;
- `v4873`: 4873's eight price/load constants DECODE_MISS_PRICE, MISS_EXTERNALITY, DECODE_WEIGHT, BATCH_PRICE,
  INTERFERENCE, CONTEXT_PRICE, LOAD_UNIT, RAMP_START, each searched in [¼, 4] × its value (log), defaults first.
  These are the terms that price prefill against decode and set the load scale, which C2 shows differ on this card.

**4873-r6k is adopted** iff on `r6k_val`, versus 4873: mean E2E ratio's 95 % upper bound < 1 and P99 E2E ratio's
95 % lower bound ≤ 1 (not significantly worse). Configuration-cluster bootstrap as on A100. Otherwise the evolved
router stays 4873.

## 6. RL router

`evolve.rl_plan.rl_es`, the b0 recipe on this profile, **base = 4873** (unchanged, so the router is "4873 plus a
learned correction" as on A100, independent of §5's outcome):

```
--device rtxpro6000 --arch linear --fitness gen8 --pairs 6 --sigma 0.03 --lr 0.05 --eval-every 10 --bound 0
--length 2000 --windows 16 --hours 15.5
--configs Tool3:4:{2.0,2.5,3.0,4.0,5.6}, Conv3:4:{same}, Tool3:8:{1.0,1.25,1.5}, Conv3:8:{same}
--exclude Tool3:4000:2000,Tool3:20000:3608,Conv3:0:2004,Conv3:10000:2031
--eval-starts Tool3:0/8000/14000,Conv3:2500/5000/7500   (learning curve, draws 40–43, eval length 2000)
```

- 8R at half the scale gives the same per-replica load as 4R; it is there for generality, not for the test.
- **Seeds 20, 21, 22, 23.** The A100 recipe replicated in 1 of 3 further seeds, so selection among seeds is part
  of the method and is declared here.
- **Checkpoint rule** (unchanged): per seed, the best gen8 on the learning curve, then `r6k_val`.
- **Pass test per seed** (unchanged): gen8 vs 4873 on `r6k_val`, 95 % upper bound < 1.
- **The RL-r6k router** is the passing seed with the lowest `r6k_val` gen8. Reported with the replication count
  (k of 4 seeds pass). If none passes, there is no RL-r6k router, and that is reported.

## 7. Final sim test (`r6k_test`, run once)

Run after the RL-r6k choice, the 4873-r6k decision and the tuned files are frozen, and after the pilot's loads are
declared (amendment). Routers: the A100 bar (16), the r6k-tuned Mooncake and Dynamo, 4873, 4873-r6k if adopted,
RL-r6k if any, and the A100 RL router es_gen8div_b0_40. Reported on all of mean/P50/P95/P99 of E2E, TTFT, TPOT, TBT,
ratio to 4873, cluster bootstrap over (trace, scale) with draws together. Claims, mean first, P99 as a guard:
- the evolved router (4873 or 4873-r6k) has lower mean E2E than every baseline;
- RL-r6k is non-inferior to 4873 on mean E2E within max(3 %, 2σ_rep) and reported on gen8.

## 8. Caveat to carry into the paper

The A100-trained routers saw the hardware windows' requests during their search (search3b used the full v4 traces
at 8R; different draws, scales and prompt format). Only the r6k routers have them held out.

## Files

- v3 traces (untracked, too large for git): `mooncake_toolagent_trace.csv` sha256 9c67924e4e8e1a50…,
  `mooncake_conversation_trace.csv` 55ca824e0f123d0e… (copies of `~/vidur-rl-working/data/processed_traces/`).
- Hardware windows: `data/r6k/Tool_4000_2000_3281d502d2bbca0f.csv` d6831336dd121f3b…,
  `data/r6k/Conv_0_2000_2fe221fad356221c.csv` d007257036497a44… (from `~/vidur-rl-sim6000/calib/r4/`).
- Stages and exclusions: `evolve/evaluate.py` (`R6K_*`); tests `tests/test_r6k.py`.

## Amendment 1 (2026-10-17, after the probe, before any tuning or validation run)

**Loads.** The probe (Alpine 34557370, `evolve/r6k/out/probe_load.json`, 64 runs, none failed) makes §3's rule
inapplicable on Tool:
- Tool3 w0: round robin's mean scheduling delay is 1.90 s at scale 4 and 0.92 s at 5.6. So high = 5.6, and
  "moderate ≥ 1.4 × high" lies off the grid.
- Conv3 w3000: rr queue 0.83 s at 4, so high = 4 and moderate = 5.6, as sim6000 predicts for the hardware.

The loads for `r6k_search` and `r6k_val` are therefore fixed at **2.5, 4.0, 5.6 on both traces**: 4 and 5.6 are
sim6000's predicted hardware loads, and 2.5 is the heavy end of the requested range. Router means at these loads
were seen only on the probe's search windows.

**Format wording (corrects §2).** The 512-token rounding concerns Tool only. In v3 Tool every prompt is a multiple of
512. In v3 Conv, prompts keep their real lengths (3 of the 2000 hardware Conv prompts are multiples of 512; realsys's
check). The verification is unaffected: Conv3 rows 0–2003 minus the 4 over-cap requests equal the hardware Conv file
row for row (`tests/test_r6k.py`).

## Amendment 2 (2026-10-18, before any RL validation result)

**RL relaunch.** The first RL jobs (Alpine 34557600/602/603/604) passed the §6 command through `sbatch --export`,
which splits on commas. Their command stopped after `--configs Tool3:4:2.0`, and the runs used the defaults:
- the a100 profile and `tails4` fitness;
- Tool3 at 4 replicas and scale 2.0 only;
- length 4000 and no exclusion.

Their outputs are kept as `out/es/INVALID_cmd_truncated_a100_r6k_gen8_s2*`. They were never validated and are not
used. The §6 recipe, unchanged and with eval length 2000, was relaunched through `evolve/r6k/es_seed.sh` as Alpine
34601469–34601472. Each run's `meta.json` was checked against §6 at start: device, fitness, lengths, ES settings,
16 configurations, exclusions and learning-curve starts. The tuning, probe and validation jobs had no comma in their
commands, and their scenarios set the device in code, so they are unaffected.

**4873-r6k (§5) outcome**, for the record: not adopted. On `r6k_val`, the mean E2E ratio to 4873 is 1.002
[0.999, 1.005] (`out/val_compare.json`).
