# EXP33: prior-work routers vs 4873 and es — pre-registration

Written 2026-10-26 by the priorwork session (Kamil's request of 2026-10-26, relayed in `~/vidur-rl-priorwork/BRIEF.md`),
before any comparison run. Vault note: `Vidur_EXP33_Prior_Work_Routers`. Changes after the first comparison run are
appended as dated amendments, never edited in place.

## 1. Question

On the scenarios the paper uses, do 4873 (evolved) and es (es_gen8div_b0_40, the RL correction) beat the published
learned and LLM-designed routers we cite, on mean latency (E2E, TTFT, TPOT, TBT), with P99 as a guard and medians and
goodput capacity reported? What does each router cost per decision?

## 2. Simulator and reproducibility

- Branch `priorwork` of vidur-rl, from `h100` at f0214d6. Additions: router-side hold (`dispatch()` policies,
  re-offered on every arrival and after every batch that completed a request, as in Glia's Vidur fork; commit
  8cbb89d), the ports in `evolve/policies/priorwork/`, `evolve/priorwork/`.
- Identity checks (behaviour-preserving for routers without `dispatch()`):
  - H100: 4873 on ToolHW rows 0–999, 8R, scale 0.8973, draw 1 equals sim6000's EXP24 run
    `rv/h100-8b-r8-df/Tool_h1000_s0.8973_p1/4873` request for request (max |diff| 0.0) at code_version
    `c18b7fc0cb62` (`evolve/priorwork/identity_h1.py`).
  - A100: 4873 on test3b `l3_8b-16R-Tool-s0.5-full-p9` equals vidur-working's cached EXP15 run on all 21 summary
    metrics (max |diff| 0.0; checked at cd416352bffe, before the hold change; re-checked at the final code version
    before the A100 runs).
- Every router of one scenario set runs in this tree, on one prediction bundle: H100 `4534b960326a7810:0c948f08d024`,
  A100 `8e6e78e985b87b2a:82b720c5cc7e`, RTX the r6k Mac bundle (fingerprint recorded with its identity check, before
  the RTX runs). Reference routers are re-run here, not taken from other trees, so each table is one code version and
  one bundle (registry §3 threat 7).

## 3. Routers

### 3.1 Prior work (ported; sources and full deviation lists in each file's docstring and in `~/vidur-rl-priorwork/notes/`)

| name | source (URL @ commit, licence) | what it decides on | fidelity |
|---|---|---|---|
| glia | Glia HRA router, arXiv 2510.27176 v5 Fig. 10 (printed only; not released) | router queue, shortest prompt first; KV head-room admission (1.6 × prompt, 3 % free); lowest projected usage | full rule, constants as printed; waiting-prompt tokens from the router's ledger |
| engram | Engram Fig. 21 = Zenodo 20015246 run 9 (mit-nms/Engram @ 5295858, MIT) | router queue, shortest prompt first; per-replica waiting cap 1/4 (decode/no decode); KV ≤ 95 % | full rule; replicas run vllm_v1, not Sarathi |
| lodestar | gangmuk/Lodestar @ 23da2a0b, Apache-2.0 | online-trained MLP predicts −TTFT per replica; K = 2 prefix candidates under KV saturation; AIBrix fallback | offline rows from training windows (RUN's start-up retraining), then online; decode tokens estimated; dropout off at inference |
| gorgo | atoniolo76/GORGO @ 4d6adafc (MIT in pyproject) | uncached prompt tokens (router trie) + 0.5 × queued prompt tokens | network term dropped (co-located replicas); weight 0.5 as learned by GORGO's ES |
| llumnix | llumnix-ray v0.1.0 @ 312edd8a, Apache-2.0 | highest freeness (free KV − waiting prompt blocks) / requests | dispatch only: no migration (the paper's gain), so ≈ its INFaaS++ baseline |
| dualmap | ASISys/DualMap @ 24816acc, MIT | two consistent-hash candidates per adaptive prefix key; cache hit first, less-loaded one if expected TTFT > SLO/2 | the paper's no-rebalance ablation: no router queue or rebalancing; prefill rate calibrated per device |
| astrolabe | AKafakA/Block @ b2141b3e, MIT | power of two choices on predicted E2E from simulating each candidate replica | added by amendment before its first run (port in progress) |

Settings are fixed now, from the paper or the released code; nothing is tuned on any validation or test set. Device
constants: DualMap's prefill time per token is measured on a validation window (`calib_prefill.py`; H100
5.078e-05 s/token); Lodestar's offline rows come from training windows only (`lodestar_collect.py`; H100: Tool3 rows
8000–9999 and Conv3 2004–4003, 8/16R, heavy and high, draw 4; 15,553 rows, sha256 d4301a3e…). A100 and RTX variants
are built the same way (A100 from search3b, RTX from r6k's search windows) before their runs.

Not ported, cited: **OLP** (2607.03948): its code uses true output lengths and holds and re-sorts requests at the
router; without true lengths it is a different policy. **Intelligent Router** (2408.13510): no code; its DQN's
hyper-parameters are not published and its "no action" needs a central queue with a length classifier on prompt text
we do not have. **Preble, Mooncake, AIBrix, SGLang, Dynamo** are covered by existing ports (Mooncake, Dynamo, AIBrix
prefix+load is Lodestar's fallback).

### 3.2 Reference routers (re-run in this tree)
4873 (`evolve/finalists/v4/v4_487363424749.py`), es (`evolve/finalists/rl_es/es_gen8div_b0_40.py`), Mooncake default
and tuned (A100 tuned seed 0; on RTX also mooncake6 = r6k-tuned), Dynamo (and dynamo6 on RTX), LOR, sticky-LOR.

## 4. Scenario sets (every set has a heavy-load cell)

| set | system | runs per router | contents |
|---|---|---|---|
| S1 `h1` | H100 (h100df), Llama-3-8B TP1 | 54 | EXP31's `h1_test` + `h1_test_mod`: Tool3 14000/16000, Conv3 6000 (2000 requests), 8R at 0.5982 / 0.8973 / 1.256 and 16R at half, draws 1–3. **Heavy** = 0.5982 (8R), 0.2991 (16R) |
| S2 `test3b` + `test3b_heavy` | A100 v4.1 tables (EXP15), contrast only | 45 + 6 | test3b as in EXP15 (8R 1.0 / 0.75, 16R 0.5; Tool/Conv p9–p13, Synth p8–p12); heavy: 8R Tool/Conv full trace at 0.6, draws p9–p11 |
| S3 `r6k_test` + `r6k_heavy` | RTX PRO 6000 profile, 4R | 36 + 6 | r6k_test (ToolHW/ConvHW first 1000, loads 3.0 / 4.0 / 4.8, draws 1–6); heavy: load 2.0, draws 1–3 |

h1_test is EXP31's held-out test set; it is only run here with settings fixed above (simulation session's rule,
2026-10-26). Runs seen before this pre-registration (port checks, no setting changed afterwards): one h1_test run
(Tool3 14000, 8R heavy, draw 1) of 4873, es, Mooncake tuned, gorgo, llumnix, dualmap, lor; the h1 validation window
Tool3 20000, 8R heavy, draw 2 for all ports.

## 5. What is reported, and how it is read

- Per router and metric (mean, P50 and P99 of E2E, TTFT, TPOT, TBT): geometric mean of router / 4873 over the set's
  runs, 95 % cluster bootstrap over configurations (draws of one configuration together; `es_validate.compare`),
  wins. Overall and per load (heavy reported separately). Same against es.
- Reading, fixed now, for each prior-work router X and each set: **4873 beats X** if the lower bound of X/4873 mean
  E2E is > 1; **X beats 4873** if its upper bound is < 1; otherwise not distinguishable. P99 E2E/TTFT read the same way,
  as a guard. es vs X likewise. No multiple-comparison claim across routers; the table is descriptive.
- Goodput capacity: highest load at which ≥ 90 % of requests meet TTFT ≤ T and TPOT ≤ P, (2 s, 50 ms) and
  (5 s, 100 ms), log-linear in arrival rate between tested loads, bounds marked (S1 and S3; S2 has two 8R loads plus
  the heavy cell, so bounds only).
- Decision overhead: mean, P99 and max wall time of each routing call and of `observe()`, on validation windows
  (Tool3 20000, heavy, 8R and 16R; `overhead.py`; machine recorded).
- **Hardware follow-up (agreed with vidur-realsys, 2026-10-26):** the 1–3 prior-work routers with the lowest S3 mean
  E2E ratio (ties: S1) and an upper bound below that of the best tuned baseline go to realsys with their decision
  inputs, overhead, file, commit and sha256. Routers that need Vidur-only state (astrolabe's timing model) are flagged.

## 6. Compute

Mac mini, niced, ≤ 4 workers (S1, S3, overhead, data collection), Cashew for S2 if the VPN is up. Estimate:
S1 ≈ 14 routers × 54 × 35 s ≈ 7 CPU-h; S2 ≈ 14 × 51 × 5–8 min ≈ 60–95 CPU-h; S3 ≈ 2 CPU-h; astrolabe adds up to
1.5–6× its own runs; collection and calibration ≈ 3 CPU-h. Total ≈ 75–110 CPU-h.

## Files
`evolve/priorwork/compare.py` (sets, scoring, goodput), `overhead.py`, `calib_prefill.py`, `lodestar_core.py`,
`lodestar_collect.py`, `identity_h1.py`, `routers.json`; ports in `evolve/policies/priorwork/`; tests
`tests/test_priorwork.py`.

## Amendment 1 (2026-10-26, before any comparison result was read; S1 had started at 619b9b4 with the 13 other routers)

- **astrolabe added** (§3.1 row): `evolve/priorwork/astrolabe_core.py` + `evolve/policies/priorwork/astrolabe_h100df.py`.
  Power of two choices on predicted E2E; each probe rebuilds the candidate replica from the router's ledger and runs
  our `VLLMV1ReplicaScheduler` forward with Vidur's own execution-time predictor (the run's bundle, key checked,
  never trained). Output lengths: running mean of observed decode tokens (prior 256); a request past its estimate
  ends within 10 tokens (Astrolabe's rule). Deviations listed in the file (DEVIATIONS): mean-length estimator instead
  of RoBERTa; decode progress dead-reckoned; **the timing model is the simulator's own, so its predictions have no
  model error, which favours astrolabe**; vllm_v1 instead of vLLM 0.7.2 V0. Port checks on the h1 validation window
  (Tool3 20000, 8R heavy, draw 2): completes 2000/2000; predicted vs actual E2E Pearson 0.46 (0.89 with true
  lengths); choose() 10.1 ms mean, 45 ms P99. Needs Vidur-only state (the timing model): simulation only unless a
  hardware-side predictor is substituted. Runs on S1 after the 13 routers, then S2/S3 with device variants.
- **Core modules frozen:** lodestar_core.py and astrolabe_core.py are frozen at this commit; compare.py records their
  sha256 in every report (cached runs are keyed by the policy file only). Any change to a core needs a new policy
  file name.
- **Disk:** kept outputs drop Vidur's plots/ folder (3.0 of 3.6 MB per run; unused); request_metrics.csv and
  token_gaps.npz are kept.

## Amendment 2 (2026-10-27, before any comparison result was read)

- The first S1 attempt (started 2026-10-26 22:16 at 619b9b4) was lost when the Mac's disk filled (ENOSPC, about
  22:30–06:00); no S1 result was read or kept (the run cache was last saved before it started). S1 restarts with all
  14 routers (amendment 1's astrolabe included) from 6780e22 + this amendment.
- compare.py now runs one router at a time (each router's runs are saved when it finishes) and stops if less than 5 GB
  is free.
- lodestar_core.py: the collection-mode dump appends one pickle per flush instead of rewriting the whole file (only
  the collection path changes; routing is unchanged). New core sha256 recorded in each report (core_sha256).
- Device constants measured before the outage (validation or selection windows): DualMap prefill time per token RTX
  6.732e-05 s (r6k validation window Tool3 20000, 4R, scale 32, draw 2), A100 1.143e-04 s (Tool full trace, 8R,
  scale 6.0, draw 2, a heldout3 selection draw). Lodestar RTX offline rows: 7,726 rows from r6k search windows, sha256
  28df7579…; A100 rows to be collected (search3b, Tool/Conv 8R 0.75, draw 1).
