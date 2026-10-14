# SELECTION.md: pre-registered selection and test protocol (v4 campaign)

Status: **IN FORCE from 2026-10-10** (Kamil's decisions in §8, committed before any v4/v4.1 held-out run of a
searched policy; the campaign simulator is v4.1, `code_version` cbeb5161f046). From now on, change it only by
appending a dated amendment that says which results had already been seen at that time. Never edit earlier text.

Scope: the v4 simulator campaign for both contributions, the evolved router (workstream D) and the RL router
(workstream E), on the T1 headline deployment: Llama-3.1-8B TP1 on A100, 8 or 16 replicas, using the
Meta-Llama-3-8B profile. Generalisation deployments (T1 70B, T2 models, H100) follow §7.

## 1. Data splits

The `pN` suffix is the perturbation seed: `100N + 1`, plus in-bucket jitter. "Load" is the trace time-scale.

| Split | Use | Configurations | Draws (Tool/Conv · Synth) | Status at drafting |
|---|---|---|---|---|
| search3b | Search fitness; RL training and rollouts | 8R at loads 1.0 and 0.75 | p1, p4 · p7 | Used by v3 pilots (allowed) |
| heldout3 | Choosing among frozen candidates; RL checkpoint selection | 8R at 1.0 and 0.75; 16R at 0.5 | p2, p3, p5 · p1–p3 | Used for v3 pilot selection: selection-only, never a claim |
| **test3b** | **One-time test of the frozen finalists of both methods** | Same shape as heldout3 | p9–p13 · p8–p12 | Bar baselines computed on v3 for p9–p11; **no policy evaluated** |
| test3c | Fallback test for any policy frozen after test3b has been opened | Same shape | p17–p19 · p16–p18 | Unseen |
| test4 | Ensemble or portfolio experiment (option B or E); unseen loads | 8R at 0.85 and 1.25; rate-shift scenario (built in v4) | p14–p16 · p13–p15 | Unseen |

Notes:
- test3b grows from 3 to **5 draws** (p12 and p13 added; Synth p11 and p12) to meet the ≥5 paired-seed rule. This is
  decided now, before any policy is run on it.
- test3 (p6–p8) is spent: it was used once for the v3 pilots and is not reported.
- Draws p4–p6 were used on Llama-2 TEST_V2. They are reused here only as search draw p4, which is allowed.

## 2. Baseline bar

- **Bar before the first v4 search** (search fitness is computed against it): bar `v4t` (`evaluate.BARS["v4t"]`:
  `BAR_V4` plus the two tuned members), i.e.
  - LOR, sticky-LOR, round robin, random, LOP-uncached;
  - our Mooncake and Dynamo re-implementations;
  - power-of-two choices and JSQ variants, on outstanding requests and on KV usage;
  - Mooncake and Dynamo constants tuned by an **equal-budget** BO/ES control, the falsifier.
    - Tuning uses search3b only.
    - Its budget equals one evolve run's candidate count (125 vs 60 iterations x 2 proposals + 3 seeds = 123).
    - The tuned constants are frozen with our finalists.
    - The tuner scores its candidates against `BAR_V4` (the tuned members do not exist yet), while the evolve runs
      score against `v4t`, so a tuner fitness and an evolve archive fitness are never compared directly. Tuned
      control and evolved finalists are compared only through `bar_eval` (and `head_to_head`) on the same bar.
- **Added after the freeze** (test-time bar members, reported as added after the freeze): Lodestar and Astrolabe;
  Llumnix, OLP and global Mooncake if time allows. New baselines never re-select our finalists.
- Every comparison is reported **two ways**:
  1. against the per-configuration best of the bar (strict oracle bar);
  2. head-to-head against each baseline separately.

## 3. Metrics and the primary order

- **Primary order, fixed now:**
  1. P99 E2E (= TTLT);
  2. P99 TTFT;
  3. token pace: P99 TPOT over requests and P99 TBT over tokens.
- Search and selection objective: tail4. That is equal weights on the log ratios of the four P99s against the bar;
  fitness = −(mean + 0.5 × worst) − mean-E2E penalty, unchanged from v3.
- Secondary metrics are reported, never used for selection: SLO attainment, DistServe goodput, P99.9, mean E2E,
  router overhead, prefix hit ratio.

## 4. Freeze and finalist rules

- **Evolved:**
  - At least 3 independent search runs per method on search3b.
  - Re-evaluate the top 5 programs of each run on heldout3.
  - Freeze **at most 3 finalists** across all runs, by heldout3 fitness.
- **RL:**
  - At least 5 training seeds per method variant (planning, expert iteration, model-free control).
  - Checkpoint and hyperparameter selection use heldout3 learning curves only.
  - Freeze **at most 3 finalists** in total: the best planning-based finalist, the best distilled or fast finalist,
    and the best model-free control.
  - An RL finalist is the policy as frozen. If it runs online lookahead, its compute budget per decision is fixed at
    the freeze and reported, and it is labelled not deployable at serving time. Planners use sampled (never true)
    output lengths.
- **Primary finalist per method:** for evolved, the frozen finalist with the best heldout3 tail4 fitness, with ties
  broken by better heldout3 P99 E2E. For RL, the online-lookahead (planning) finalist is primary (Kamil, 2026-10-10);
  the distilled network is reported alongside as the deployable RL policy. Both are recorded in a dated amendment
  **before** test3b is opened.
- **Joint freeze:** no fixed date; it happens when RL is ready (Kamil, 2026-10-10). The evolved finalists are frozen
  when their search ends and stay untested until then; both methods are then tested on test3b in one run. Any policy
  frozen after test3b is opened is tested on test3c and reported as frozen later.
- Frozen files are copied to `evolve/finalists/v4/` with their sha256, `code_version` and bundle key recorded in the
  amendment.

## 5. Claims and decision rules (test3b; 95 % cluster bootstrap over configurations, paired)

- **Evolved headline claim.** The primary finalist improves on the strict bar if the upper CI bounds of both its P99
  E2E ratio and its P99 TTFT ratio are below 1. Token pace is non-inferior if the upper CI bounds of TPOT and TBT
  are ≤ 1.05. Each tail is reported with its CI regardless.
- **RL claims, two of them:**
  - **(a) vs the bar:** the same rule as the evolved claim.
  - **(b) vs the evolved primary finalist, on the same draws:**
    - **superiority:** the upper CI bound of the paired ratio of the geometric mean of the four tail ratios is
      below 1;
    - **non-inferiority:** that same upper bound is ≤ 1.05 (the margin calibrated on v3 pilots).
    - Report which held.
- **Regime split:** knee, load 1.0 and 16R are reported separately; no claim is made beyond the regimes tested.
- **Hindsight oracle:** reported as an upper bound and as the gap closed by each method,
  `(bar − method) / (bar − oracle)`. It is not a competitor.
- Every finalist is reported, not only the primary one. Multiple comparisons: primary-finalist claims are
  confirmatory; everything else is exploratory and labelled as such.

## 6. Robustness, reported but not used for selection

- Predictor perturbations of ±5–10 % (crude and reseed variants) plus a noise floor.
- Router seeds: at least 5 for any stochastic policy.
- Tool hash-chain-session sensitivity arm.
- The knee-dense load curve.

## 7. Generalisation deployments (fresh draws only, never used for selection)

- **GEN_L3.1-70B:** TP4, loads 2.0 and 3.0, draws p2 and p3. Optionally recalibrate on p1.
- **H100 twins** of T1.
- **T2:** ShareGPT/LMSYS on the 4K-profiled models. Each model is calibrated on its own p1 draw; tested on p2–p4.
- Frozen finalists are run unchanged. Re-searching per deployment is a separate, labelled experiment (open question
  §3 of the plan).

## 8. Decisions (Kamil, 2026-10-10)

1. Non-inferiority margin 1.05 for token pace and for RL-vs-evolved.
2. Joint freeze: whenever RL is ready (§4); evolved finalists wait, frozen and untested.
3. test4 (portfolio / learned selector) is deferred; it stays untouched.
4. At most 3 finalists per method.
5. RL primary finalist: the online-lookahead planner (compute per decision stated; not deployable at serving time);
   the distilled network is reported as the deployable RL policy.

## Amendment 2026-10-11 (bar membership)

- **Bar for v4.1 search, selection and test:** the ten default routers (BAR_V4) plus every installed tuned control
  (`evolve/policies/tuned/*_best*.py`): tuned Dynamo (its tuned constants equal the defaults) and three tuned Mooncake
  versions (tuning seeds 0, 1, 2; 125 evaluations each; same objective, search3b, fitness against BAR_V4).
- **Simulator:** v4.1, `code_version` cbeb5161f046.
- **Results seen when this was written:**
  - validation-set scores of π_e and seed-0 tuned Mooncake against BAR_V4;
  - the v4.1 headroom pilot verdict;
  - no v4.1 search results and no test-set results.
- **RL claims:** claim (a) (vs the bar) and claim (b) (vs the evolved primary finalist) are separate. (b) does not
  require (a).

## Amendment 2026-10-12 (evolved v4.1 freeze; approved by Kamil)

- **Searches:** v4_ds_r1, v4_ds_r2 (DeepSeek), v4_claude_r1 (Claude via relay); search3b, tail4, bar v4t; 123
  candidate evaluations each; seeds l3_48a067a7df63, l3_1dafac434ec8, l3_6595477664aa.
- **"Top 5 programs of each run" (§4) read as the top 5 behaviourally distinct programs:** programs with identical
  search fitness (to full precision, i.e. the same decisions on search3b) count as one, represented by the earliest.
  v4_ds_r1's best fitness was shared by 12 programs.
- **Validation (heldout3, bar v4t = 14 routers, tail4; `finalists/v4/v4_val_v4t_{ds,cl}.json`):** all 15 programs
  and π_e score below 0. Frozen, by heldout3 fitness:

  | Finalist | Run | heldout3 fitness | geo E2E / TTFT / TPOT / TBT | sha256 |
  |---|---|---|---|---|
  | `v4_487363424749.py` (primary) | v4_ds_r2 | −0.0199 | 1.005 / 0.997 / 1.013 / 0.974 | 487363424749b062edb3787e8d0b103bdfe75e3581dc72a926fab2d857b11208 |
  | `v4_fef09c9e6c9b.py` | v4_claude_r1 | −0.0213 | 1.008 / 1.005 / 1.016 / 0.974 | fef09c9e6c9b596eedf0ba94b9b25522ef362f693b57bf6ba041074f0a8b38f7 |
  | `v4_3b4ba4e688e6.py` | v4_claude_r1 | −0.0259 | 1.012 / 1.000 / 1.017 / 0.976 | 3b4ba4e688e6188c7e70d327425c4b4130ecb8633b6eb039748f1c3067562c1a |

  π_e on the same bar: −0.0591 (1.035 / 1.083 / 1.002 / 0.930).
- **Simulator and bundle:** v4.1, `code_version` cbeb5161f046; prediction bundle 8e6e78e985b87b2a:82b720c5cc7e.
- **Results seen when this was written:** search3b and heldout3 scores of these programs, π_e and the bar; step B
  (failed gate) and the first B2 window. No test-set (test3b) results.
- The finalists stay untested until the joint freeze with RL (§4).

## Amendment 2026-10-12 (reporting; Kamil)

- **Every comparison also reports the centre and each router.** Alongside the P99 tails: mean and P50 of E2E and
  TTFT; against the bar (the per-scenario, per-metric envelope of the 14 routers) and against each router on its
  own, with cluster-bootstrap CIs and win counts (`evolve/analysis/policy_table.py`). The claim rules (§5) are
  unchanged: the bar stays the target, the single-router comparisons show where a policy sits.
- **C follows B2** whatever B2's gate says; C is judged by its improvement over its starting point, reported with
  the same tables.
- **Results seen when this was written:** the heldout3 policy table of the three evolved finalists and π_e
  (`finalists/v4/policy_table_heldout3.json`); B2 Tool w4000 and w6000.

## Amendment 2026-10-13 (step C; Kamil's choices C1–C12)

Design deck: https://claude.ai/artifact/6uQBgTFkhFqsg5JyPbNbe9 (v11). Kamil accepted every recommendation except
C9's fail branch, where he chose a larger C0 before deciding (below).

- **What C learns (C1, C3, C4).**
  - The router is the base router plus a learned correction. The correction predicts each replica's score gain over
    the base router's pick, from the router's view only.
  - The router keeps the base pick unless the best predicted gain exceeds a margin, so a model that learns nothing
    changes nothing.
  - The base router is the frozen primary finalist `v4_487363424749.py` (4873).
  - Ablations: a π_e base; a free-standing network. Control: copying the recorded winner.
- **Labels (C2; changes §4's "planners use sampled output lengths" for C's training labels only).**
  - Each training decision is scored on the window's recorded future: true arrivals and output lengths.
  - The router never sees output lengths. Planners keep sampled lengths.
  - A guessed-future label arm (8 sampled futures) runs once, as an ablation.
- **RL finalists (C5; replaces §4's planning primary and §8 item 5).**
  - The learned router is the primary RL finalist, and RL claims (a) and (b) of §5 are confirmatory for it. It takes
    the distilled network's slot.
  - Online planners are reported as exploratory analysis.
  - The model-free control (C11) keeps its slot and runs after C0.
  - If C0 (and C0b) fail, the learned router is dropped and the planners stay as analysis.
- **No arrival model (C6).**
- **Score of a choice (C7; adapts Q1).** Over the affected requests:
  - Score = Σ E2E + w · (hinge E2E + hinge TTFT + hinge TPOT + hinge TBT).
  - Each hinge is Σ (x/τ − 1)+ over requests; TBT's runs over token gaps, scaled by the window's requests per token.
  - τ per metric is the bar's P99 on that training window: the best of the 14 v4t routers run on it.
  - w per trace is step B's transferred weight: Tool 1,774 s, Conv 2,336 s.
  - Q1's R0 and R2 reward ablations are deferred until C0 passes; Q7's noisy-predictor arm goes to the planner study.
- **Training traffic (C8).**
  - New arrival draws p20 and p21 of Tool and Conv, 8R, load 0.75.
  - Request ranges of 2,000: Tool starts 0, 4000, …, 20000; Conv starts 0, 2000, …, 10000. That is 12 ranges × 2
    draws = 24 windows, with the usual warm-up cut.
  - Draws p14–p19 stay reserved (§1). heldout3 stays for selection; test3b stays unopened.
  - As in every set here, draws replay the same trace requests with other arrival times. The left-out fit (C9 item 1)
    is the check against memorising requests.
- **C0 and its pass mark (C9).** The model class and the margin grid are fixed in C0's reviewed code before any
  decision is scored.
  - **Data:** 4873 routes the 24 windows. Every 6th decision after the cut is scored (about 6,400), with 8 forked
    rollouts to the window end.
  - **Item 1:** leave one request range out (both of its draws). The margin is chosen by an inner split of the 11
    training ranges. Over the pooled left-out decisions, at least 30 overrides, and a mean score gain per override
    whose 95 % CI is above 0. The CI comes from a bootstrap over the 12 request ranges.
  - **Item 2:** at each left-out decision, take the model's best alternative to 4873's pick. The Spearman correlation
    of its predicted and recorded gain must have a 95 % CI above 0 (same bootstrap).
  - **Item 3:** one router, trained on all 24 windows, routes the 6 heldout3 Tool/Conv 8R 0.75 scenarios (p2, p3, p5).
    - The geometric mean of the four P99 ratios (new / 4873) must be below 1, with one-sided p < 0.05 from the B
      gate's pooled statistic against 20 runs of 4873 with arrivals jittered by up to 5 ms.
    - No 95 % cluster-bootstrap CI may lie wholly above 1 for P99 E2E, TTFT, TPOT or TBT, mean E2E or P50 E2E.
    - The other 21 heldout3 scenarios are reported.
  - **If items 1 or 2 fail: C0b (Kamil, 2026-10-13).**
    - Add Tool's other 5 request ranges (starts 2000, 6000, …, 18000; 17 ranges in all), with draws p20 and p21.
    - Score every 3rd decision in all windows, reusing C0's scored decisions: about 18,000 decisions, about 200
      further CPU-hours.
    - Rerun items 1–3 with the same rules. Across the two looks, the false-pass rate per item is at most 5 %
      one-sided, against 2.5 % for one look; this is reported.
    - If C0b also fails item 1 or 2, C stops and Kamil decides.
  - **If items 1 and 2 pass and item 3 fails:** report; more data or a shorter horizon, Kamil decides.
  - **Power:** with about 1,000 score units of spread and 12 ranges, a true mean gain of 200 per override passes
    item 1 at roughly even odds. B2-sized gains (+35) would likely be missed.
  - **Diagnostics, reported whatever the outcome:** left-out explained variance, override rate, agreement with 4873,
    and the Tool/Conv split.
- **Budget (C10).** Approved now: C12 and C0 (about 250–350 CPU-hours), and C0b if triggered (about 200 more). Full
  C (about 9,500 CPU-hours in all) waits for C0's result and Kamil's approval.
- **C12 (informs, no gate).**
  - The v4.1 headroom pilot is rerun with 4873 as base and rollout router: the same 4 windows (Tool and Conv, 8R,
    load 0.75, w0+2000 and w2000+2000, p1), the same code (decision shas unchanged, `code_version` cbeb5161f046) and
    the same pre-registered verdict (`--mode pilot`). Only `--policy` differs (repo 0c4533b).
  - p1 is one of the search draws 4873 was selected on, so the headroom it shows is conservative.
  - Alpine tree `vidur-rl-c12`, chains `c12_{tool,conv}_w{0,2000}` = jobs 34086837–68.
- **Results seen when this was written:** everything in the earlier amendments, plus B2's gate (fail) and audit
  (inconclusive) and the B2 check job. No C0 or C12 results; no test-set results.
