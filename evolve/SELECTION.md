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
