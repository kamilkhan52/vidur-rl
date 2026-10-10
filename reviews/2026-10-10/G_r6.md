# Review G r6: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, commit eb58f2b)

**Verdict: changes requested (two post-processing rules; the run code is ready).** All six G_r5 items (a)-(f) are in the code, and the labels match the docstring at every boundary I constructed. Tests: **20 passed** (158 s).
1. As pre-registered, "no headroom" is close to unreachable. The weight rule requires a strict cross-fitted per-decision tail gain. The no-headroom gate requires that no cross-fitted per-decision gain exists. A world with truly zero headroom therefore returns "inconclusive".
2. The effect sizes printed with the claim are design MDEs measured from the wrong origin, not bounds the data support. They understate the gain that cannot be excluded by 2-4x.

Both fixes are in `pilot_verdict` / `window_verdict` post-processing. They do not orphan any log (`post_sha` recomputes the rows). No finding touches a `code_version`-hashed file.

What I ran (at most 2 of my own simulator processes; disk 17 GB free before and after; no output trees kept):
- `pytest -q evolve/rl_plan/test_headroom.py`: 20 passed.
- Pure-Python checks, no simulation:
  - `scratchpad/g6/labels.py`: label boundaries.
  - `scratchpad/g6/regimes.py`: weight rule vs no-headroom gate on synthetic open-loop rows.
  - `scratchpad/g6/bound.py`: pooled bound.
- **One smoke run** (`scratchpad/g6/noise_smoke.py`, 2 processes, ~2 min wall): Tool 8R s0.75 w0+2000 p1, pi_e, `noise_run` at d = 1e-9 (seed 0), and d = 5 ms (seeds 0-8).

Scratchpad = `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/`.

## G_r5 pre-launch items: status
| r5 | Status |
|---|---|
| (a) trade counted as negative | **Fixed.** New `trade` label: one tail better and the other worse, or both better with the pace guard failed. It never counts toward either claim. Residual edge: minor finding 4. |
| (b) weight rule | **Fixed as specified.** It uses cross-fitted term changes, `MIN_DEV_RATE` = 5 %, strict improvement (block-CI hi < 0 for one hinge, mean <= 0 for the other), and `rule_met` reaches `pilot_verdict`. But combined with (f) it blocks "no headroom": finding 1. |
| (c) pace guard | **Fixed.** The limit is L <= max(P97.5(N), P97.5(P)). The knife edge itself was a 3-run artefact: 9 runs put TPOT between −0.027 and +0.030 (table below). |
| (d) effect size + pooled statistic | **Implemented, but the bound is miscomputed**: finding 2. `pooled_stat` matches its docstring. The null over leave-one-out draws is enumerated exactly at 20^4. |
| (e) 0 ∈ N check | **Fixed** (`inconclusive_noise_model`). Smoke: d = 1e-9 reproduces pi_e exactly (all four log ratios 0.0), so the jitter mechanism itself is neutral. The band is offset (minor finding 3). |
| (f) cross-fit | **Fixed.** World B uses one `jitter_future` seed per point for every candidate. pi_e's B rollout is the floor rollout (test asserts `xfit[pe].sum == full[pe].sum + delta.sum`). B forces only the others (probe fork), and consistency is checked on `policy_choice`, `n` and `perturbed`. |
| minor `pilot_verdict` trusts stored labels | **Fixed.** It recomputes labels with current shas and post_sha, requires exactly the 4 pre-registered tags, reads the weight from the open-loop row, and has tests for stale labels, a stale post_sha and a wrong window set. Residual: minor finding 5. |
| nit tails tie-break | **Fixed** (tolerance on the unweighted summed E2E; `test_tails_tie_break`). |

**Docstring rules vs code, side by side.** I checked the labels in order, the weight rule, the pace guard, the noise-model check, the pooled Z and the pilot conjuncts. The code implements exactly what the docstring states. Constructed rows (N = linspace(±0.02, 20), P = linspace(−0.06, 0.10, 20); N band [−0.019, 0.019], P5(P) = −0.052):
```
both = P2.5(N) exactly                     -> null
both = P97.5(N) exactly                    -> null
e2e below N, above P5(P); ttft within      -> inconclusive
e2e below N, above P5(P); ttft worse       -> negative      (finding 4)
e2e beyond both; ttft worse                -> trade
e2e beyond both; ttft within               -> inconclusive
both beyond, tpot at limit                 -> positive
both beyond, tpot above limit              -> trade
e2e worse, ttft within                     -> negative
```

**Smoke run: noise runs on Tool w0+2000** (log P99 ratio vs unjittered pi_e):

| seed | E2E | TTFT | TPOT | TBT |
|---|---|---|---|---|
| d = 1e-9 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0 | +0.051 | +0.012 | −0.027 | −0.005 |
| 1 | +0.022 | +0.080 | −0.026 | −0.005 |
| 2 | +0.003 | +0.073 | −0.027 | −0.009 |
| 3 | +0.037 | +0.013 | +0.001 | +0.009 |
| 4 | +0.009 | +0.073 | +0.030 | +0.001 |
| 5 | +0.024 | +0.088 | −0.015 | −0.052 |
| 6 | −0.014 | −0.031 | −0.011 | −0.020 |
| 7 | +0.043 | +0.022 | −0.008 | −0.030 |
| 8 | −0.006 | +0.012 | +0.009 | −0.009 |

Summary:
- E2E: median +0.022, sd 0.023, 7/9 > 0.
- TTFT: median +0.022, sd 0.041, 8/9 > 0.

---

### [major] The weight rule and the no-headroom gate exclude each other, so a world with no headroom returns "inconclusive"
`weight_sweep:1075-1084` (rule ii), `pilot_verdict:1346-1353`, docstring "Tail weight" and "no headroom"

**Mechanism.** Two of the "no headroom" conjuncts point in opposite directions:
- "Every window's weight rule met" needs a weight at which the hindsight argmin improves a tail hinge **strictly**, cross-fitted in world B. That statement says a per-decision gain exists.
- "Every window's cross-fitted one-step gain has block-CI lo <= 0" needs that no per-decision gain exists.

The outcomes then split as follows:
- **No per-decision gain (true null).** No weight passes (ii), so `rule_met: False`, which counts as "inconclusive".
- **Per-decision tail gains exist.** The rule is met, but at the chosen weight the hinge terms dominate the score, so `gain_xfit.lo > 0`, which also counts as "inconclusive".
- **The only path to "no headroom"** is a small weight (25 m / 100 m) at which chaos in the summed-E2E term happens to widen the total-gain CI over 0. That is a coin flip on the nuisance term, not evidence.

**Evidence** (synthetic open-loop rows, `regimes.py`):
- 165 points, R = 8.
- Chaos per rollout: sd 300 (sum), 1.5 (each hinge). For scale, the G_r5 smoke saw the E2E-hinge change of one c* move by ~2 under jitter.
- 10 seeds per regime. "Eligible" = `rule_met and gain_xfit.lo <= 0`, i.e. the window can count toward "no headroom".
```
regime 0 (no candidate truly better):            no-headroom-eligible 1/10  (9/10: tails, rule_met False)
regime 1, one candidate hinge -1, sum +30:       1/10
regime 1, hinge -3, sum +30:                     4/10  (eligible ones all chose 25m/100m; 100m/400m/tails picks have lo > 0)
regime 1, hinge -6, sum +300:                    2/10
```
- Four windows must all be eligible, so P("no headroom") is about 0.1^4 in the true-null world.
- On the real data: the G_r5 smoke point had a cross-fit gain of 892 / 1,250 > 0 at 25 m. If that is typical, the likely outcome of a closed-loop null is "inconclusive" whatever the truth.
- The run plan therefore cannot answer the negative side of the pilot question.

This would change a conclusion: true "no one-step headroom" is reported as "inconclusive" after ~93 CPU-h.

**Fix** (post-processing only; commit before any open-loop output is read, to keep the pre-registration clean):
- Drop both conjuncts from "no headroom": "no headroom" = all 4 `null`/`negative` **and** the pooled Z >= its null q5, with the bound from finding 2.
- Report `rule_met` and `gain_xfit` as the **mechanism**, not as gates:
  - (i) `rule_met: False`: "even in hindsight, the one-step argmin does not improve a tail term per decision (cross-fitted)". This is the strongest form of no headroom. At `tails` the closed loop has no mean term, so G_r5's worry (a negative caused by a mean-for-tail trade) does not apply.
  - (ii) `rule_met: True`: "per-decision tail gains exist (gain_xfit CI) but do not reach window P99 beyond the bound".
- The alternative is to keep the gates and drop "strict" from rule (ii). That leaves the rule decided by chaos (G_r5 finding 2b), so I do not recommend it.
- Add a test in which a window set with `rule_met: False` / `gain_xfit.lo > 0` and null labels can still yield "no headroom".

Hashed: no. The weight rule's *choice* is unchanged, so the closed-loop runs are unaffected and can launch now.

### [major] The effect sizes stated with "no headroom" are design MDEs from the wrong origin, not bounds; they understate what the data cannot exclude by 2-4x
`pooled_stat:1291-1293` (`mde = (q5 - q50) / inv`), `window_verdict:1248` (`mde_pct = expm1(P2.5(N))`), docstring "EFFECT SIZE"

The docstring says: "No headroom means: no one-step-lookahead P99 gain larger than the reported bound". Neither bound supports that sentence.

**(a) Pooled.** `effect_size_ruled_out_pct` does not depend on the observed Z. It is the shift at which the *median* null reaches q5, i.e. 50 % power.
- Under a common log shift δ, Z = Z0 + δ·Σ1/s_w. Gains are excluded at one-sided 5 % only for δ < (Z − q95)/Σ1/s_w.
- Evidence (`bound.py`; 4 windows, N ~ N(0, 0.02), every window's L equal):
```
common L=+0.000: Z=-0.23 q5=-4.47 pass=True reported ruled-out -2.10%  data-based 95% bound -1.96%
common L=-0.010: Z=-2.30 q5=-4.47 pass=True reported ruled-out -2.10%  data-based 95% bound -2.93%
common L=-0.020: Z=-4.37 q5=-4.47 pass=True reported ruled-out -2.10%  data-based 95% bound -3.90%
```
- With every window's lookahead 2 % better, the pilot passes the pooled gate (the windows are `null`) and prints "no gain beyond 2.1 %". The data allow 3.9 %.

**(b) Per window.** `mde_pct` is measured from 0 (unjittered pi_e), but the noise band is not centred on 0.
- On Tool w0 the median of N is +2.2 % (table above).
- The reported E2E `mde_pct` would be ≈ −1.2 % (P2.5 ≈ −0.012).
- What the window can actually call is a shift of median − P2.5 ≈ 3.4 %.
- A lookahead with L = 0 cannot exclude a gain of up to L − P97.5(N) ≈ −5 %.

**Fix** (post-processing):
- Per window: report the one-sided bound L − P97.5(N) (log, and as %) as "ruled out".
- Pooled: report (Z − q95)/Σ1/s_w.
- Keep the design MDEs only under a "detectable" label.
- Reword the docstring's EFFECT SIZE paragraph to match.

Hashed: no. Can follow the runs (rows recompute), but the wording is part of the pre-registration.

### [minor] The noise band on Tool is offset: jittered pi_e is worse than unjittered in 7/9 (E2E) and 8/9 (TTFT) runs
Observation from the smoke; mechanism not established.

What the smoke shows:
- d = 1e-9 gives exactly 0, so the permutation and stable sort are neutral.
- At 5 ms, the median log P99 is +0.022 for both E2E and TTFT. A sign test on TTFT gives p ≈ 0.02.

Consequences:
1. **The noise-model check (pre-registered) will sometimes trip.** A normal fit to these 9 runs puts P(0 outside a 20-run band) at ~5 % per window, and ~19 % that one of 4 such windows is `inconclusive_noise_model`. This costs power, not correctness.
2. **The offset has two possible origins, and they differ in what they mean:**
   - Unjittered pi_e is "lucky" on this realisation. Then L vs N is fair.
   - The jitter itself hurts trajectories. Then the unjittered closed loop gets a free ~2 % edge against N (raising the null rate of "better" per metric). It would also bias the cross-fitted gain upward, since G_r5's point had pi_e's choice second-worst in B.

   The 4-window × 2-metric conjunction keeps a false "headroom" unlikely in either case.

**Cheap pre-launch discriminator** (~3 min): 9 `noise_run`-style jittered runs of `lor` and `mooncake` on the same window. If both are also worse than their unjittered runs, the jitter is harmful. Report it with the pilot either way.

Hashed: no.

### [minor] "Below the noise band but not beyond placebo" plus the other tail worse is labelled `negative` and counts toward "no headroom"
`window_verdict:1233-1238`

- The code matches the docstring ("negative: at least one worse and **neither better**", where "better" includes L < P5(P)).
- G_r5's fix was "neither below P2.5(N)". The two differ when P5(P) < P2.5(N), which is plausible: random deviations at a 30-80 % rate spread the P99s much wider than ±5 ms jitter.
- Constructed row above: E2E −3 % (beyond noise, within placebo reach), TTFT +3 % (worse) gives `negative`.
- The pooled Z partly guards this case, but only if the E2E gain recurs across windows.

**Fix:** `negative` = at least one tail worse and neither tail below P2.5(N). The mixed case becomes `inconclusive`.

Hashed: no.

### [minor] `pilot_verdict` does not check that the 4 windows share the simulator version and the prediction bundle
`pilot_verdict:1311-1313`

- It reads only `meta["scenario"]`. The per-mode `decision_sha` covers `headroom.py` and `sim_fork.py`, not `vidur/**`.
- Summaries written under different `code_version`s or bundles (e.g. one window rerun on the Mac, or after a simulator change) would be pooled silently.
- **Fix:** require identical `meta` except `scenario` across the 4 windows, and `code_version` equal to the current `ev.code_version()`. Otherwise return `invalid_windows`.

Hashed: no.

### [nit] Smaller issues
- `pilot_verdict:1322-1325`: when an open-loop row lacks `xfit` or a weight choice, `chosen_weight` raises, and the window is reported as "stale open-loop statistics". The real cause is missing rollouts.
- Docstring Alpine chain: it runs only `--mode closed` (default `--tail-weight auto`). Without the open-loop row, every job exits at once and `afterany` lets the chain run out.
  - Put the four modes in one `CMD` in order (`openloop && closed && placebo && noise`). Each resumes or reuses.
  - Note that Conv's per-rollout cost is unmeasured (the cost table is Tool only). Check it from the first open-loop points.

---

## Task 4: is the run plan ready to launch?
- **Run code: yes.**
  - Rollouts, cross-fit, forcing, controls and resume are tested.
  - The noise is non-degenerate and neutral at d → 0.
  - The budget is ~96 CPU-h: open loop 4 × ~5, closed 4 × 18, controls 4 × ~1.1. That fits two 22-CPU Alpine chains.
  - Nothing in findings 1-2 changes what a run computes or which weight it uses, so the logs survive the fixes.
- **Decision rule: not yet.**
  - Finding 1 must change so the plan can return "no headroom".
  - Finding 2 must change so the stated bound is true.
  - Commit both, and ideally finding 4 and the version check, **before the first open-loop output is read**. They are ~30 lines of post-processing plus tests.
  - The `lor` / `mooncake` jitter check (finding 3) is optional but cheap. It tells which way the noise-band offset should be read.
- With those changes the plan answers both ways:
  - headroom: all 4 windows `positive`;
  - bounded no-headroom: all null/negative, pooled Z >= q5, a data-based bound, and mechanism (i) or (ii) stated;
  - inconclusive otherwise.
