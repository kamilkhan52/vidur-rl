# Review G r5: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, commit 8b564db)

**Verdict: changes requested (the pre-registered rules, not the mechanics).** Every G_r4 finding is addressed in code and the tests pass (**18 passed**, 151 s). The arrival-jitter noise is non-degenerate on Tool: 3 runs move the P99s by up to 8 %.
Three rule-level problems remain. (1) "No headroom" counts a *trade* (E2E much better, TTFT worse) as "negative". (2) The weight rule decides on per-point term changes that one smoke point shows are mostly chaos, and the rule can pick a weight at which the lookahead never deviates. (3) On Tool the TPOT noise band is ~0.1 % wide, so the TPOT guard is a knife edge.
Task 2: dropping the ">= 90 % within floor" leg is justified (smoke: jitter-only chaos in candidate differences is as large as the differences themselves). A cross-fitted gain is a better real-vs-chaos test than a best-of-R chaos baseline, and it costs about R−1 extra `full` rollouts per open-loop point. No finding touches a `code_version`-hashed file.

What I ran (at most 2 of my own simulator processes; disk 17 GB free throughout; no output trees kept):
- `pytest -q evolve/rl_plan/test_headroom.py`: 18 passed.
- **One smoke run** (`scratchpad/g5/smoke.py`, 117 s wall, Tool 8R s0.75 w0+2000 p1):
  - pi_e;
  - 3 `noise_run`s (d = 5 ms);
  - at one open-loop point (eligible decision #1000, k = 1342, t = 173.7 s, 816 affected requests), all 8 candidates' `full` rollouts under three conditions: unjittered (A), and with `jitter_future` seeds 11 (B) and 12 (B') applied to *every* candidate. The B rows test the cross-fit idea.
  - Output: `scratchpad/g5/smoke.txt`.
- Pure-Python checks on constructed rows: `window_verdict` / `pilot_verdict` with a trade, `weight_sweep` degeneracy, and the `select_candidate` tolerance at the `tails` weight.

Scratchpad = `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/`.

## G_r4 findings: status
| r4 | Status |
|---|---|
| 1 per-point floor identically 0 | **Fixed.** `jitter_future` (U(±5 ms) on every arrival after the decision, seeded per point) is applied to pi_e's own choice, `full` only. It cannot be 0 by construction. Note that a trivial part is always non-zero: E2E is measured from the shifted arrival, so the summed E2E moves by ~ms even without any routing change. The smoke floor at k = 1342 is **687 score units** (w = 25 m), the same order as the candidate differences (median \|D_c\| 270). By design it is applied to pi_e's choice only, not to every candidate. Finding 5 shows why that matters. |
| 2 window noise N degenerate | **Fixed.** `noise_run` jitters every arrival after the cut, with all per-request trace fields permuted together (checked: the generator has exactly `_arrived_at`, two token arrays, three lists; arrivals are pulled lazily, so the swap is safe). The ±5 ms is far below the trace's 3 s bucket (p1 spreads each bucket U[0, 3 s)·0.75; mean gap ~0.13 s on Tool). Smoke log P99 ratios vs pi_e: E2E +0.051 / +0.022 / +0.003, TTFT +0.012 / +0.080 / +0.073, TPOT −0.027 / −0.026 / −0.027, TBT −0.005 / −0.005 / −0.009. Non-degenerate, but see findings 3 and 5. |
| 3 weight makes the loop trade tail for mean | **Addressed** by `weight_sweep` and `--tail-weight auto`. Its soundness: finding 2. |
| 4 verdict pools stale/other-kind controls | **Fixed.** Controls are filtered by per-mode `decision_sha`, noise rows grouped by `kind:delta` and never pooled, and counts reported (`noise_by_kind`, `stale_controls`). The head is filtered by sha and weight. Tested with stale placebo rows, a second kind and two weight arms. |
| 5 i.i.d. bootstrap | **Fixed.** Circular moving-block bootstrap (`BLOCK = 20`); the indexing is correct. `lag1` is reported. The docstring states the CIs are conditional on the one trajectory. The test shows the block CI is > 2x the i.i.d. one on repeated data. |
| 6 control code in the closed sha | **Fixed.** `decision_sha(mode)`: closed = core + `closed_loop` + sim_fork. Jitter, placebo and noise code are outside it. Remaining caveat (docstring says so): `child_rollouts` is core, so any change to it, including the noise-floor hook, still orphans closed logs. Land before launch. |
| nit placebo rate filter | **Fixed.** `head_rows` is shared by the placebo default rate and the verdict, including the weight. |
| nit sampled-lengths session hits | **Fixed** (docstring "Known gaps"). |
| nit cross-window claim not code | **Fixed** (`pilot_verdict`, `--mode pilot`). Residual: finding 6. |

---

### [major] A tail *trade* is labelled "negative" and counts toward "no headroom"
`window_verdict:1143` (`neg = any(lab[m] > q(N, m, 97.5) for m in ("e2e", "ttft"))`), `pilot_verdict:1172`, docstring rule

**Failure scenario.** The lookahead cuts P99 E2E by 30 % and raises P99 TTFT by 5 % (above P97.5(N)).
- `positive` fails (TTFT not below the band), and `neg` is true, so the window is labelled `negative`.
- If all 4 windows look like this, `pilot_verdict` returns **"no headroom"**, although the lookahead found a large E2E gain.
- In r3 this case was guarded by the open-loop leg; with that leg dropped, the label alone decides the claim.

**Evidence** (constructed rows; N ~ N(0, 0.02)):
```
E2E -30%, TTFT +5%: negative N97.5 ttft 0.0303
pilot with 4 such windows: no headroom
```
**Fix.**
- `negative` = at least one of E2E/TTFT above P97.5(N), **and neither below P2.5(N)**.
- A mixed case is `inconclusive` (or a separate `trade` label that never counts toward either claim).
- Add the test case. Pre-register the change before launch.

Hashed: no.

### [major] The weight rule decides on per-point term changes that are mostly chaos, and it can select a weight at which the lookahead never deviates
`weight_sweep:989-1020`, docstring "Tail weight"

**(a) Degenerate choice (verified on constructed rows).**
- Suppose the alternatives buy a lower hinge with a higher sum, and the smallest grid weight is too small to flip any decision.
- Then the argmin is pi_e at every point, every Δ is 0, and "does not worsen either tail term" (≤ 0) is met trivially.
- The rule returns that weight with `rule_met: True`. The closed loop is then pi_e (L ≈ 0, `null`), which counts toward "no headroom".

```
choice {'label': '25m', ..., 'rule_met': True} deviates@choice 0.0 deviates@100m 0.0 d_hinge@100m 0.0
```
On the Tool smoke points w = 25 m deviates almost always, so this is a robustness gap, not the likely outcome. It is cheap to close.

**(b) The decision statistic is chaos-dominated (smoke, one point, k = 1342).**
- The term changes of the *same* argmin (A's choice, c6 at 25 m) evaluated in the three worlds:

  | world | d_sum | d_hinge (E2E) | d_hinge_ttft |
  |---|---|---|---|
  | A (unjittered, what the rule uses) | −243.6 | **+1.118** | −0.957 |
  | B (jitter seed 11) | +169.9 | **−0.815** | −1.580 |
  | B' (jitter seed 12) | −35.0 | **−1.073** | −1.666 |

- The E2E-hinge change flips sign under a ±5 ms perturbation.
- At 100 m the in-sample argmin is c0 under A and B', but c1 under B.
- The rule averages ~165 such points. With lag-1 correlated neighbours (r4: neighbours share their future), the effective n is closer to 165 / 20 ≈ 8 blocks. Whether the mean sits just below or just above 0 is then close to a coin flip, and "the smallest w that passes" selects precisely the w at which the in-sample mean is borderline.

**Fix** (not hashed; ~+R−1 `full` rollouts per open-loop point, shared with Task 2's cross-fit):
- Compute the sweep's term changes **cross-fitted**: choose the argmin on the unjittered rollouts, then evaluate its term changes against pi_e's choice in a jittered world B, with the same `jitter_future` seed for every candidate (CRN). The existing noise rollout already is pi_e's choice in B, so only R−1 rollouts are new.
- Require a **strict** improvement of at least one hinge (block-CI upper bound < 0 for E2E or TTFT, the other mean ≤ 0) and `deviates > 0`.
- If no w qualifies, use `tails` and propagate `rule_met: False` into `pilot_verdict`: a `negative` window under `rule_met: False` should not count toward "no headroom".
- The alternative from r4 still stands: pre-register two closed arms (25 m and `tails`), at +72 CPU-h.

Hashed: no.

### [major] The TPOT guard is a knife edge: N's TPOT band on Tool is ~0.1 % wide and offset from 0
`window_verdict:1142` (`all(lab[m] <= q(N, m, 97.5) for m in ("tpot", "tbt"))`)

**Evidence** (3 noise runs; preliminary):
- log P99 TPOT −0.0268 / −0.0256 / −0.0270 (spread 0.0014); TBT −0.0045 / −0.0045 / −0.0088.
- With 20 such runs, P97.5(N_tpot) ≈ −0.025. The lookahead must therefore have a P99 TPOT **≥ 2.5 % below pi_e's** to be `positive`.
- The guard is meant to catch "E2E/TTFT bought with token pace". In practice it becomes "TPOT within ±0.1 % of perturbed pi_e", which a lookahead that deviates at 30-80 % of decisions will rarely meet.
- A failed guard yields `inconclusive`, not a false "no headroom", but it can make the headroom claim unreachable.

**Fix.**
- Pre-register a tolerance: L_tpot ≤ max(P97.5(N), P97.5(P)) (random deviation at the same rate is the natural "no-gain pace" reference), or ≤ P97.5(N) + 0.02. Likewise for TBT.
- Have `window_verdict` report which clause failed.

Hashed: no.

### [major] "No headroom" has no stated effect size; the cross-window test is underpowered as specified
Docstring rule, `pilot_verdict`

- `null` means "within the window's P99 lottery band". From the smoke, that band is several percent wide (E2E 0.00-0.05, TTFT 0.01-0.08 from 3 runs).
- "All 4 null" therefore bounds the one-step-lookahead P99 gain at roughly that band's width, not at 0.
- The verdict should say so. Otherwise a real 3 % gain reads as "no headroom".

**Fix** (post-processing only):
- **Report the band:** report each window's minimum detectable effect, P2.5(N) and P5(P) per metric. State "no headroom" as "no one-step gain beyond X % in P99 E2E/TTFT".
- **Add a pooled statistic (pre-register it):** z_w = (L_w − median N_w) / IQR-scale(N_w), summed over the 4 windows. Compare it with the same sum built from one held-out N draw per window (leave-one-out resampling of the N rows). This halves the detectable effect relative to per-window bands and uses runs that already exist.
- Use the pooled statistic as the "no headroom" criterion together with the per-window labels (e.g. pooled z within the central 90 % **and** no window `positive`). Keep "all 4 positive" for the headroom claim, or use pooled z below its 2.5 % quantile plus no window negative.

Hashed: no.

### [minor] Jittered worlds systematically disfavour pi_e's own trajectory in the smoke: check that 0 lies inside N
Observation; mechanism **not established**.

- **Noise runs:** all 3 are worse than pi_e on P99 E2E and TTFT, and all 3 better on TPOT by an almost constant 2.7 %.
- **The open-loop point:**
  - Under both jitter seeds pi_e's choice (c4) is **second-worst of the 8 candidates** (only c5 is worse): D_c(B) = −233 … −1,359 for every alternative except c5 (+257), and D_c(B') similarly (alternatives' mean shift vs A: −1,166 / −1,440).
  - Unjittered, the alternatives straddle pi_e (−172 … +1,277).
  - Under pure chaos, P(pi_e last or second-to-last in both seeds) is about 6 %.
- **Ruled out:**
  - pi_e reads no time (only `ctx.rng` at the last tie-break).
  - The generator has no other per-request arrays.
  - The w0 p1 arrivals are not the search scenario's realisation (first 2,000 arrivals of the full-trace Tool s0.75 p1 differ by up to 1.13 s).
- **Why it matters:**
  - The rule compares L with N's quantiles, which is correct whenever L behaves like "another draw". But if the unjittered reference is atypical among its own perturbations, then pi_e itself (L = 0) can fall outside [P2.5(N), P97.5(N)].
  - Then a lookahead that matches pi_e exactly is labelled negative or positive.
- **Fix:**
  - Have `window_verdict` report whether 0 ∈ [P2.5(N), P97.5(N)] for E2E and TTFT. Pre-register that a window where it is not is `inconclusive (noise model)`.
  - A 25 s check: `noise_run` with d = 1e-9 vs pi_e. Not run here, to stay within the one-smoke budget.

Hashed: no.

### [minor] `pilot_verdict` trusts stored per-window labels
`pilot_verdict:1152-1176`

- It reads `rows["verdict"]["label"]` as last written. It does not recompute `window_verdict` with the current `decision_sha`s and `post_sha`.
- It does not check that the 4 directories are the pre-registered windows (meta `scenario` tags), or that each window's open-loop `rule_met` holds.
- A window whose last write predates a code change, or a 5th smoke directory passed by mistake, is silently accepted.
- **Fix:** recompute the verdicts from rows with the current shas, check the 4 scenario tags, and surface `rule_met`.

Hashed: no.

### [minor] Tests of "the jitter perturbs the run" pass trivially
`test_open_loop_selection_and_resume` (`any(abs(v) > 0 ... noise delta)`), `test_placebo_and_noise_runs` (`not filecmp.cmp(...)`)

- Both pass even if the jitter changed no routing or batching decision.
- The CSV differs because the arrival column changed, and the summed E2E moves by ~ms from the arrival shift alone.
- **Fix:** assert something the arrival shift alone cannot produce. For example: some request's replica, or its `prefill_completed_at − arrived_at`, differs by more than 2·d. Or a P99 changes by more than 1e-3 on the Conv test window.

Hashed: no.

### [nit] At the `tails` weight the summed-E2E tie-break is mostly disabled by the relative tolerance
`select_candidate:329`. The tolerance is `1e-9 · |s_pe|`, and with w = 10^9 m that is ≈ m·(hinge + hinge_ttft) seconds of summed E2E. Example: m = 20 s and hinges 70 give a tolerance of 1,400 s, so equal-hinge candidates with sum gains up to that size keep pi_e. The docstring says "the summed E2E a tie-break". Either use an absolute tolerance on the hinge part at `tails`, or say so.

---

## Task 2: the changed "no headroom" rule, and a better real-vs-chaos test
**Dropping the leg is justified.** Smoke at k = 1342, candidate differences D_c = s_c − s_pe over the 7 alternatives:

| w | sd of D (A) | chaos: sd of D(B) − D(B') | Spearman A–B | Spearman B–B' |
|---|---|---|---|---|
| 25 m | 484 | 377 | 0.32 | 0.50 |
| 100 m | 807 | 1,185 | 0.46 | 0.50 |

- At this point jitter-only chaos in the candidate differences is as large as the differences.
- The in-sample argmin changes between worlds (100 m: c0 / c1 / c0).
- So `full`'s best-of-R gain over pi_e is partly chaos, and "gain within one floor draw" could never be met. The new rule (all 4 null-or-negative) is sound **once the trade fix (finding 1) and an effect-size statement (finding 4) are in**.

**Is a chaos baseline better?** Yes, as a pre-registered *secondary* that tells real from chaotic one-step gain. Of the two variants, I recommend the cross-fit:
1. **Best-of-(R−1) chaos baseline** (candidates = pi_e's choice under R−1 jitter seeds; compare `full`'s best-of-R gain with the best-of-(R−1) gain among pure-chaos copies).
   - Simple and intuitive. It assumes a 5 ms jitter produces chaos of the same size as a re-routing.
   - The smoke supports this roughly at this point (jitter floor 687 vs median |D| 270 at 25 m), but it is not guaranteed elsewhere.
   - Cost: R−2 extra rollouts per point (one exists).
2. **Cross-fitted gain** (preferred).
   - Choose c* on the unjittered rollouts (A), and score s_pe(B) − s_c*(B) in a world B jittered identically for all candidates (same `jitter_future` seed, CRN).
   - The chaotic part of A's choice does not carry over to B, so the mean over points is an unbiased estimate of the expected gain of the hindsight argmin under the *same* nuisance distribution that defines N. That makes it consistent with the closed-loop rule.
   - It needs no magnitude assumption.
   - Cost: R−1 extra `full` rollouts per point (pi_e's B rollout is the existing floor rollout), about +1.5-2 CPU-h per window.
   - It also gives the weight rule a cross-fitted statistic (finding 2).
   - Caveat from the smoke: under jitter pi_e's choice shifted relative to 6 of 7 alternatives (finding 5). At this point the cross-fit gain (892 / 1,250) *exceeds* the in-sample gain (172). Average it over all points, and report it beside the in-sample gain and the jitter-floor distribution.

**Suggested pre-registration:**
- "No headroom" = per-window labels all null-or-negative (with the trade fix) **and** the cross-fitted one-step gain's block CI includes 0 (or its tail-term components are ≥ 0). Report the band as in finding 4.
- A closed-loop null with a clearly positive cross-fitted gain reads "per-decision gain exists but does not reach window P99", which is inconclusive, not "no headroom".

## Task 3: the weight rule
- Sound in intent: a necessary-condition filter ("even in hindsight the argmin must not worsen the tails").
- Two flaws as implemented (finding 2): it can choose a weight at which nothing deviates, and its per-point statistic is chaos-dominated, so "the smallest w that passes" picks a borderline w.
- **Degenerate weights:** yes, verified on constructed rows (zero deviations, `rule_met: True`). Also `rule_met: False` is not used by `pilot_verdict`.
- **Fix:** cross-fitted statistic + strict improvement + `deviates > 0` + propagate `rule_met`. Or two pre-registered arms.

## Task 4: is the run plan sufficient?
**Budget: yes.**
- Open loop 4 × 3.2 CPU-h, plus ≈ 4 × 1.8 with the cross-fit rollouts.
- Closed `full` 4 × 18.
- 20 placebo + 20 noise runs per window ≈ 1.1 CPU-h.
- Total ≈ 95 CPU-h, fine for two Alpine chains.

**Answering either way: not yet, but close.** Before launch:
- (a) fix the "negative" trade (finding 1);
- (b) cross-fitted weight statistic with a non-degenerate rule, `rule_met` propagated (finding 2);
- (c) a TPOT/TBT guard tolerance (finding 3);
- (d) effect-size reporting and a pooled cross-window statistic (finding 4);
- (e) the 0 ∈ N check (finding 5);
- (f) the cross-fitted one-step gain as the pre-registered secondary (Task 2).

All are in non-hashed `evolve/rl_plan/` code. (b) and (f) change `open_loop`, which is outside the closed `decision_sha`. With them, the plan (open loop → weight → closed `full` → placebo/noise → pilot) answers the pilot question both ways:
- headroom: all 4 windows positive;
- bounded no-headroom: all null/negative, cross-fit null, band stated;
- inconclusive otherwise.
