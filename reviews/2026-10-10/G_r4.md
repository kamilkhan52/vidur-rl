# Review G r4: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, commit 7bd2f2b)

**Verdict: changes requested (the controls and the score, not the mechanics).** Every G_r3 finding is fixed, the tests pass (15 passed, 120 s), and the new statistics code does what the docstring says.
The problem is what the controls measure. On Tool w0+2000 every tie-break of pi_e happens in the first 2.0 s, so the per-point tiebreak noise floor is exactly 0 at every open-loop point and the "no headroom" leg of the success rule cannot be met. The window-level tiebreak noise is probably degenerate for the same reason.
A 2-point smoke also shows that full's argmin under `w = 25 m` lowers the summed E2E by ~2,000 s and makes both tail hinges worse. The closed loop may therefore regress P99 because of the score weight, not because there is no headroom. No fix touches a `code_version`-hashed file (`evolve/rl_plan/*` only).

What I ran (never more than 2 simulator processes):
- `pytest -q evolve/rl_plan/test_headroom.py`: **15 passed** in 120 s.
- **One smoke run** (`scratchpad/g4_smoke.py`, Tool 8R s0.75 w0+2000 p1, 117 s wall, workers=2). It did two things:
  - a pi_e run with `ctx.rng.choice` instrumented;
  - `open_loop` at 2 points (`--every 10 --max-points 2`, horizons `T=1x+drain` + `full`, default tiebreak point noise).
  Output: `scratchpad/g4_smoke.txt`, `scratchpad/g4_out/ol.jsonl`.
- **`window_verdict` on constructed rows** (pure Python, no simulation): pooling of noise kinds and of stale rows.

Scratchpad = `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/`.

## G_r3 findings: status
| r3 | Status |
|---|---|
| 1 drain bias of `T=1x+drain` | **Fixed.** The headline is now `full`. The docstring calls `T+drain` a PROXY and quotes the 70-88 % hinge removal. The `full` child also scores each shorter horizon's set under continued arrivals (`cont`, `child_rollouts:537-540`), and the stats report `spearman_vs_cont` and `spearman_cont_vs_full`. The test checks that `cont` covers the same requests with nothing unfinished. Smoke, k=353: the T=1x drain ranks c7 at −89 and its `cont` at +325, so the drain-vs-continued difference is visible per candidate. |
| 2 open-loop agreement inflated by points that reach the window end | **Fixed.** `informative` = `n_h < n_full` per point and horizon, and `reached_end` is reported separately. The test uses a `T=1000+drain` horizon: all of its points land in `reached_end`, with top-1 = 1, regret = 0 and ref_gain = full's own. Paired horizon differences use only points informative for both. |
| 3 no CIs, no noise floor, no placebo | **Fixed in code.** Bootstrap CIs, a per-point noise rollout, placebo and noise modes, and a pre-specified rule in the docstring with `window_verdict`. What the noise measures does not hold up: findings 1 and 2. |
| 4 what a null bounds | **Fixed.** The wording is "one-step lookahead ON THAT SCORE". `full`'s own one-step gain distribution is the headline bound (`out["full"]`: median, p90, mean with CI, fraction above the floor). |
| 5 CODE_SHA orphaning logs | **Fixed.** Row and log keys use `decision_sha()` (rollout, forcing, scoring and loop source plus sim_fork.py). Summary rows store `post_sha()`. A closed row with a stale post_sha is recomputed by replaying its log. Residual: finding 6. |
| 6 stale a11 notes | **Fixed** (`a11_fork_spike.md:196`, update note; the earlier text is marked superseded). |
| nit replay / cost | **Fixed.** Replay also checks `row["t"] == req.arrived_at`, and the costs were revised. Smoke cost: 79 and 94 CPU-s per open-loop point with one proxy (8 full rollouts + T=1x drains + 1 noise rollout) at load ~4. That is consistent with the docstring's 121 CPU-s for five proxies. |

## Task 2: the new statistics, checked
- **Stratification:** correct (above). `informative` is the right criterion: a time or decision horizon whose stop leaves no further arrivals drains exactly like `full`.
- **Paired bootstrap:**
  - The pairing is correct. The resampling unit is the point, the per-point horizon differences are formed before resampling, and both horizons see the same state and the same `full` reference.
  - The i.i.d. assumption over points is not: finding 5.
- **Placebo rate:**
  - `q = rate·R/(R−1)` with a uniform replica gives P(executed ≠ pi_e) = q(R−1)/R = rate. This is correct, and pi_e's choice need not be known in advance.
  - At rate 0 the output is byte-identical to the plain run (tested).
  - The rate is the closed `full` row's `deviation_frac` over the same eligible set (arrived ≥ cut, every `plan_every`-th).
- **Noise mode:** `noise_run` reseeds `h.sim._scheduler._ctx.rng`, which is pi_e's only RNG use (`l3_48a067a7df63.py:282`, the last tie-break). That is mechanically right, but see findings 1 and 2.
- **`window_verdict`:** implements the docstring clause by clause.
  - positive = both E2E and TTFT below P5(P) and P2.5(N), and TPOT and TBT ≤ P97.5(N);
  - negative = E2E or TTFT > P97.5(N);
  - null = both within [P2.5(N), P97.5(N)].
  - positive and negative are mutually exclusive. `np.percentile` interpolates linearly: with 20 runs, P2.5 sits between the smallest and second-smallest value, and P5 just below the second smallest. That is acceptable as a pre-specified rule.
  - The row filter has a pooling bug: finding 4.

## Task 3: sampled-lengths block-hash refit
**Correct.**
- `fit_hashes` truncates to `total // block_size`, or extends with fresh negative ids that no trace contains. The tag is `r.id` for in-flight requests and `10^9 + j` for future ones, so the ids are unique, and they are identical across candidates at a decision (common random numbers kept).
- In-flight requests' KV-manager lists are extended when they are shorter.
- A KV-manager list that is longer, because the new output is shorter, is harmless. `cache_full_blocks` only uses entries up to `num_full_blocks`, and that never exceeds the new `total // block_size` (`base_kv_cache_manager.py:251-265`). The new length is always > the tokens already generated, so no allocated block is cut.
- Prompt blocks are never truncated: `(prefill + L) // bs ≥ prefill // bs`.
- The test checks the list lengths, and an end-to-end rollout under all four horizon kinds completes and differs from the true-lengths rollout.

Nit: finding 8.

---

### [major] The per-point noise floor is identically 0 at every open-loop point, so the "no headroom" leg cannot be met
`open_loop:874-879` (`pert = (point_noise, t_dec + max_t, max_n, …)`), `child_rollouts:498-504`, `openloop_stats:989-1006`, docstring "With few exact ties left after a point, the tiebreak floor can be 0 at that point"

**Failure scenario.**
- pi_e calls `ctx.rng` only on exact ties after four tie-break stages.
- On Tool w0+2000 **all 10 ties occur at t ≤ 2.0 s** (idle replicas at the start). The warm-up cut is at 50.9 s, and the perturbation is applied even later, after the longest in-window finite horizon and at least 65 decisions.
- So reseeding changes nothing: `delta` = 0 exactly. It is recorded as a measured floor (not NaN, because `perturbed` is set whenever the reseed point is reached) at every point.
- `frac_gain_above_noise` then becomes "fraction of points where full's gain is > 0". The docstring's "no headroom" condition, gain within the floor at ≥ 90 % of points, would need pi_e to be exactly full-greedy at 90 % of points.

**Evidence** (smoke run):
```
ties: 10 after cut: 0 times: [0.1, 0.5, 0.8, 1.1, 1.3, 1.5, 1.8, 1.8, 2.0, 2.0]
k 343: noise {'perturbed': 'ctx.rng seed 1000345 at t=145.756', 'delta': {'sum': 0.0, 'hinge': 0.0, 'hinge_ttft': 0.0}}
k 353: noise {... 'delta': {'sum': 0.0, 'hinge': 0.0, 'hinge_ttft': 0.0}}
full: one_step_gain mean 1972 (1958, 1986); noise_floor_median 0.0; frac_gain_above_noise 1.0
```
At both points, 6 of 7 alternative candidates beat pi_e under `full` by 800-2,000 score units.

The `token` kind is no better as a floor. By the docstring's own note, one-token changes do not cascade on Tool. Its |Δ| would be one request's TPOT-scale change against gains of ~2,000.

**Suggested fix** (not hashed):
- Use a floor that perturbs routing after the decision: π_e's own choice rolled out to `full` with **one forced uniformly random other replica at one later decision**. "Later" means after the longest in-window finite horizon, as now, or, for the headline, soon after the decision. Cost is the same one rollout per point.
- A random deviation also carries expected harm, so a cleaner alternative is to measure chaos in the candidate differences directly. Apply the same later random deviation to all candidates under two seeds (A, B), and report |Δ_c(A) − Δ_c(B)| against |Δ_c|. That costs 2× rollouts per point. Either way, a floor that cannot be 0 by construction.
- Then redefine the "no headroom" leg against it, or drop the leg and let the closed loop against the placebo decide. Pre-specify the new rule before any rerun.

Hashed: no.

### [major] The window-level noise distribution N is probably degenerate (tiebreak = relabelling idle replicas), which collapses the success rule
`noise_run:800-811` and its docstring ("an equally valid pi_e run … the spread comes from discrete routing changes, which `tiebreak` and the placebo measure"); `window_verdict`

**Failure scenario.**
- `noise_run --noise-kind tiebreak` reseeds at build time, so it changes only the ≤ 10 tie decisions at t ≤ 2.0 s (verified above). At those decisions the tied replicas are identical: every tie-break stage, including outstanding requests, compared equal.
- pi_e has no index-dependent rule: no hashing or modulo on the replica index, only loops (checked in `l3_48a067a7df63.py`). Sticky and prefix state follow the request. Replicas are homogeneous, and event order is `(time, type, id)`, where relabelling can reorder only simultaneous events on independent replicas.
- So a reseeded run is very likely a pure relabelling of replicas, and its per-request metrics are the same: **all 20 N rows ≈ 0**.

If so, the rule degenerates:
- **null** needs L_e2e = L_ttft = 0 exactly, which only a lookahead that never deviates achieves;
- **negative** = any worsening at all;
- the N2.5 condition of **positive** is vacuous;
- the TPOT/TBT guard becomes "not worse than pi_e at all", a coin-flip-strict condition across 4 windows.

**Not run** (smoke budget). The tie timing and the policy's index-freedom are verified. "N ≈ 0" is the inference.

**Check** (≈ 1 CPU-min): run `--mode noise --seeds 3` on Tool w0 and compare the P99s with pi_e's.

**Suggested fix** (not hashed):
- A noise kind that acts in the scored window, for example `deviate-once`: one uniformly random other replica forced at one uniformly random post-cut decision, over 20 seeds. That is the P99 lottery of a minimal discrete change. The placebo at rate r stays the "many random changes" control.
- Fix the kind in the docstring rule before the rerun, and correct the docstring claim.

Hashed: no.

### [major] Under `w = 25 m`, full's argmin trades tail for mean at both smoke points: the closed loop may regress P99 because of the score
docstring "Scores" (`w = 25 m` "a design choice"); the closed `full` headline

**Evidence** (smoke run). Tool w0+2000, τ_e = 94.8, τ_f = 39.6, w = 443.6; per-candidate `full` terms minus pi_e's; pi_e's parts at k=343 are mean 31,556, e2e_tail 1,820, ttft_tail 1,827:

| point | full argmin | Δ sum E2E | Δ e2e_tail | Δ ttft_tail | Δ score |
|---|---|---|---|---|---|
| k=343 (pe=7) | c0 | −2,763 | **+620** | **+185** | −1,958 |
| k=353 (pe=3) | c4 | −2,255 | **+131** | **+138** | −1,986 |

- Over all 14 non-pi_e candidates, the sum E2E is lower in 13 and the E2E hinge term is **higher in 13**.
- The full-horizon argmin makes **both** tail terms worse at both points.
- The score's mean term (~31,500) outweighs the tail terms (~3,650) about 9:1. A one-step improvement on this score is then mostly a mean improvement bought with tail.

Two points are an illustration, not an estimate. If the pattern holds, though, the closed loop's window P99 can come out **negative** by construction, and the pre-specified rule would then read "no headroom". The truth would be "this score's weight does not target P99". The pilot cannot separate the two.

**Suggested fix** (cheap; not hashed):
- The open-loop rows store every candidate's `sum`, `hinge` and `hinge_ttft` under `full`, so this is post-processing. In `openloop_stats`, report the per-term gain of full's argmin over pi_e for several weights (25 m, 100 m, 400 m, hinge-only), with the fraction of points where the argmin improves the E2E and TTFT hinge terms.
- Run the open loop **first** on all 4 windows, as planned (13 CPU-h), and choose and pre-register w from it before the closed loops. The rule: the smallest w whose argmin does not worsen the tail terms on average. Alternatively, pre-register two closed arms (25 m and the tail-dominant w), at +72 CPU-h.
- `window_verdict`'s `heads` filter would then also need to select by score and weight (today two arms give `missing`).
- Note in the docstring that a P99 regression under a mean-dominated score is not evidence against headroom.

Hashed: no.

### [minor] `window_verdict` pools noise kinds and rows from older `decision_sha`s
`window_verdict:1021-1023`

**Failure scenario.**
- N is "every row with mode noise", and P is "every placebo row at this rate". The docstring offers `--noise-kind token`, and running both kinds pools them.
- After any edit to a function in `decision_sha()` (finding 6), the controls rerun under new keys, but the old rows stay in `summary.json` and are pooled too.
  - For the closed row this fails loudly: two heads give `missing`.
  - For P and N it is silent.

**Evidence** (constructed rows; N: 20 tiebreak rows ~ N(0, 0.02), P: 20 rows ~ N(0.05, 0.03)):
- L = −0.03 is `inconclusive`. Adding 20 `token` rows at 0 gives N = 40 and turns it **`positive`**.
- 5 stale placebo rows at −0.2 turn L = −0.05 from `positive` to `inconclusive`.

**Suggested fix.**
- Filter P and N by `decision_sha` equal to the head's, and N by the pre-registered `noise_kind`.
- Or have `main` drop control rows whose `decision_sha` differs from the current one.
- Report the counts per kind.

Hashed: no.

### [minor] The bootstrap treats points as independent; nearby points are strongly correlated
`boot_ci`, used for every open-loop CI and for `one_step_gain`

- Points are every 10th decision, ≈ 1.3 s apart on Tool, while τ_e ≈ 95 s. The `full` rollouts of neighbouring points share almost all of their future.
- Smoke: points 1 and 11 decisions apart have full gains of 1,958 and 1,986, with the same "every alternative cuts the mean" pattern.
- The i.i.d. bootstrap CI over ~80-165 points will be too narrow. The CIs only serve proxy validation and the null bound, not the headline rule, hence minor.

**Suggested fix.** A moving-block bootstrap (blocks of ~10-20 points), plus the lag-1 autocorrelation of per-point `ref_gain` in the stats. Or state that the CIs condition on the trajectory and are optimistic.

Hashed: no.

### [minor] Control code is in `decision_sha`: fixing findings 1, 2 or 4 after the closed loops start orphans their logs
`decision_sha:1047-1052` includes `open_loop`, `placebo_run`, `noise_run`, `_finish_run`

- A closed `full` log (~18 CPU-h per window) is keyed by a sha that also covers the placebo and noise code.
- Findings 1 and 2 change exactly that code. If they land after the Alpine chains start, those chains' logs no longer match: the rerun starts from zero, and the old row lingers (finding 4).
- **Land all control changes before launching**, or use a per-mode sha: closed = rollout + forcing + scoring + `closed_loop` + sim_fork.

Hashed: no.

### [nit] The placebo's default rate filter is looser than the verdict's
`main:1232-1236`. The rate is taken from closed `full` rows filtered only by `plan_every`, while `window_verdict` also requires `max_decisions is None` and `future == "true"`. If only a `--max-decisions` smoke row exists, its rate is used silently. The 20 placebo runs then miss the head's rate, and the verdict says `missing` without saying why. Use the same filter.

### [nit] Sampled lengths: later session turns lose prefix hits when an earlier output is resampled longer
- A Conv next-turn prompt carries the original output block ids. A resampled-longer earlier turn caches fresh ids, and a shorter one caches fewer blocks.
- So sampled-lengths rollouts have systematically fewer session prefix hits than the true future, roughly equal across candidates.
- This falls under the documented "partial gap", but name it.
- `resample_outputs` refits only `_kv_cache_manager` lists, not a disk tier's (`_disk_kv_cache_manager`), which the pilot scenarios do not use.

### [nit] The cross-window claim is not computed
`window_verdict` labels one window. The 4-window claim and the open-loop ≥ 90 % leg are applied by hand. Add a small `pilot_verdict(summaries, openloop_rows)` so the pre-registered conclusion is code too.

## Task 4: is the ~85 CPU-h rerun plan sufficient?
**Budget: yes.**
- Closed `full` 4 × ~18 = 72 CPU-h, open loop 4 × 3.2 = 13, placebo and noise 4 × 40 × 25 s ≈ 1.1: ≈ 86 CPU-h.
- Smoke costs agree with the docstring.
- On Alpine with `--workers 8`, one window is ~1,658 decisions × ~5 s wall ≈ 2.5 h, so 8 chained 4 h jobs is ample, and two chains fit in 22 CPUs.

**Answering the question either way: not as specified.**
1. "No headroom" is unreachable: the per-point floor is 0 by construction (finding 1), and full's one-step gain was ~2,000 at both smoke points.
2. N is probably degenerate (finding 2), which makes "null" need L = 0 exactly and the TPOT/TBT guard near-impossible across 4 windows.
3. The score weight may make the closed loop P99-negative by construction (finding 3). Even a clean "negative" would then be about the score.

The likely overall outcome is therefore "inconclusive", whatever the truth. All three are cheap to fix before launch:
- (a) a post-decision routing perturbation for the per-point floor, and a post-cut `deviate-once` kind for N, checked first with 3 seeds (~1 CPU-min);
- (b) a re-pre-registered "no headroom" rule against that floor or against the placebo;
- (c) open loop first, with the per-term argmin gains across weights, then pre-register w (or two arms) for the closed loop;
- (d) `window_verdict` filters by `decision_sha` and `noise_kind`;
- (e) land all of this before the chains start (finding 6).

With these, the plan (open loop → closed `full` on Tool/Conv × w0/w2000 → 20 placebo + 20 noise per window) is sufficient for the pilot's hindsight question.
