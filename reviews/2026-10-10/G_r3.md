# Review G r3: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, commit ccc923c)

**Verdict: changes requested (methodology, not mechanics).** Every G_r2 finding is fixed in code, the tests pass (11 passed, 140 s), and the r2 sign flip is gone under `T=1x+drain`. Resume, header and replay checks work.
Two things stand in the way of the rerun. First, a probe on a non-trivial Tool point shows that the drain in `T=1x+drain` cuts the affected set's E2E hinge 3.4-8×, and how much it flatters a candidate differs by more than the decision signal. Second, the open-loop selection statistics are inflated by points where a horizon reaches the window end (identical to the reference by construction), have no CI, and have no noise floor. The closed-loop plan also has no placebo control.
No fix touches a `code_version`-hashed file: `evolve/rl_plan/*` and `evolve/sim_fork.py` are outside `vidur/`.

What I ran (never more than 2 active simulator processes):
- `pytest -q evolve/rl_plan/test_headroom.py`: **11 passed** in 140 s.
- **r2 blind-spot probe under the new default** (`scratchpad/g3_conv.py`; Conv 4R s1.5 w6000+200, K = 80/120/160, horizons `T=1x` censored, `T=1x+drain`, `full`; workers=1).
- **Drain-bias probe** (`scratchpad/drain_bias.py`; Tool 8R s0.75 w0+2000, K=400, t=57.9 s, cutoff 152.8 s < last arrival 254 s, so a non-trivial point; serial children, 52 s wall). For every candidate I scored three things:
  - (a) the `T=1x+drain` score of its affected set S_T (855 requests);
  - (b) the same S_T scored in the full continuation, where arrivals keep coming;
  - (c) the `full` reference over all 1,679 remaining requests.

  τ_e = 94.84, τ_f = 39.55, w = 25 × 17.74 (π_e on this window, v4, from the a11 notes).

Paths are relative to `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/`.

## G_r2 findings: status
| r2 | Status |
|---|---|
| 1 T=1x blind spot / censoring sign flip | **Fixed as stated.** The default is `T=1x+drain`, and the pure cutoff is kept with a warning. Audit fields are logged per candidate (`routed_e2e`, `routed_censored`, `post_hinge`, `n_unfinished`). The docstring is corrected. Repro on Conv: at K=120, candidate 2 scores **4307.5 under drain against π_e's 4078.1**, the uncensored values of r2, so the flip is gone. `post_hinge` is 0.171/0.057/0.132/0.052 at K=80, equal to the full continuation. On this window, though, the drain score equals `full` exactly at every candidate and point (equal `n`: the cutoff is past the last arrival), so **the test scenario cannot see drain bias**. `test_censoring_probe` proves only "nothing censored, drained ≥ censored", which holds by monotonicity. The new problem is finding 1. |
| 2 summary reuse ignores code | **Fixed.** `CODE_SHA` (headroom.py and sim_fork.py as loaded) plus the scorer settings are in every row key and log header. Side effect: finding 5. |
| 3 sweep cannot discriminate | **Replaced** by `--mode openloop` against a `full` reference. The design is right, but the statistics need work: findings 2 and 3. |
| 4 tail weight wording, TPOT/TBT | **Fixed.** The weight is described as a design choice, and `vs_pi_e` (4 log P99 ratios, mean, tail4) is on every closed-loop row. |
| 5 resume, failed rows | **Fixed.** Append-and-replay, partial last line dropped, header check, π_e-choice check on replay, "not reached" check at the end, `ok: False` rows not stored. The interrupted-then-resumed test equals the uninterrupted run byte for byte, and passes. |
| 6 sampled-lengths naming | **Fixed** (`--future sampled-lengths`, documented as a partial gap). |

Mechanics checked, no finding:
- **Nested horizons.** One child per candidate, with grandchildren draining shorter horizons. `test_nested_horizons_equal_single_rollouts` shows each equals a single-horizon rollout. `RUSAGE_CHILDREN` includes grandchildren, because each child reaps its own.
- **Affected set** (task 2). Arrivals are generated lazily, at their trace time ≤ cutoff (`TraceRequestGenerator`, `_next_time > cutoff` stop), so S = in flight at the decision + arrivals ≤ cutoff is the **same set for every candidate**, and `evaluate_candidates` asserts equal `n`. Sets differ *between* horizons, so only within-horizon rankings are compared, and that is what `openloop_stats` does.
- **A crash between the parent's `apply_decision` and the log write.** The decision is re-planned on resume, deterministically (true future; sampled mode seeds per decision index), so the result is the same.

---

### [major] `T=1x+drain` under-scores the affected set's E2E tail 3.4-8×, and the drain's flattery differs across candidates by more than the decision signal
`headroom.py:17-20` (docstring "the routed request's and late arrivals' tails are scored"), `:421-431` (`finish`, `drain`); default `--horizon T=1x+drain`

**Failure scenario.** At the cutoff, arrivals stop and S_T drains contention-free. Requests still queued or decoding at the cutoff, about the last τ_e of arrivals, finish much faster than they would under load. A candidate that leaves more work in flight at the cutoff is flattered more. So stopping arrivals biases toward candidates that defer work past the cutoff, which is the hypothesis of task 2.

**Evidence** (Tool w0+2000, K=400, π_e = 5; differences are against π_e's score; lower is better):

| c | (a) drain on S_T | (b) same S_T, arrivals continue | (c) full, all requests | flattery (b)−(a) | E2E hinge (a) → (b) |
|---|---|---|---|---|---|
| 0 | +455.7 | +99.9 | +755.3 | 2773 | 1.12 → 3.82 |
| 1 | +227.4 | +283.2 | −74.7 | 3185 | 0.84 → 4.62 |
| 2 | +37.9 | +94.1 | +578.1 | 3185 | 0.73 → 4.15 |
| **3** | **−350.5** | **+342.1** | −155.9 | **3821** | 0.67 → 5.49 |
| 4 | −182.5 | +110.1 | +551.6 | 3421 | 0.78 → 4.78 |
| 5 (π_e) | 0 | 0 | 0 | 3129 | 0.76 → 3.84 |
| 6 | +26.7 | +193.5 | +1159.4 | 3296 | 1.39 → 5.30 |
| 7 | +459.6 | +884.7 | +2441.9 | 3554 | 0.79 → 4.88 |

- The drain removes 70-88 % of S_T's E2E hinge, depending on the candidate (π_e: 0.76 against 3.84). TTFT hinges are unchanged to the third digit.
- The flattery ranges over **1,048** across candidates, 3× the largest drain gain over π_e (350).
- `T=1x+drain` would deviate to candidate 3, the most flattered. On the same requests with arrivals continuing, candidate 3 is the **second worst** (+342).
- Spearman between the rankings: (a)~(b) 0.17, (a)~(c) 0.64, (b)~(c) 0.07.

This is one decision with n = 8, an illustration rather than an estimate. The mechanism is structural, though: S_T contains the requests arriving in the last τ_e before the cutoff, and those are exactly where window E2E tails form.

**Compared with `H=64+drain`.** H=64 drains after ~8 s, with ~140 of ~200 requests still in flight, so nearly the whole set is scored contention-free. `T=1x+drain` scores the routed request (usually done by the cutoff) and the early part of S_T under real contention, so it is less biased for the decision's own request. It is still strongly biased for the late part of S_T, which is large: 855 requests here.

**Suggested fix.**
- Do not present `T=1x+drain` as the fix until the open-loop selection backs it. Reword the docstring: tails of late arrivals are scored "in a contention-free drain", which under-states them.
- In the open-loop `full` child, also score each shorter horizon's set under continued arrivals, as (b) above. The first `n_h` requests of `affected + made` in the `full` child are exactly S_h. This is free and separates drain bias ((a) against (b)) from post-horizon externality ((b) against (c)).
- Cheaper alternatives to compare open-loop: a longer arrival tail before the drain (score S_T, but keep arrivals for T + τ_e before draining); or score S_T under `T=2x+drain` dynamics.
- Land these before launching the open-loop runs: the `full` restriction needs rollout output that the current logs do not hold.

Hashed: no.

### [major] Open-loop agreement is inflated by points where the horizon reaches the window end, which are identical to `full` by construction
`headroom.py:727-758` (`openloop_stats`)

**Failure scenario.** When `t_dec + T ≥` the window's last arrival, a `T=<t>+drain` rollout includes every remaining arrival and then drains to the end. It is then **identical** to `full`: same `n`, same completion times. The Conv probe above shows exact equality at all 12 (point, candidate) pairs. Such points contribute Spearman 1, top-1 agreement, deviate agreement and `ref_gain` = the reference's own. Longer horizons hit the window end at more points, so the selection mechanically favours them.

**Evidence.** Fraction of the `--every 10` points (~165 per window) where the horizon reaches the window end. This uses window arrival times × 0.75, a 20 % cut and π_e's P99 E2E (Tool 94.8 s, Conv 159.7 s; w2000 assumed similar):

| window | T=0.5x | T=1x | T=2x |
|---|---|---|---|
| Tool w0 | 23 % | 48 % | **94 %** |
| Tool w2000 | 28 % | 52 % | **96 %** |
| Conv w0 | 23 % | 40 % | 78 % |
| Conv w2000 | 22 % | 45 % | 85 % |

`T=2x+drain` on Tool is a near-duplicate of `full`. `T=1x+drain`'s statistics are half trivial.

**Reference bias** (task 2):
- The `full` reference's own end-of-window drain is the same drain the scored window metric has (no cool-down cut). So for the pilot's windowed hindsight question it is the right reference, not a biased one.
- It does make points in the last τ_e of the window unlike steady state. Those should be reported separately in any case.

**Suggested fix.**
- In `openloop_stats`, stratify by whether the horizon was truncated (`n_h < n_full`, already in the log) and report the statistics, with counts, on the non-trivial points. Choose the horizon on those, and weigh cost against the trivial fraction of the closed loop.
- This can be recomputed from the existing logs (see finding 5 on keys).

Hashed: no.

### [major] No CI or noise floor for selection; no placebo control for the closed loop's window P99
`openloop_stats` (means only); the recommended rerun plan

**Open-loop.**
- After stratification, `T=1x` has ~80-90 informative points per window, and `ref_gain` is heavy-tailed: per-candidate `full` differences at K=400 span −156 to +2442.
- Rank agreement over 8 candidates mostly measures how bad candidates are ordered. The closed loop acts only on the argmin and the deviate-or-not decision.
- The `full` reference's candidate differences are dominated by requests arriving **after** the horizon: (b)~(c) Spearman 0.07 above, and up to +1557 of candidate 7's +2442 comes from post-cutoff arrivals. Whether that is genuine externality or chaotic trajectory divergence after one changed placement, it caps how well any shorter horizon can agree. It is also something a hindsight full-horizon planner can exploit without a causal policy being able to.

**Closed loop.** Window P99 over ~1,650 scored requests is the ~17th-largest E2E. Any deviation sequence perturbs the trajectory, and the resulting P99 shift has a spread nobody has measured. Without that spread, a few-% "improvement" is not distinguishable from a trajectory lottery, and that is exactly the "positive must not be an artifact" condition.

**Suggested fix.**
- **Primary criterion:** paired per point, `ref_gain` and regret (`s_r[best_h] − s_r[best_r]`), with paired bootstrap 95 % CIs on horizon differences. Keep Spearman as secondary. If the CIs of the two candidate horizons overlap, add `--every 5` on the second window.
- **Noise floor for the reference:** at each point, also roll out π_e's own choice with a decision-irrelevant perturbation, for example +1 output token on one request arriving after the longest finite horizon. Report |Δfull| beside the candidate gains.
- **Placebo for the closed loop:** run 10-20 closed loops in which a random replica is forced at the same deviation rate as the lookahead run (no rollouts; one π_e-speed run each, < 1 CPU-min). Report the lookahead's `vs_pi_e` against that distribution. Pre-register the success rule, e.g. tail4 < placebo 5th percentile on ≥ 3 of 4 windows.

Hashed: no.

### [minor] What a null bounds: say "one-step improvement on the proxy score", and report the reference's own one-step gain
Docstring "Question"; the summary table

- A null closed loop at a horizon that only partly agrees with `full` bounds the gain of one-step lookahead under the proxy score (sum + hinges at π_e's τ), not "the headroom".
- The open-loop run already measures the cleaner quantity: `full`'s own `ref_gain` at sampled π_e states is one-step policy improvement against Q^{π_e} in hindsight. If it is ≈ 0 relative to the noise floor at almost all points, π_e is near-greedy with respect to its own Q, the policy-iteration fixed-point condition, in this proxy. That is the strongest null statement the pilot can make.
- Report its distribution (median, p90, the fraction of points above the noise floor) as the headline bound. The closed loop then tests whether the one-step gains compound into window P99.
- Caveat: those states are π_e's, not the lookahead trajectory's (standard; note it).

Hashed: no.

### [minor] `CODE_SHA` covers all of headroom.py: a post-processing or docstring edit orphans hours of rollout logs
`headroom.py:129-130`, `:882-893`

- Any edit to headroom.py changes every row key. That includes `openloop_stats` (findings 2 and 3 are exactly such edits) and the docstring. An open-loop log (≤ 5.5 CPU-h) or a half-finished closed loop (≤ 20 CPU-h) then cannot be resumed or re-summarised. The rerun starts from zero in a new `<key>.jsonl`.
- The open-loop rows store every per-candidate term, so the statistics are pure post-processing.

**Suggested fix.** Add `--stats-from <jsonl>`, which recomputes `openloop_stats` from a log whose header matches in everything except the stats code. Or hash only the rollout-relevant functions (`child_rollouts`, `horizon_terms`, `request_costs`, `apply_decision`, `resample_outputs`, sim_fork) for the key, which mirrors `code_version`'s function-level hashing.

Hashed: no.

### [minor] The a11 notes are stale against r2/r3
`/Users/kamil/research/vidur-rl-prior-art/notes/a11_fork_spike.md:162-194`

- The notes still describe `commit_policy_state` as the fix; it has been deleted and replaced by sim_fork's `override_next_decision`.
- The last correction says the default is "T = π_e P99 E2E, arrivals continue" (the r2 pure cutoff). It is now `T=1x+drain`.
- They do not record the censoring blind spot, the drain bias or the two-step plan.

Update them before the rerun results are written there.

Hashed: no.

### [nit] Replay checks only π_e's choice; the cost bound is slightly off; measured costs are lower
- `closed_loop` replay (`:632-637`) compares only `pe`, a 1-in-8 coincidence per decision. Cumulatively that is strong, but also comparing `row["t"] == req.arrived_at` makes a divergence certain to be caught at the first wrong decision.
- "Closed loop at `T=1x+drain` ≤ 20 CPU-h": by the docstring's own numbers it is 44.7 × 1,658 = 20.6 CPU-h.
- My probe ran prefix + `T=1x` drain + a continuation to `full` for 8 candidates at t=58 s in 52 CPU-s, ≈ 6.2 per candidate, against the docstring's 10.1 for `full` alone. The stated costs are conservative upper bounds at the current load (~7).

Hashed: no.

## Task 3: is the two-step rerun sensible and sufficient?
**Sensible:** open-loop horizon selection, then a closed loop at the chosen horizon on Tool and Conv, w0 and `--start 2000`. It is the right order, 4 windows give 4 P99 realisations, and the cost is fine: open-loop ≤ 4 × 5.5 CPU-h; closed loop 4 × ≤ 20.6 CPU-h at `T=1x+drain`, or up to ~37 CPU-h each if `T=2x+drain` or `full` wins. Run it on Alpine with `--workers 8`.

**Not yet sufficient.** Before launching:
1. Add the continued-arrival restriction of each horizon's set to the open-loop `full` child (finding 1).
2. Stratify the statistics by truncation and add paired bootstrap CIs on `ref_gain` and regret (findings 2 and 3), preferably recomputable from logs (finding 5).
3. Add the reference noise-floor rollout (finding 3).
4. After the closed loops, run the random-deviation placebo per window and pre-register the success rule (finding 3).
5. Report `full`'s own one-step `ref_gain` distribution as the null bound (finding 4).

The `sampled-lengths` arm stays optional for the pilot's hindsight question. It becomes necessary before any claim about attainable (causal) headroom.
