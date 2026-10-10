# Review G r8: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, fix commit e153a6c, tree at 3d498d8)

**Verdict: approve with minors. Stop rule met: no blocking or major finding.**
1. The G_r7 pairwise reference is implemented as specified. It is used everywhere the centred N was used: band, below/worse/within, pace limit, `ruled_out`, `detectable_pct` and `pooled_stat`. The 380 dependent pairs do not invalidate any stated quantile or coverage. All coverage figures are empirical, and they hold under my rerun (window bound 96-98 %, pooled 94-95 %). The only effect of the dependence is that the pooled null leaves out scale-estimation error. That costs about 1-2 points, and it falls on the conservative side of the gate. Minor 1.
2. The operating characteristic reproduces. True null: "no headroom" 69-70 %, the rest "inconclusive", false "headroom" 0 in 600 pilots. True -20 %: "headroom" 68-73 %. Tests: **21 passed** (384 s). No finding touches a `code_version`-hashed file, and none changes run code.

What I ran (no output trees; no simulator process of mine besides pytest; at most 2 Python processes of mine at once):
- `pytest -q evolve/rl_plan/test_headroom.py`: 21 passed, 2 warnings (wandb/seaborn), 383.7 s.
- `scratchpad/g8/oc.py`: model B pilots (`test_headroom._window_b`, i.e. G_r7's model, with TPOT/TBT drawn) through the real `pilot_verdict` with `ev.code_version` patched. 950 pilots in all.
- `scratchpad/g8/pooled.py`: `hr.pooled_stat` on the pairwise reference compared with a run-wise leave-out null. 400 pilots, rho = 0.
- Scratchpad = `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/g8/`. Results are in `*.jsonl` and `pooled_r0.txt`.

## G_r7 items: status
| r7 | Status |
|---|---|
| major: one-draw reference | **Fixed.** `verdict_inputs:1260-1265` builds `N` from `itertools.permutations(N_raw, 2)`, so there are 380 values from 20 runs, symmetric, with median exactly 0. `window_verdict` and `pooled_stat` consume this `N`. `N_raw` is still used for `noise_offset` and `zero_in_raw_noise_band` (the uncentred check), which is correct. `n_noise` now reports runs (20) and `n_reference` reports pairs (380), and a test checks both. `NOISE_CENTRE` is in `post_sha`'s constants, so stored rows recompute. |
| minor: pace guard not exercised | **Fixed.** `_window_b` draws L's TPOT/TBT like everything else. The test asserts a rate (>= 45 % "headroom" at -20 %) instead of 5/5. Trade windows appear (2-3 % of windows at -20 %), so the guard is exercised. |
| minor: power statement | **Fixed.** The POWER paragraph says that roughly 20 % or more is needed on both tails in every window, and that an "inconclusive" at 10-15 % is not evidence against headroom. |
| nit: bound wording | **Fixed.** "nominal 97.5 %, realised ~96-98 %" for the window bound, "~95 % realised" for the pooled bound. Both match my rerun (below). |
| test edits | The changed constants (`lab(-0.3, 0.1)`, `_vs(-0.07, 0.1)`, trade `(-0.3, 0.15)`) follow from the band being sqrt(2) wider. The assertions still test the clause they name: I checked that `vw` is still below the band but not `better`. |

## Task 1: is the pairwise reference handled correctly given that the 380 pairs are not independent?

**Per-window quantiles.** P2.5/P97.5 of the 380 differences estimate the quantiles of N_j - N_k (a U-quantile). Their precision is that of ~20 runs, not 380. However, no claim in the code relies on 380 independent values:
- The coverage figures in the docstrings are labelled as empirical, from the synthetic model.
- Measured window-bound coverage at delta = 0:

| rho | E2E | TTFT |
|---|---|---|
| 0 | 96.5 % | 96.2 % |
| 0.1 | 97.8 % | 96.6 % |

This is consistent with "~96-98 %". At rho = 0 the reference is exact rather than conservative, and coverage is still >= 96 %.

**Pooled statistic: the one place where the dependence matters.**
- `pooled_stat`'s null takes each of the 380 values as a "leave-one-out draw". It drops D_jk, but 37 other pairs that contain run j or run k remain (including D_kj = -D_jk).
- The scale s_w is therefore effectively the realised one for every draw, so the null of Z is conditional on the realised s_w. L's z, by contrast, carries the error of s_w, because L is out of sample. The r6 per-run LOO had that error built in.
- The expected effect is a slightly narrow null: Var(z_L) ≈ 1 + Var(ŝ/s), with the relative sd of the IQR scale ~0.2-0.25 at 20 runs.
- Measured at rho = 0, 400 pilots (MC se ~1.1 pt):

| | bound covers δ (E2E / TTFT) | false gate rate Z < q5 (E2E / TTFT) |
|---|---|---|
| current (pairwise LOO) | 95.5 % / 94.0 % | 6.8 % / 6.0 % |
| run-wise leave-out (scale from the other 18 runs' pairs) | 96.5 % / 95.5 % | 5.5 % / 5.2 % |

- The oc runs agree: pooled coverage 93-95.5 % across null and -20 % pilots, and gate rate 5.0-7.7 % under the null.
- **Consequences:**
  - The gate errs toward "inconclusive" (a true null is held back from "no headroom" slightly more often).
  - The printed pooled bound is at most ~1 point short of its 95 %.
  - Neither makes a conclusion wrong or misleading. This is minor 1.

**Consistency between the pooled and window references.** Both use the same `N`. The pooled z uses median(D) = 0 and s = IQR(D)/1.349 ≈ sqrt(2)·σ_run. That is the right scale for L - δ under rho = 0 and conservative above it. The bound algebra (Z - q95)/Σ 1/s_w is consistent with the window rule. Under a common δ, the δ terms cancel exactly, so coverage does not depend on δ. Measured: 93-95 % at both δ = 0 and δ = -20 %.

## Task 2: operating characteristic (model B, 20 noise and 20 placebo runs, real `pilot_verdict`)

| truth | rho | pilots | "no headroom" | "inconclusive" | "headroom" | window labels |
|---|---|---|---|---|---|---|
| null | 0 | 300 | **69.3 % ± 2.7** | 30.7 % | **0** | null 85.1, inconclusive 7.6, negative 7.1, positive 0.2, trade 0.1 |
| null | 0.1 | 300 | **70.0 % ± 2.6** | 30.0 % | **0** | null 87.1, inconclusive 7.2, negative 5.2, trade 0.4 |
| -20 % both | 0 | 150 | 0 | 26.7 % | **73.3 % ± 3.6** | positive 91.5, inconclusive 5.8, trade 2.7 |
| -20 % both | 0.1 | 200 | 0 | 32.0 % | **68.0 % ± 3.3** | positive 90.5, inconclusive 7.4, trade 2.1 |

- **Confirmed:**
  - Null gives "no headroom" ~70 % (the commit says ~72 %).
  - False "headroom" never occurred (0 / 600 null pilots).
  - -20 % gives "headroom" ~70 %.
  - "No headroom" never occurred at -20 %.
- **Why a null fails to reach "no headroom":**
  - About 7 % of windows have a tail below the band by chance, which makes the window "inconclusive". So P(all 4 windows null/negative) ≈ 0.93^4 ≈ 0.74.
  - The pooled gate then removes another ~5 %.
  - This is a property of the design (8 one-sided tests plus the pooled gate), as G_r7 said.

---

### [minor] Pooled null is conditional on the realised noise scale (the pairwise "leave-one-out" is in-sample)
`pooled_stat:1342-1347` (LOO over `N`, which is now the 380 pairs)

- **Failure scenario:** the noise runs of a window have a small IQR by chance. L's z is inflated, but the null draws are scaled by the same small s_w, so the null does not widen to match.
- **Effect at rho = 0:**
  - The null 5 % gate fires 6-7 % of the time instead of 5 %. That is conservative: slightly more "inconclusive".
  - The pooled bound covers 94.0-95.5 % instead of 95 %.
- **Evidence:** the table in Task 1, `pooled_r0.txt`.
- **Fix (optional):** draw the null for pair (j, k) against the pairs of the other 18 runs, i.e. `rest = np.delete(x_raw, [j, k])` with its own pairwise median and scale. This needs `pooled_stat` to receive `N_raw`. Measured: 5.2-5.5 % and 95.5-96.5 %.
- **Recommendation:** not worth a change on its own. Any `post_sha` change makes every open-loop row "stale" (`weight_status`), which forces an `--mode openloop` replay per window (cheap, no rollouts, but it is a step). Adopt the fix only if post-processing changes anyway. Otherwise state the bound as "≈95 % (93-95 % realised)".
- Hashed: no.

### [minor] A per-window `negative` label is not evidence of harm: ~1 in 4 true-null pilots has one
`window_verdict` labels; docstring "Labels"

- **Failure scenario:** the lookahead is truly neutral. Each window is still labelled `negative` 5-7 % of the time, because "worse" is one-sided at 2.5 % on each of E2E and TTFT, and the 20-run quantile is realised at ~96.5 %.
- **Evidence:** 25.3 % (rho = 0) and 20.3 % (rho = 0.1) of null pilots contain at least one `negative` window.
- The claim is unaffected, since `negative` counts toward "no headroom". The risk is only in the write-up, if it says "the lookahead was worse than the lottery in window X".
- **Fix:** one sentence in the docstring or the report template: "a single `negative` window occurs in ~1/4 of true-null pilots; read it as `null`".
- Hashed: no.

### [nit] Three different figures for the true-null "no headroom" rate
- The figures differ by source:
  - Commit message: "~72 %".
  - Module docstring POWER: "~3/4 of the time".
  - Docstring of `test_synthetic_operating_characteristic`: "~77 %" (G_r7's number).
- Measured here: 69-70 % ± 2.7.
- Use "~70 %" in both docstrings.
- Hashed: no. The test docstring is not in `post_sha`. The module docstring is not in `post_sha` either (`_src_sha` hashes function sources only).

## Task 3: anything else that could make a conclusion misleading
- I found none. Points checked:
  - **Centring.** The jitter offset h and pi_e's own draw c0 cancel in D. The lookahead is unjittered, so the remaining assumption is only that it has no jitter harm itself. This is stated, and it is settled.
  - **Outliers.** A single outlying noise run puts 19 of the 380 differences (5 %) at the outlier's level on each side, so P2.5/P97.5 of D move to roughly that level. This widens the band, which is conservative for "headroom". `ruled_out` and `detectable_pct` then print the correspondingly loose bound. The direction is honest, not misleading.
  - **Zero-spread noise.** `pooled_stat` returns an error, so "no headroom" is unreachable for that metric (correct). `zero_in` is trivially true.
  - **Post-processing changes.** `post_sha` covers `verdict_inputs`/`window_verdict`/`pooled_stat` and `NOISE_CENTRE`. Any later change makes `pilot_verdict` report "stale" until the open-loop rows are replayed. It cannot silently mix old and new rules. The weight choice cannot change on replay unless `weight_sweep`, `crossfit` or the constants change.
- Runs: no comment, since the run code is unchanged and settled.
