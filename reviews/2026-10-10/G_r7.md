# Review G r7: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, fix commit ec5bc77, tree at b09f8e3)

**Verdict: changes requested (one post-processing change; the runs can launch now).** Every G_r6 fix is in the code and tested; tests **21 passed** (330 s, at HEAD b09f8e3 = after the v4.1 merge, code_version `cbeb5161f046`).
1. One major: centring N on its median removes pi_e's own lottery draw from the reference, while L still contains it. L is the difference of two draws (lookahead minus pi_e on the same arrivals); the centred band describes one. Under the variance split the G_r6 jitter data support, the "95 %" bounds cover ~88 %, and a true null reads "negative" in 20 % of windows. Pairwise differences N_j - N_k fix it: they are still centred at 0 (offset-free), and they restore the operating characteristic. It is a ~5-line change in `verdict_inputs`, post-processing only.
2. 20 -> 40 noise runs: it helps only under the current one-draw band (true-null "no headroom" 51 -> 60 %). With the pairwise reference it changes nothing (77 % vs 76 %). Optional.

No finding touches a `code_version`-hashed file. Settled decisions are not reopened: the pairwise reference keeps the centring (median 0, jitter offset removed) and changes only the spread.

What I ran (no output trees; disk 17 GB free before and after; no simulator process of mine besides pytest):
- `pytest -q evolve/rl_plan/test_headroom.py`: 21 passed.
- `scratchpad/g7/oc.py`: synthetic windows fed through the real `pilot_verdict` (`ev.code_version` patched). About 6,000 synthetic pilots in all.
- Scratchpad = `/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/g7/`. Results are in `oc*.jsonl`.

## G_r6 items: status
| r6 | Status |
|---|---|
| 1. weight rule / no-headroom gates exclude each other | **Fixed.** `rule_met` and `gain_xfit` are reported as mechanism (i)/(ii) and are no longer gates. Tests cover `rule_met: False` and `gain_xfit.lo > 0` windows reaching "no headroom". |
| 2. bounds not data-based | **Fixed in form.** Per window: `ruled_out` = L - P97.5(N). Pooled: `bound` = (Z - q95)/sum 1/s_w. Both follow from the stated model and depend on the observed L/Z (test). The model's spread assumption is wrong: finding 1. |
| 3. noise band offset | **Handled by centring** (settled). The lor/mooncake check is reported honestly in the docstring and in `noise_offset` / `zero_in_raw_noise_band`. `inconclusive_noise_model` is unreachable once centred, as documented. |
| 4. below-band-and-worse labelled negative | **Fixed** (`negative` requires no tail below the band; test). |
| 5. one setup across windows | **Fixed.** `meta` must be identical except `scenario`, and `code_version` must be current. I checked that it will not refuse a valid set: `.cache_v4/baselines.json` holds one bundle fingerprint (`8e6e78e985b87b2a:82b720c5cc7e`) for every Tool and Conv scenario, so Tool and Conv windows share the bundle. |
| nits (error reasons, Alpine chain) | **Fixed** (`weight_status`; the chain runs openloop && closed && placebo && noise). |

---

### [major] The centred noise band describes one lottery draw, but L contains two; the stated bounds over-claim and a true null is often labelled "negative"
`verdict_inputs:1240-1242` (centring), `window_verdict:1247-1292`, `pooled_stat`, docstring "EFFECT SIZE" ("L = delta + a lottery draw distributed like the centred N")

**Model.** Write a policy's window log P99 as μ_π + g(A) + c(π, A):
- g is the arrival component, shared by all policies on the same arrivals.
- c is trajectory chaos. Any change of decisions or arrivals redraws it.

The three quantities the verdict compares are then:
- Lookahead: L = δ + c_la - c0, where c0 is pi_e's own unjittered draw. The c0 term is there because pi_e is itself only one realisation of the lottery.
- Noise: N_j = h + (g_j - g0) + c_j - c0, where h is the jitter harm.
- Placebo: P_j = δ_pl + c'_j - c0. It shares c0 with L, so the placebo comparison is correctly paired.

What centring does:
- Centring N on its median removes h **and** c0. N_j - med N has spread sqrt(σ_g² + σ_c²), i.e. one draw.
- L - δ = c_la - c0 has spread sqrt(2) σ_c. That is two draws, and the c0 in L is not cancelled by anything.
- The centred band is therefore too narrow whenever σ_g < σ_c, i.e. whenever the shared arrival component is less than half the lottery.

**What the data say about the split.** The G_r6 runs used the same jitter seeds (`2_000_003 + seed`) for pi_e, lor and mooncake. The correlation across policies at equal seeds estimates the shared share ρ = σ_g²/σ²:

| | pi_e-mooncake | pi_e-lor | mooncake-lor |
|---|---|---|---|
| E2E | -0.13 | -0.09 | 0.45 |
| TTFT | 0.54 | -0.26 | -0.09 |

- The mean is about 0.07 (n = 9 each, so this is weak evidence).
- The lottery is mostly trajectory chaos, so the centred band is too narrow by roughly sqrt(2(1 - ρ)) ≈ 1.3-1.4.
- The offsets agree with this reading. All three policies were worse when jittered, which is consistent with a shared h > 0 plus a random c0 per policy.

**Evidence.** Synthetic pilots through the real `pilot_verdict`. Each window has 20 noise runs and 20 placebo runs. Per-draw sd: E2E 0.022 and TTFT 0.041 (the G_r6 Tool w0 values), TPOT/TBT 0.02. Jitter offset h = +0.022.
- Model A is the fixer's synthetic: c0 ≡ 0, L = δ + one draw.
- Model B is the structure above with c0 random, ρ = 0.1.

| true null (δ = 0) | "no headroom" | "inconclusive" | "headroom" | windows `negative` | pooled 95 % bound covers δ | window 97.5 % bound covers δ |
|---|---|---|---|---|---|---|
| current, model A (fixer's), 400 pilots | 50.7 % | 49.2 % | 0 | 12.1 % | 94.3 % | 93.2 % |
| current, model B ρ = 0.1, 400 | **38.3 %** | 61.8 % | 0 | **20.1 %** | **88.5 %** | **88.7 %** |
| current, model B ρ = 0.4, 300 | 49.3 % | 50.7 % | 0 | 13.0 % | 93.0 % | 92.7 % |
| **pairwise**, model B ρ = 0.1, 300 | **76.7 %** | 23.3 % | 0 | 4.2 % | 95.0 % | 98.0 % |
| pairwise, model B ρ = 0.4, 200 | 90.0 % | 10.0 % | 0 | 1.8 % | 96.5 % | 99.1 % |

Why this would change a conclusion:
- The bound printed with "no headroom" is stated as one-sided 95 %. It covers ~88 % and is about 1.3-1.4x too tight.
  - On Tool w0 with L = 0, the window can exclude only gains beyond ~6 % (E2E) or ~11 % (TTFT). The docstring says 4-8 %.
  - That is the G_r6 over-claim in a smaller form.
- One in five truly-null windows would be reported as "lookahead worse than the lottery" (`negative`).
- Under a true null, "inconclusive" is the most likely outcome.

**Fix** (post-processing; commit before any closed, placebo or noise output is read):
- Make the reference distribution for L the ordered pairwise differences D = {N_j - N_k : j ≠ k} (380 values from 20 runs). Use D everywhere the centred N is used now: band, below/worse/within, pace limit, `ruled_out`, `detectable_pct`, and `pooled_stat`. `pooled_stat` falls back to its seeded draws, since 380^4 > `max_enum`.
- Properties of D:
  - It is symmetric about 0, and h and c0 cancel. The settled centring is preserved, and so is the docstring sentence "assumes the lookahead shares pi_e's offset". D actually needs only that the lookahead suffers no jitter harm, because L is unjittered.
  - Var(D) = 2(σ_g² + σ_c²) >= Var(L - δ) = 2σ_c² for any split. So the reference is valid whatever ρ is: exact at ρ = 0, conservative above it.
- Implementation: in `verdict_inputs`, `NOISE_CENTRE = "pairwise"` builds `N = [{m: a[m] - b[m]} for a, b in itertools.permutations(N_raw, 2)]`. I fed exactly this through the existing code to get the rows above.
- Reword the EFFECT SIZE paragraph: "L is the difference of two lottery draws (lookahead and pi_e on the same arrivals); its reference is the difference of two noise runs".
- Add a test with c0 random (model B) asserting that the pooled bound covers δ at about the nominal rate.

Hashed: no. `post_sha` covers `NOISE_CENTRE` and `verdict_inputs`, so stored rows recompute. The runs, the weight choice and the open-loop rows are unaffected.

### [minor] The synthetic "true positives read headroom" test never exercises the pace guard; with a realistic pace draw a true -15 % gain often reads "inconclusive" via false "trade" windows
`test_headroom.py::test_synthetic_null_reaches_no_headroom` (`_window`: `_vs(*L)` sets L's TPOT/TBT to exactly 0.0)

- In the test, L's TPOT and TBT get no lottery draw, so the pace guard can never fail. The 5/5 "headroom" therefore says nothing about the guard.
- With L's TPOT/TBT drawn like everything else (no true pace cost), the guard runs as 8 one-sided tests: 4 windows × TPOT/TBT.

| true -15 % on both tails, no pace cost | "headroom" | windows `trade` |
|---|---|---|
| current, model A, L pace exactly 0 (the test's setup) | 83.5 % | 0 |
| current, model A, L pace drawn | **64 %** | 6.6 % |
| current, model B, L pace drawn / exactly 0 | 50 % / 66.5 % | 6.1 % / 0 |
| pairwise, model B, L pace drawn / exactly 0 | 27 % / 31.5 % | 1.9 % / 0 |
| pairwise, model B, **-20 %**, L pace drawn / exactly 0 | 66 % / 73 % | 2.2 % / 0 |

- After the major fix, the wider pairwise pace band brings the guard's false "trade" rate down to ~2 % per window, costing about 5-7 points of "headroom". That is acceptable, so I do not ask for a pooled pace test.
- The remaining loss is the TTFT lottery itself. One draw has sd ~4 % on Tool w0, so a difference of two draws has sd ~6 %.

**Fix:**
- Draw L's TPOT/TBT in `_window`.
- Assert the "headroom" rate at a realistic effect (e.g. >= 50 % at -20 %) instead of 5/5 at -15 %.
- State in the docstring that, on the G_r6 spreads, "headroom" needs gains of roughly 20 % or more on both tails in every window. That way an "inconclusive" at a true 10-15 % gain is not read as a negative. The per-window `detectable_pct` already prints the actual figure.

Hashed: no.

### [nit] Bound wording
- `window_verdict` calls `ruled_out` a "one-sided 97.5 % confidence bound". The empirical P97.5 of 20 runs is a ~93 % predictive bound even when the model is right (model A: 93.2 %; with 40 runs, 94.6 %).
- With the pairwise reference it is ~98 %.
- Say "≈97.5 %" or give the realised level.

---

## Task 2: the synthetic experiments, rates per conclusion

All rows are through the real `pilot_verdict` with 20 noise and 20 placebo runs per window. "Trade" = E2E -15 % and TTFT +15 %. Percentages are over pilots.

| truth | current, model A (fixer's) | current, model B | pairwise, model B |
|---|---|---|---|
| null | NH 50.7 / inc 49.2 / H 0 | NH 38.3 / inc 61.8 / H 0 | **NH 76.7 / inc 23.3 / H 0** |
| -15 % both | H 64 / inc 36 | H 50 / inc 50 | H 27 / inc 73 (-20 %: H 66) |
| trade | inc 100 (97 % of windows `trade`) | inc 100 (94 %) | inc 100 (77 %; the rest `inconclusive`) |
| -3 % both | NH 5 (bound covers δ in 53 % of them) | NH 3 (33 %) | NH 17 (89 %; 95 % overall) |

Reading the table:
- **The fixer's figure reproduces.** Under model A a true null reads "no headroom" 51 % of the time (fixer: ~55 %), "inconclusive" otherwise, and never "headroom".
- **No false "headroom".** It never occurred in any null configuration: about 2,000 null pilots, with windows `positive` <= 0.3 %.
- **No false "no headroom" on a trade.** A trade never reads "no headroom".

**20 -> 40 noise runs.**

| true null, "no headroom" rate | 20 runs | 40 runs |
|---|---|---|
| current, model A | 50.7 % | 59.8 % |
| current, model B | 38.3 % | 43.0 % |
| pairwise, model B | 76.7 % | 76.0 % |

- Under the current one-draw band, 40 runs help: model A window-bound coverage goes from 93 % to 95 %. But 40 runs do not repair model B's 88 % coverage.
- Under the pairwise reference they change nothing material.
- What limits a true null below ~80 % is the design itself: 8 one-sided per-window tests plus the pooled test. The noise count does not.

**Recommendation:**
- Pre-register the pairwise reference (finding 1). It is the material improvement: a true null reads "no headroom" 77 % of the time instead of 38-51 %, the bounds reach nominal coverage, and false "headroom" stays at 0.
- Going to 40 noise runs is harmless and cheap: about 20 × ~17 s of pi_e per window on Tool, and Conv is unmeasured. It is not needed. If you do it, fix the count before launch.
- I do not recommend loosening the "all 4 null/negative" conjunction. A window whose tail falls below the pairwise band is real evidence of a possible gain.

## Task 3: data-based bounds and median-centring
- **Formulas.** Correct for the model they state. The pooled bound (Z - q95)/sum 1/s_w achieved 94-95 % coverage under model A, and the test checks that it depends on the observed Z.
- **Labelling of the centre.** Honest. The offsets are reported, the assumption "the lookahead shares pi_e's offset" is stated, and so are the uncentred check and the lor/mooncake numbers.
- **The spread is not right.** The model line "L = delta + a lottery draw distributed like the centred N" is the part that fails: finding 1. With the cross-policy correlation near 0, the bounds over-claim by ~1.3-1.4x. The pairwise reference fixes this and keeps the centring.

## Task 4: ready to launch?
- **Runs: yes, now.**
  - Run code is unchanged since the r6 approval. Tests pass at HEAD with v4.1 merged.
  - Every pending change is post-processing, covered by `post_sha`. It does not touch the weight choice.
  - Launch at code_version `cbeb5161f046`. Ship the matching `prediction_bundle_8e6e78e985b87b2a.pkl` so all 4 windows share `meta`.
  - Any further hashed change before the verdict makes `pilot_verdict` return `invalid_windows`. That is the correct behaviour, but plan the ~4-day window around it.
- **Decision rule: one change first.**
  - Commit the pairwise reference and its test before any closed, placebo or noise output is read. The pace-guard test fix and the docstring power statement can go in the same commit.
  - Decide 20 vs 40 noise runs before the first noise job.
