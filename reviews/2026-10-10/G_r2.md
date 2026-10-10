# Review G r2: closed-loop lookahead headroom pilot (`evolve/rl_plan/headroom.py`, `test_headroom.py`, commit 7e88ab5)

**Verdict: changes requested.** The mechanics are now solid: forcing goes through sim_fork, the tests check real counterfactual exactness, the in-process π_e equals `run_one`, and child failures are handled. All eight r1 findings are fixed or answered.
The new default objective has a structural blind spot. With `T=1x` and `elapsed` censoring, the E2E hinge is exactly 0 for the routed request and for every later arrival, so the tail term sees only requests already in flight. A probe shows censoring flipping a candidate's sign relative to π_e. The proposed `--len 600` sweep cannot separate the horizons. Summary reuse ignores changes to `headroom.py` and `sim_fork.py`.
No fix touches a `code_version`-hashed file: `evolve/rl_plan/*` and `evolve/sim_fork.py` are outside `vidur/`.

What I ran (≤ 2 processes, never more than one simulator at a time apart from the test's forks):
- `pytest -q evolve/rl_plan/test_headroom.py`: **8 passed** in 85 s.
- **Mutation check** (`/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/mutate_fifo.py`): `sim_fork._fifo_move` made to verify but not move the FIFO entry (the E_r1 #4 bug). The deviation test's comparison then **fails** at K=120 and at K=60, and passes with the real move. The test tests its property.
- **In-process π_e vs `ev.run_one`** on Tool 8R s0.75 w0+600 p1: all 30 keys identical (wall excluded), same bundle `8e6e78e985b87b2a:82b720c5cc7e`. The Conv test window is covered by the passing test.
- **Censoring probe** (`/private/tmp/claude-501/-Users-kamil-hermes-memories-20-Research-Vidur-RL/32d6c6bb-b693-42cc-bb1d-80b925355fae/scratchpad/smoke_censor.py`, Conv 4R s1.5 w6000+200, in-process deepcopy forks, serial, 105 s): at K=80, 120 and 160, every candidate was rolled out at T=1x and T=2x as `child_rollout` scores it. Each was then continued to the window end to get the uncensored score of the same affected set.

## r1 findings: status
| r1 | Status |
|---|---|
| 1 short drained horizon | **Fixed in form.** There is now a time horizon with arrivals continuing, plus an H/T sweep. The new default has its own blind spot: finding 1 below. |
| 2 lex rationale | **Fixed.** The default is the scalar `tail` score, a 1e-8 hinge gain can no longer outrank the sum, and the docstring and notes are corrected. `lex` is kept as the documented r1 rule. |
| 3 stale `summary.json` | **Mostly fixed.** Meta = scenario, policy and Mooncake sha, `code_version`, bundle. Rows are keyed by every parameter, including `max_decisions`, `plan_every` and `cut`. The test covers refusal per meta key. Gap: finding 2. |
| 4 sampled futures | **Fixed.** In-flight lengths are resampled conditioned on generated tokens (preemption folding handled). The pool is outside the window, and I checked that `_window_csv` filters, then slices, which matches `output_pool`. The cap is `max_request_tokens`. The test checks no true length survives and the parent is untouched. Residual wording issue: finding 6. |
| 5 child failures | **Fixed.** `select` with a deadline, every child killed and reaped in `finally`, pid/signal/exit status, traceback and stderr tail. Tested for SIGKILL, exception, hang and success, with no zombies left. Not addressed: torch threads at fork (latent only). |
| 6 completion gate and bundle | **Fixed** in `score_output`. Residual: finding 5. |
| 7 duplicate forcing | **Fixed.** `apply_decision` = `override_next_decision(h, choice, record=rec)`. The parent re-checks that π_e's choice matches the children's and that the executed replica matches. |
| 8 tests | **Fixed.** Deviation exactness against a true counterfactual (π_e's own `_select` returns r, so the policy books its own FIFO), with the mutation check above. Parent consistency after 3 forced deviations with forks. Constructed scoring cases. Sampled-future leakage. Fork failures. Summary refusal. `evaluate_candidates` asserts equal `policy_choice` and `n`. |

**Task 3 judgement.** Two tests together establish exactness of the executed trajectory:
- the deviation test: forcing equals the policy choosing r itself;
- the parent test: the forked-and-continued parent equals a fresh run forced with the same decision sequence.

The parent test alone would not catch inexact forcing, since both sides use `apply_decision`, but the deviation test does. `main` refuses when `pi_e_harness` ≠ in-process π_e on 5 metrics. The check is skipped under `--skip-baselines`, and also when the harness run failed (warning only), so a run that skips it has no such guarantee. That is acceptable if documented.

---

### [major] At T=1x the tail term cannot see the routed request or any later arrival; elapsed censoring can flip a candidate's sign versus π_e
`headroom.py:186-207` (`request_costs`), `:345-347` (cutoff), default `--horizon T=1x`, `--censor elapsed`

**Failure scenario.** The rollout stops at `cutoff = t_dec + T`. Every completion it records is ≤ cutoff, and an unfinished request is scored at `cutoff - a`. So any affected request arriving at a ≥ t_dec has e ≤ cutoff − a ≤ T.

With `T=1x`, T = tau_e, which gives `(e/tau_e − 1)+ = 0` **identically** for:
- the routed request itself, since its GlobalScheduleEvent is at its arrival time;
- every arrival during the horizon.

The E2E hinge, the term that was added to see tails, is therefore a function only of requests already in flight before the decision. TTFT is similar: arrivals after `T − tau_f` cannot carry a TTFT hinge, and on Tool that is the last 42 % of the horizon.

Consequences:
- A placement that turns the routed request into a 3×P99 request costs at most `T − e_x(π_e)` in the sum term and nothing in the hinge.
- Protecting one older in-flight request from 1.1·tau to 1.0·tau earns 0.1·w. On Tool, w = 25 × 17.7 s = 442 s, so that is ≈ 44 s of score.
- Lookahead at the default will trade new requests' tails for old ones'. A window P99 E2E regression could then come out of a score improvement. That is an artifact.
- For the same reason, a null result at T=1x does not bound the headroom.

Separately, `elapsed` censoring under-counts by the residual time of every request unfinished at the cutoff. Candidates that leave more work past the cutoff look better.

**Evidence** (Conv 4R s1.5 w6000+200; tau_e 92.4 s, tau_f 66.8 s, w = 668):
- At T=1x, the **post-decision E2E hinge is 0.000 for all 4 candidates at all 3 decisions** under the rollout's scoring.
- On the same affected sets, the uncensored continuation has post-decision hinges of 0.171 / 0.057 / 0.132 / 0.052 at K=80 and up to 0.302 at K=120.
- K=120, π_e = 0. Candidate 2 shows the sign flip:

| Candidate | Censored score | Uncensored score |
|---|---|---|
| π_e (0) | 4051.6 | 4078.1 |
| 2 | 3841.4 | 4307.5 |

  Censored, candidate 2 is better than π_e by 210 (it has 2 unfinished requests at the cutoff). Uncensored, it is worse by 229: its E2E hinge goes 0.116 → 0.418 and its TTFT hinge 0.499 → 0.826. The argmin (candidate 1) was unchanged here, but the score's ordering against π_e, which decides whether to deviate, is not reliable.
- At T=2x on this window nothing is censored (the window ends first), and censored = uncensored exactly.

**Suggested fix (any one; not hashed):**
- **(a) Preferred:** a `T=<t>+drain` mode. Run T seconds with arrivals, then stop arrivals and drain the affected set (the H-mode code path after the time cutoff). No request is censored, and the routed request's and late arrivals' tails become visible. Cost: one drain, ≈ 1 s per rollout on Tool per a11.
- **(b)** Make `remaining` the default and add an estimated remaining wait for requests still before prefill. It remains a lower bound.
- **(c)** Default to T ≥ 2x and score the hinge only over requests with `a ≤ cutoff − 2·tau_e`. This is partial.

Also log per decision the routed request's own E2E and censored flag per candidate, so the effect can be audited. The docstring sentence "`T=1x` covers a tail request's whole lifetime" should be corrected: it covers it exactly up to tau, and no further.

Hashed: no.

### [major] Summary reuse ignores the code that produces the rows: a fix to `headroom.py` or `sim_fork.py` silently reuses old lookahead rows
`headroom.py:591-592` (meta), `:642-645` (reuse)

**Failure scenario.**
- `meta` = scenario, policy name and sha, Mooncake sha, `code_version`, bundle. `code_version` hashes `vidur/**` and four harness functions, but neither `evolve/sim_fork.py` nor `evolve/rl_plan/headroom.py`.
- Rerun into the same `--out` after a sim_fork forcing fix (5c36f7b is under review now; E_r1 #4 was exactly this) or a scoring or censoring change. Every lookahead row with the same parameter key prints "already in …, reused", and the table mixes pre-fix and post-fix rows.
- The commit message claims "no stale reuse".
- `FORMAT` only helps if bumped by hand.

**Suggested fix.** Add `headroom_sha` and `sim_fork_sha` (file sha of both modules as loaded) to `meta`, or to each lookahead row's `params`. The latter keeps the π_e and baseline rows reusable across planner fixes; those rows depend only on `code_version` and sim_fork's stepping, and stepping is guarded by the harness equality check.

Hashed: no.

### [major] The recommended sweep (`--len 600 --horizon H=64,T=0.5x,T=1x,T=2x`) cannot discriminate between the horizons
Usage in the docstring (`headroom.py:71-72`) and the fixer's rerun plan

**Measured** (Tool 8R s0.75 w0+600):
- π_e's P99 E2E = 70.2 s, warm-up cut = 16.7 s, 478 scored requests. The scaled, jittered arrival span is ≈ 83 s.
- **T=1x:** every planned decision (t ≥ 16.7 s) has `t + 70.2 > 83`, so arrivals stop before the cutoff in every rollout. T=1x and T=2x are both "arrivals until the window end, then a censored drain". Neither tests the "sustained load after the decision" property that motivated the time horizon.
- **T=0.5x:** ~half the decisions hit the window end too.
- **Window P99:** over 478 requests it is the ~5th-largest E2E. Which handful of requests land in the tail will dominate differences between the four closed-loop arms.
- **Mismatched horizon:** `T=1x` resolves against the 600-window P99, so it is not the full run's T=1x in seconds (94.8 s on w0+2000).

On w0+2000 (span ≈ 252 s scaled, cut ≈ 50 s), T=1x already hits the window end for roughly the last 45 % of planned decisions. That is faithful to the scored episode in hindsight, but it makes short windows a poor test bed for the horizon choice.

**Suggested fix.** Choose the horizon open-loop first, on the 2000-request window, the a11 way:
- At every 10th decision after the cut (≈ 165 points), score all 8 candidates under each horizon and under a long reference (`T=3x+drain`, or a full continuation as in my probe).
- Report rank agreement with the reference, and how often each horizon's deviation decision vs π_e (gain > 0) has the same sign as the reference's.

That costs ≈ 165 × 8 × (sum of the horizons' rollout costs), about 1-3 CPU-h, and directly answers which horizon ranks candidates like the long run. Then run closed loop on w0+2000 at the chosen horizon, on at least 2 windows, since one window's P99 is ~16 requests.

On cost: 14 CPU-h at T=1x matches the measured 30.4 CPU-s per decision × 1,658. T=2x will be ~1.5-2× that.
- On this Mac (12 cores, load ~6 from long simulations), 4 workers is the right ceiling.
- On Alpine acpu, `--workers 8` would run all candidates in one batch per decision.
- The Conv figure in the docstring ("T=1x (32.9 s)") does not match π_e's P99 E2E on that window, which is 92.4 s as measured above and in the test scenario. Recheck the 6.0 CPU-s figure.

**`--plan-every K`** answers a different question: the headroom of lookahead applied at 1/K of the decisions, with π_e elsewhere. Its rollouts actually match the executed future better, since most later decisions really are π_e's. A null at K>1 does not bound the K=1 headroom (gains need not be additive), and a positive is not a per-decision estimate. It is fine as a cheap first look if reported under that name. The docstring already says so. `--max-decisions M` is similar, but it is biased to the start of the window.

Hashed: no.

### [minor] The `tail` weight is not the window objective's marginal rate as the docstring derives it; TPOT/TBT are measured but not summarised, and the "< 5 %" claim is false
`headroom.py:38-47`; `evaluate.py:score`

**The objective.** `score()`'s per-scenario term is `Σ_m w_m log(P99_m / bar)`. The fitness is `−(mean + 0.5·max)` over scenarios, so the weight on each log P99 is 0.375 for a single scenario, not 0.25. The mean-E2E term is the one-sided `log(mean/best_mean)`, applied only when mean > 1.05 × the best baseline's mean.
- At π_e's operating point it is very likely inactive. On the full Tool s0.75 trace, the best baseline mean is sticky_lor's 25.8 s. The w0+2000 baselines are not cached, so this is unverified for that window.
- When inactive, the objective's marginal weight on the sum is 0. When active, w = 37.5 m.
- `25 m` is therefore a design choice, a mean regulariser, not a derivation from the objective. It may be a reasonable choice, but the docstring presents it as "the window objective's" trade-off.
- A related detail: requests that arrived before the warm-up cut are in early decisions' affected sets but not in the window metrics.

**TPOT/TBT.** The justification "they moved by < 5 % under any placement rule seen so far" is contradicted by the r1 table in `a11_fork_spike.md`: LOR's P99 TBT is 0.246 vs π_e's 0.224, +9.8 %.
- Could lookahead win E2E/TTFT by wrecking token pace? Partly yes. Per-request TPOT feeds the E2E sum, but P99 TBT (stalls from prefill interference on decoders) is a tail of token gaps that an E2E sum dilutes.
- The rows do record `p99_tpot` and `p99_tbt`: `score_output` = `run_one`'s metrics. So a regression would be visible, but only if the reader compares by hand.

**Suggested fix.**
- Reword the derivation as a choice.
- Add to every lookahead row its `tail4` log-ratio vector against π_e: `0.25·log(P99_m/P99_m(π_e))` per metric, plus `log(mean_e2e ratio)`. A win bought with TBT then shows in the same row.
- Optionally log the affected set's max TBT per candidate (not scored) to diagnose.

Hashed: no.

### [minor] A long run has no resume, and a failed lookahead row is cached as if final
`headroom.py:491` (`open(log_path, "w")`), `:646-653`

**No resume.**
- One child failure (jetsam on a busy Mac, a timeout) at decision 1,500 of 1,658 raises out of `closed_loop`. Nothing is stored, and the rerun truncates the `.jsonl`: ~13 CPU-h lost.
- Resume is cheap and already proven: replay the logged `chosen` sequence with `apply_decision`, exactly what `test_parent_continues_like_fresh_forced_run` checks, which takes ~25 s for a Tool window. Then continue planning.

**Failed rows cached.** If the completion gate fails, `score_output` returns `ok: False`. The row is still stored under its key and is "reused" forever after, unlike the baselines, which are deliberately not stored on failure.

**Suggested fix.**
- Append to the log, and add a `--resume` that replays it.
- Do not store `ok: False` rows, or have reuse skip them.

Hashed: no.

### [nit] "Sampled futures" de-hindsight only output lengths
`headroom.py:54-61`

Arrival times, prompt lengths, prefix-hash ids and sessions of future arrivals stay true. The docstring is accurate ("the output length … is resampled"), but `future: sampled` rows will be read as the information gap of plan §6 E2.3, and a11 defines that gap as resampled arrivals *and* lengths. Name the mode `sampled-lengths`, or say in the row that it is a partial gap.

Two further notes:
- The policy's views (`outstanding`, `pending_prefill`, `waiting`, `running`, `kv_usage`) expose no decode lengths, so π_e's choice in a child cannot depend on resampled lengths. The parent's `policy_choice` check would catch it if it did.
- For `mix-*` transforms the pool excludes only the base trace's window. Not used by the pilot.

Hashed: no.

## Other checks (no finding)
- **Common random numbers and ties.** In true-future mode the children fork from one parent state with identical RNGs. In sampled mode one seed per decision index (`1000003·n + 7`) is shared by all candidates and all horizon arms. For `tail`/`sum`/`hinge`, ties and gains ≤ `min_gain + 1e-9·|s_pe|` keep π_e. The tie test covers a duplicated candidate. `decided_by` attributes a deviation to its largest per-term gain.
- **Affected set.** The affected set (`_inflight` + every generated arrival ≤ cutoff) is identical across candidates by construction. `evaluate_candidates` asserts equal `n`.
- **Window end in hindsight.** Rollouts near the window end see no more arrivals, which is what the scored window also sees. This is faithful in hindsight.
- **Fork plumbing.** Children never inherit a pending override: the parent sets one only after collecting. Later children inherit earlier siblings' read fds but no write fds, so EOF detection is unaffected.
