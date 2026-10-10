# Review G r1: closed-loop lookahead headroom pilot (`evolve/rl_plan/`, uncommitted)

**Verdict: changes requested.** The mechanics are sound: forcing is exact for l3_48a0, the parent stays consistent, RNG is isolated, and the in-process π_e run matches `run_one`.
The score measures something narrower than "P99 headroom". With H=64 ≈ 8 s of arrivals followed by a drain into an empty system, a null or negative result does not bound the headroom. On Tool, lex is driven by noise-level hinge deltas, contrary to its documented rationale.
`summary.json` silently reuses a stale π_e, tau and cut. No fix touches a `code_version`-hashed file (`evolve/rl_plan/*`, `evolve/sim_fork.py` are outside `vidur/`).

What I verified (≤ 2 processes, small window `Conv 4R s1.5 w6000+200 p1`, policy l3_48a067a7df63):
- **Forcing exactness.** At decision 120 (π_e picks 2), I forced each of 0, 1, 3 two ways: headroom's `force_first` + `commit_policy_state`, and sim_fork's `override_next_decision` (bc0b369). I ran both to the end of the trace. `request_metrics.csv` is byte-identical in all 3 cases.
- **In-process π_e vs `evolve.evaluate.run_one`.** All 22 common metrics are identical, with the same bundle `96bd5ae2116525e5` and empty clamp_hits. The Tool/Conv w0+2000 summaries also match `pi_e_harness` exactly on the 5 stored metrics.
- **Rollout bookkeeping.** `_inflight` covers every routed-not-finished request plus the queued one (20 = 20 and 25 = 25 vs `_assigned` + queue). `n` is equal across all candidates (28, 33). π_e's choice is the same in every child.
- `pytest -q evolve/rl_plan/test_headroom.py`: 2 passed in 70 s.

## Answers to the six questions (short)
1. **Forcing.** `commit_policy_state` is exact for l3_48a0. The policy is `choose()`-only, and its only state is the FIFO. Trimming depends only on the views, which are taken before the pick, so "trim, then append n at pick, then move to choice" equals "trim, then append n at choice". `ctx.rng` draws are identical. The byte-identical continuation above confirms this. Use sim_fork's mechanism anyway: see finding 7.
2. **Scoring.** Drain scoring, not truncated. The candidates share common random numbers exactly: in true-future mode the future is deterministic and RNG states are identical at the fork; in sampled mode every candidate gets the same `sample_seed`. Ties go to π_e via the `c != pe` key and the final 1e-9 check. What the score measures: findings 1 and 2.
3. **Parent consistency.** Yes. The ledger books `best`, the FIFO entry is moved, and the session map holds `best`. The verified continuation equals sim_fork's forced continuation. Rollouts run in separate address spaces, so the parent's global, `ctx.rng`, generator and torch RNGs cannot change. The neutral test (`candidates="pe"`, byte-identical CSV) confirms the fork path leaves the parent untouched.
4. **os.fork.** Works in practice (≈13k forks in the Tool run). Gaps: finding 5.
5. **Baselines.** Same window CSV, seed, bundle and metric formulas as `run_one`, verified. Missing pieces (completion gate, bundle record): finding 6. Stale reuse: finding 3.
6. **Tests.** They test their stated claims, but not exactness against a counterfactual, the deviation path, or the scoring: finding 8.

---

### [major] The rollout objective is a short-horizon, drained proxy; it does not measure window P99 headroom
`headroom.py:152-157` (`child_rollout`), `main` default `--H 64`

**Failure scenario.** The pilot asks whether per-decision lookahead improves window P99 tails. Each rollout admits only H=64 future arrivals, then stops arrivals (`g.num_requests = g.num_requests_generated`) and drains in an otherwise empty system.

**Evidence.** On Tool s0.75 (the run's own log), 64 decisions span a median of 7.8 s of arrivals (p10/p90: 5.9/9.5 s). For comparison, π_e's mean E2E is 17.7 s and its P99 E2E is 94.8 s.
- The affected set is a median of 206 requests, ~140 of them already in flight. Its fate is scored with no competing load after ~8 s.
- The externality of a placement on traffic arriving after those 8 s is invisible: cache affinity built up, KV pressure, a replica's backlog when the next burst lands. Under sustained load, that externality is what forms P99 tails.
- The objective is "summed E2E of in-flight requests plus the next 8 s of arrivals, when the world then goes quiet". One-step policy improvement guarantees nothing about the window P99 under this proxy.

A negative or null result is therefore not evidence of small headroom, and the Tool row (even unconfounded) would not support "lookahead does not help".

**Suggested fix.**
- Use a time horizon comparable to the tail: run the rollout for T seconds of simulated time (T ≳ P99 E2E, e.g. 100 s) with arrivals continuing.
- Score truncated at T: completed requests by E2E, unfinished ones by their age at T (or age plus a remaining-work estimate). Alternatively, drain only after a long horizon.
- At minimum, report an H sweep (64 / 256 / 1024) on a short window before the full runs.

Hashed: no.

### [major] The lex rationale is false on Tool: hinge is non-zero at 77 % of decisions, and deviations select on noise-level hinge deltas that usually worsen the mean
`headroom.py:13-15` (docstring), `:218-221`; `a11_fork_spike.md` ("pure hinge is 0 for all 8 candidates at most decisions")

**Failure scenario.** The docstring says that with tau at the window P99, the hinge is 0 for every candidate at most decisions, so lex "plans on the tail when the tail is at stake and on the mean otherwise". The cause of the mismatch:
- Tail requests stay in flight for ~95 s, about 12 H-windows.
- Every decision's affected set overlaps them, so π_e's hinge is positive almost always.
- Lex compares hinge first, with only a 1e-9 absolute tolerance. It therefore deviates on any hinge improvement, however small, regardless of the sum.

**Evidence** (`out/tool_s075_w2000_h64/lookahead_lex_H64_true.jsonl`):
- π_e's hinge is > 0 at 76.8 % of the 1,658 decisions.
- 324 of 466 deviations are hinge-driven. Their hinge gain has a median of 0.0028 (≈ 0.27 s summed above tau), a p10 of 2.4e-4 and a minimum of 8.1e-8.
- 75.6 % of those hinge-driven deviations make the predicted sum worse.
- The rollout sum's spread across candidates has a median of 1.3 %.

Caveats:
- This run is the confounded one (policy memory bug). Its tails may be inflated, and that would inflate how often hinge > 0.
- On Conv, the first 109 decisions had hinge 0 everywhere, so the claim is trace-dependent.

The mechanism (overlap of long-lived tail requests) does not depend on the bug, so re-check this on the rerun.

**Suggested fix.**
- Give the hinge a meaningful tolerance: e.g. deviate on hinge only if the gain exceeds a fraction of tau, such as 1 s summed, or a relative threshold. Otherwise fall through to the sum.
- Alternatively, restrict the hinge to the rollout's own arrivals plus requests whose E2E is still decision-sensitive, or score a proper CVaR over the affected set.
- Correct the docstring and the notes either way.

Hashed: no.

### [major] `summary.json` caches π_e, the baselines, tau and cut without checking what they were computed for
`headroom.py:264-289`

**Failure scenario.** Cached entries are reused whenever their key exists in `<out>/summary.json`. The key is just `"pi_e"`, `"lor"` and so on. The scenario, policy file, `code_version` and bundle are not checked.
- Run `--trace Tool --out X`, then `--trace Conv --out X`, or rerun after the simulator or the `POLICY` constant changes.
- The second run takes tau (`summ["pi_e"]["p99_e2e"]`) and the warm-up cut (`summ["pi_e"]["cut"]`) from the first run.
- It plans with the wrong hinge threshold and the wrong warm-up boundary.
- `summ.update({"scenario": sc.tag, ...})` then relabels the stale π_e, LOR and Mooncake rows with the new scenario. The comparison table is silently wrong.

Also, `--max-decisions` is not part of the lookahead key, so a partial run overwrites or poses as a full one. `planned` does reveal it.

**Suggested fix.**
- Store `{scenario, policy sha, code_version, bundle}` in the summary and refuse (or recompute) on a mismatch.
- Include `max_decisions` in the key when it is set.

Hashed: no.

### [minor] "Sampled futures" still see true output lengths of in-flight requests; the pool and the cap are not as the trace filter
`headroom.py:20-22`, `:117-131`

**Failure scenario.** The docstring says that in sampled mode "the rollouts no longer see the true output lengths". Two problems:
- Only requests not yet arrived are resampled. The ~140 in-flight requests keep their true remaining decode lengths, and those dominate the drain-scored affected set.
- The resampling pool is the whole window, future included.

The cap also differs from the trace filter:
- It is the replica's KV capacity (483,328 tokens for l3_8b).
- The trace filter is `max_request_tokens(sc)` = min(KV capacity, 131,072 context).
- So a resampled request can exceed the 128K context the v4 traces deliberately exclude.

**Suggested fix.**
- Also resample the remaining output of in-flight requests (conditioned on the tokens already decoded), or state plainly that only future arrivals are de-hindsighted.
- Cap at `ev.max_request_tokens(sc)`.
- Draw the pool from the pre-window history or the warm-up only.

Hashed: no.

### [minor] Child failure handling: no timeout, uninformative crash error, zombies and fd leaks on the first error
`headroom.py:160-192`

**Failure scenario and evidence.**
- **A child killed by a signal** (jetsam/OOM, SIGKILL) gives an empty pipe. The parent raises `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` with no pid or exit status. Repro: `_fork_child(None, lambda: os.kill(os.getpid(), 9))` then `_collect`.
- **The first failing child of a batch** raises out of the list comprehension in `evaluate_candidates`. The remaining children are never read or waited on. Repro: the second child stayed in state `Z` (zombie), and its read fd leaks.
- **A hung child** (infinite loop, wedged replica) blocks `r.read()` forever, because there is no timeout.
- **The error payload** is `repr(e)` only, without a traceback.
- **Threads at fork time.** torch is imported (`set_seeds`) and the parent has 2 native threads. In practice this was fine over ≈13k forks, but it is a latent macOS fork-safety risk.

**Suggested fix.**
- Read with `select` and a deadline, and `os.kill` the child on timeout.
- Check the `os.waitpid` status and report pid, signal and exit code.
- Collect every child in a `try/finally` that reaps the rest.
- Send `traceback.format_exc()`.
- Optionally call `torch.set_num_threads(1)` before the loop.

Hashed: no.

### [minor] `score_output` lacks `run_one`'s completion gate and does not record the prediction bundle
`headroom.py:53-67`

**Failure scenario.**
- A lookahead run that strands requests would still be scored on the completed subset. `completed`/`n_expected` are recorded but not enforced, and the warm-up cut would then be computed from the completed requests.
- The lookahead and π_e entries carry no `bundle`/`clamp_hits`, unlike `run_one`'s output. Same-bundle comparability with `pi_e_harness` is therefore not recorded. I verified it once: same bundle `96bd5ae2116525e5`.

**Suggested fix.** Raise (or mark `ok: False`) when `completed != n_expected`, and add `bundle_fingerprint(predictor_info["bundle_key"])` and `clamp_hits`.

Hashed: no.

### [minor] Duplicate forcing code: switch to sim_fork's mechanism (concrete recommendation)
`headroom.py:81-114` vs `evolve/sim_fork.py:override_next_decision`

**Status.** `force_first`/`commit_policy_state` is exact for l3_48a0, verified byte-identical to `override_next_decision` above.

**Why it is weaker than sim_fork's version.**
- It checks only that `q[pick][-1] == n`. It does not verify that the call's change was exactly "trim, then append at pick" (`_fifo_move` does).
- It has no `on_forced` hook and no `policy_state="independent"` opt-in. A policy with any other state is refused, which is correct but limiting.
- Under a `route()` policy with bursts, `q[pick][-1]` would be the last burst member's entry, not the first request's, so the assert fails or a wrong entry moves. This does not apply with p1 and l3_48a0.
- It will drift from sim_fork as sim_fork is fixed.

**Recommendation.**
- Add an optional `record: dict | None = None` to `override_next_decision`. In `_forcing_call`, set `record["policy_choice"] = chosen`.
- Allow `choice_for_first=None` to mean "record only, do not override".
- Then `force_first(h, choice, rec)` becomes `override_next_decision(h, choice, record=rec)`, and `commit_policy_state` is deleted.
- The pending override lives in `gs.__dict__`, so it works unchanged in an `os.fork` child and in the parent right before the decision. sim_fork is not hashed.

Hashed: no.

### [minor] Tests check the implementation's own formula, not counterfactual exactness, and miss the deviation and scoring paths
`test_headroom.py`

**Gaps.**
- **`test_forced_decision_moves_policy_ledger_entry`** asserts the arithmetic `commit_policy_state` performs (pop at pick, append at c). It would still pass if that arithmetic were the wrong model of "same state as if π_e had chosen c".
- **The second half of `test_neutral_closed_loop_and_forced_rollout`** (forced onto π_e's own choice == unforced) is satisfied trivially: `commit_policy_state` returns early when `pick == choice`. It tests only that rollouts are deterministic.
- **The neutral test** uses `candidates="pe"`, so `closed_loop`'s deviation path is never exercised. That path covers the parent `force_first` with `best != pe`, the assert, and the lex/tie logic.
- **Nothing tests** that the affected set is complete, that `n` is equal across candidates, the sampled mode, or child-failure surfacing.

**Suggested fix.** Add three tests (all short on the Conv 4R w6000+200 scenario):
- At decision 120, compare `force_first(c)` against `override_next_decision(c)` over a full continuation (byte-identical CSV, as I ran: ~20 s per choice).
- Run `closed_loop(candidates="all", max_decisions=5)` and assert that it completes, that the parent assert holds, and that the outputs complete.
- Assert in `evaluate_candidates` that every child reports the same `policy_choice` and `n`. A one-line runtime check is also worth having in `closed_loop`.

Hashed: no.
