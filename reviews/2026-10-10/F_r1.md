# Chunk F, round 1: RL learner changes, offline diagnostics and logging

**Verdict: approve with minors** (one major: a gap in recording which run is which, harness-only).
The equivalence claims hold: defaults give byte-identical results to the pre-change learner, `log=True` gives the
same actions and gradient steps, and the store-once fix is correct. I checked all three by experiment, not only by reading the code.
The rl_offline headline conclusions survive re-derivation; two of the reported numbers need a caveat or a relabel.

Scope: `git diff 1823e99..HEAD` on the chunk's paths (af87600, 2610e67, 7fb6d38). Scratch scripts live in this session's scratchpad.
Targeted tests: `tests/test_learned_scheduler.py tests/test_rl_offline.py tests/test_rl_stats.py`: 24 passed.

## What I verified (no finding)

- **Defaults are byte-identical to before.** I ran the current learner and the pre-7fb6d38 learner (`git show 7fb6d38^:...`, injected
  in-process) on 4 designs: seed_ttft_bandit (outcome reward), seed_dqn_interval, seed_dqn_queue (interval reward)
  and seed_dqn_teacher (teacher_mix 0.5, imitation 1.0). Each ran 2 episodes with a checkpoint and a buffer reload. The digests (net tensors,
  buffer arrays, stats JSON minus timing) are identical in all 8 cases.
- **log=True only observes.** The same 4 designs x 2 episodes x {train_seed None, 3} give digest(log on) == digest(log off) in every case.
  The run used `log_interval=3` and `log_trace_frac=0.5` and reloaded the buffer and its `.meta.npz` sidecar. Reading the code agrees: the target-net forward pass for
  drift runs under `no_grad` on `Linear`/`ReLU` layers only (no dropout or batchnorm, no RNG). `clip_grad_norm_` is called exactly as
  before (only its return value is now kept). `_log_update` runs after `opt.step()` on detached tensors computed before the step. The new
  early return in `on_request_end` only adds an `Outcome` construction, which raises no exception and changes no state, before returning when `d is None`. The
  `t1` timer only exists inside the `try`, and any exception propagates before it is read.
- **The golden digest is not tautological.** `GOLDEN_OFF[2]` (3e5817...) is reproduced by the pre-change learner
  (7fb6d38^) and also by the v3 learner (1823e99). So it is a genuine pre-change reference.
- **Store-once is correct in every path.** Request end: `d` comes from `_decisions` and is popped on store. Next
  decision: `_maybe_store(p)`. Episode-end flush: `_maybe_store(self._prev)`. The guard catches the two double paths (a bandit whose request finishes
  before the next decision, and the last decision of the episode). DQN never double-stored. `test_each_decision_is_stored_once` does catch the regression:
  with the guard defeated by a pytest plugin that resets `d.stored = False`, it fails `71 == 42`.
- **train_seed works as described.** Arrivals are untouched because the simulator and trace seeds come from `sc.seed`, and `ctx.rng` of the design still follows
  the scenario seed. `torch.manual_seed(train_seed)` runs only for a fresh net (`_pending_ckpt is None`). With a loaded
  checkpoint, train_seed acts only through the learner and teacher `random.Random`.
- **`run_design`'s appended `CONFIG = dict(globals().get('CONFIG') or {}, **opts)` is safe.** It handles a missing CONFIG, `CONFIG = None`
  and any mapping. Every option is a valid DEFAULTS key, so `resolve_config` accepts it. `globals` is banned by `evolve/guard.py`, but the guard
  only checks the proposed source in the driver, never the run file. Scoring inference uses the original design, which is equivalent
  because the options matter only in training.
- **rl_stats still reads old runs the same way.** I re-ran it on every archive and diffed against the committed `out/rl_stats.json`. The only differences are
  the new `logging: null` and `design/config/{train_seed,log,...}` keys, the code_version strings and the
  "not available" text. On a logged two-episode dir, `rl_stats --train-dir --offline` and `rl_offline.analyse_dir` both run
  (traces source, who fractions, exact chunks).
- **rl_offline statistics are clean on the e413 Tool finalist.** These were re-derived. Folds hold out whole episodes, and dedup runs before the folds, so a
  duplicate pair (always within one episode) cannot cross folds. Held-out EV equals R^2 to within 0.004 on every fold: the mean
  residuals are small, so EV's tolerance of a mean offset does not flatter it. Fold variance is 0.97 to 1.04 x the pooled variance, so a
  pooled ceiling is comparable with fold EV. Dedup removes exactly the double-store count on d2cb (1320 = 1320 from the
  per-episode counters) and 6 extra rows on e413 (1743 vs 1737; negligible). Only 15 rows have duplicated feature vectors,
  so excluding each point from its own neighbour list in the Gamma test is reliable.

## Findings

### [major] The run manifest cannot identify the learner version
`evolve/rl_task.py:195` (`write_manifest`), and likewise `evaluate_design:360`.
- **Failure scenario.** `learned_global_scheduler.py` is in `_VERSION_EXCLUDE` precisely so that it can change without invalidating
  baselines. The manifest's run identity, however, is `code_version` plus the design sha256, with no hash of the learner file and no git commit.
  Two RL runs made with different learners therefore get identical manifests: for example, before and after the store-once fix, which changes bandit
  training (af87600), or after any future fix to the loss, the target or the logging. The spec calls the manifest "everything needed to identify the run".
- **Evidence.** `ev.provenance()` returns only code_version and library versions. `grep` finds no sha of the learner
  file anywhere in `rl_task.py`.
- **Fix.** Add `learner_sha256` (and the `evolved_global_scheduler.py` sha, or all `_VERSION_EXCLUDE` RL files) plus `git rev-parse HEAD`
  and a dirty flag, when available, to `write_manifest`. Ideally also add them to the `evaluate_design` result, so that archive rows carry them.
- **Hashed file:** no (harness only).

### [minor] train_seed=0 reuses the default learner and teacher RNG streams
`vidur/scheduler/global_scheduler/learned_global_scheduler.py:214`
- **Failure scenario.** `rseed = train_seed * 1_000_003 + gs.seed`, so `train_seed=0` gives `rseed = gs.seed`. That is exactly the
  `train_seed=None` stream for exploration, batch sampling and the teacher's `ctx.rng`. Seed 0 differs from the legacy run only in
  network initialisation (`manual_seed(0)` instead of `set_seeds(sc.seed)`). `train_seeds()` defaults to `(0, 1, 2, 3, 4)`, so a
  5-seed CI would share streams with any untagged default run. There is also an older overlap that persists: the teacher stream `rseed + 1` of the episode with scenario
  seed s equals the learner stream of the episode with seed s+1. Within one LOTO protocol these pairs exist, e.g. episodes p4 and p5.
- **Evidence.** Arithmetic. No finalist has been trained with train_seed yet (spec: "Not yet run for any finalist"), so no result is affected.
- **Fix.** For example, `rseed = (train_seed + 1) * 1_000_003 + gs.seed` and teacher `rseed + 500_001`, or
  `random.Random(f"{train_seed}/{gs.seed}/learner")`. Change it before the first multi-seed run.
- **Hashed file:** no (the learner is excluded).

### [minor] test_train_seed_seeds_the_learner_only does not test "the workload is unchanged" or the RNG part
`tests/test_learned_scheduler.py:291`
- **Failure scenario.** `len(set(decisions)) == 1` holds for any seed handling: `decisions` is the number of routed requests, which equals the
  trace length (the arrivals here come from a trace file, and `learning_starts=10**5` means no training). A bug that let train_seed change
  arrivals or perturbation seeds would still pass. The test also never checks that exploration and batch sampling follow train_seed (only initialisation, through
  `nets`). The golden test covers one design path (DQN + teacher, outcome reward, one episode, no buffer reload), and it does not
  distinguish the store-once fix: the v3 learner gives the same digest. My broader runs above cover those paths, but the suite does not.
- **Fix.** Compare `request_arrived_at` from the request metrics (train_seed 1 vs 2 vs None) and
  `explore_frac`/`action_share` (equal for the same seed, different across seeds). Optionally add the bandit design and a two-episode reload to the
  log-on == log-off digest check (about 10 s each).
- **Hashed file:** no.

### [minor] The Gamma-test noise ceiling ("~0.7") depends on the subsample size; the gap conclusion still holds
`evolve/analysis/rl_offline.py:169` (`n_max=20000`) and `:433` (`n_max=8000` within episode)
- **Failure scenario.** The ceiling is extrapolated linearly from delta in [4.9, ~6.5] (59 standardised dimensions) down to delta = 0. It moves with the
  number of rows used, so the reported 0.70 is a property of the 20k subsample, not of the data. The "within-episode" column uses
  8k rows per episode, so the pooled and within-episode ceilings mix sample sizes and cannot be compared with each other.
  The key `ceiling_ev_upper_from_gamma1` is misnamed: gamma(1) over-estimates the noise, so 1 - gamma1/var is a *lower* bound on the ceiling.
- **Evidence.** e413, target r_out, after dedup:

  | k \ n | 10k | 20k | 40k |
  |---|---|---|---|
  | 5 | 0.624 | 0.698 | 0.753 |
  | 10 | 0.646 | 0.697 | 0.721 |
  | 20 | 0.631 | 0.671 | 0.695 |
  | 1 - gamma1/var | 0.572 | 0.619 | 0.651 |

- **Conclusion impact.** "Held-out EV ~0.5 vs ceiling ~0.7" survives, and the gap grows: the ceiling's lower bound at 40k rows is already
  0.65, above the held-out 0.52 and the row-split 0.61. Write it as "ceiling >= 0.65 (lower bound), point estimate 0.70 to 0.75 depending on
  n and k", not "0.7".
- **Fix.** Use all rows (or report the n-curve), use the same n for the pooled and within-episode estimates, and rename the key to `ceiling_ev_lower_from_gamma1`.
- **Hashed file:** no.

### [minor] "Reward-model argmax = rule on 49%" is computed on r_out, which is not what the learner maximises
`evolve/analysis/rl_offline.py:481-494`
- **Failure scenario.** When the design has a price feature, `main_t = "r_out" = r + price[a]`. The bandit regressed `r = r_out - price[a]`, so its
  greedy choice is `argmax(r_hat)` or `argmax(r_out_hat - price)`, not `argmax(r_out_hat)`. The report calls it "Fitted reward model's argmax".
- **Evidence.** e413, width 16, same folds: r_out model 0.490 (as reported), r model 0.502, r_out model minus price 0.512.
  The conclusion (about half the decisions agree with the rule) does not change.
- **Fix.** Report the agreement on `r` (the learner's objective), or label the current number "outcome model (price excluded)".
- **Hashed file:** no.

### [minor] Width is selected on the test folds, and the data curve varies rows rather than episodes (interpretation)
`evolve/analysis/rl_offline.py:426`, `:497-500`
- **Width selection.** `best_w` is chosen by held-out-fold EV, and the ablation, beyond-rule and data-curve results are then reported on the same folds. This is a small
  optimistic bias: the widths differ by at most 0.02 on e413.
- **Data curve.** `cv_ev(frac=...)` subsamples rows uniformly from the same ~9 training episodes. So "EV flat from 10% to 100% rows" shows
  that more rows *of these episodes* do not help. It says nothing about more episodes or more workload diversity, which the held-out-episode gap
  (0.52 vs row-split 0.61) points to. No number changes, but this caveat belongs next to the "data is sufficient" claim.
- **Hashed file:** no.

### [minor] The train() docstring claims "off = exactly the v3 behaviour and outputs"
`evolve/rl_task.py:277`
- **Failure scenario.** With the options off, train() behaves exactly like the *pre-logging v4* learner. That is not v3: the store-once fix (af87600) changes
  bandit training. v3 bandit buffers held about 1.6 to 1.8% duplicated rows (1320 and 1737 on the finalists), and a repeat of a v3 bandit run gives different
  numbers. This is a false-equivalence statement of the kind the brief targets.
- **Fix.** Write "off = exactly the behaviour before the logging change (v4 learner; v3 bandits double-stored early completions)".
- **Hashed file:** no.

### [nit] The golden-test provenance names a commit that is not in this repo
`tests/test_learned_scheduler.py:212`: "HEAD 29b4dc9" is a hermes-memories (vault) commit, and `git cat-file` fails for it in vidur.
The digest reproduces with the learner from `7fb6d38^` (and from `1823e99`), so cite that instead.

### [nit] The driver does not validate --rl-log-trace-frac
`evolve/driver.py`: the flag is ignored without `--rl-log`. An out-of-range value (or `nan`, which `repr` writes into `design_run.py` as an undefined name)
makes every candidate fail at training time with fitness -1e9, which burns proposer budget, instead of failing at the command line. Check `0 < f <= 1` in `main()`.
