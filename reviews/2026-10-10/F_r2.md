# Chunk F, round 2: RL learner, rl_task, rl_offline, rl_stats, logging

**Verdict: approve with minors.** All round-1 findings are fixed. I checked each one against the new code.
The three central claims hold in my own experiment, which ran outside the fixer's tests in a temporary copy of the tree. Defaults are byte-identical to the af87600 learner; `log=True` changes nothing; `train_seed` streams are distinct and leave arrivals unchanged.
New findings: 2 minor and 3 nits. All are harness or doc issues, and none touches a `code_version`-hashed file.

Scope: `git show 7435890` (chunk paths) plus the driver check in 8dab76b. Scratch scripts are in this session's scratchpad (`eqv.py`, `golden_bandit.py`,
`compare.py`), and the temporary trees are in `scratchpad/tree{A,B,C}`. The main checkout was not modified. My own git calls ran with `GIT_OPTIONAL_LOCKS=0`.
Targeted tests: `tests/test_learned_scheduler.py tests/test_rl_offline.py tests/test_rl_stats.py`: **29 passed** (129 s).

## Independent re-verification of the central claims

Method: I made two copies of the tree with `git archive HEAD` (vidur, tests, rl_designs, policies, finalists), with `data/` and `cache/` symlinked.
Tree A uses the HEAD learner (sha256 9f1b257a..., identical to the working tree). Tree B is HEAD with `learned_global_scheduler.py` from **af87600**.
Tree C is HEAD with the learner from 1823e99 (v3). Each run follows the `vidur/main.py` order (run, `write_output`, `save_final_checkpoint`) and uses 3
replicas on the conv-session slice (42 requests). It runs 2 episodes (scenario seeds 7 then 8), and the second reloads the first episode's network and replay buffer.
I used three designs: seed_ttft_bandit (outcome reward), seed_dqn_teacher (teacher `lor.py`, which draws from `ctx.rng`, teacher_mix 0.5)
and seed_dqn_interval (interval reward). All three use learning_starts 8, batch 8 and target_update 5, and every ep2 made 19 gradient steps. The digest covers the net
tensors, the **optimizer state**, every replay-buffer array and the stats JSON minus timing/logging keys. Each run also produces
a separate digest of `request_metrics.csv`, of the arrival columns and of the stored actions.

| Check | Result |
|---|---|
| Defaults: tree A vs tree B (af87600), 3 designs x 2 episodes, with and without arrival jitter | **identical** in all 12 episode digests (ck, request_metrics, arrivals, actions, explore/teacher fractions, action shares) |
| `log=True` (log_interval 3, log_trace_frac 0.5) vs off, train_seed None and train_seed 3, 3 designs x 2 episodes | **identical** in all 12; on writes `.updates.npz`, `.trace.npz`, `.buffer.meta.npz` and a `logging` block; off writes none of them |
| train_seed None / 0 / 1 / 1 again / 2, with arrival jitter on (U[0,3) s), epsilon 0.5 | arrivals identical across all 5 in every design and episode; the jitter check has teeth (the jittered arrivals differ from the unjittered ones); ts1 == ts1 (repeat) byte for byte; None, 0, 1 and 2 give 4/4 distinct action sequences and 4/4 distinct checkpoints in every design and episode, so **train_seed 0 is no longer the legacy stream** |
| Bandit golden digests (`GOLDEN_BANDIT`) | the af87600 learner reproduces both (6af7f650..., 8b900b57...; buffer 42/84 rows = decisions). The v3 learner (tree C) gives e280e046.../f840bf5f... with **71/142 buffer rows for 42/84 decisions**, so the test does pin the store-once fix as the spec says |
| `rng_seeds` | (None, s) -> (s, s+1) unchanged; 800 seeds for train_seed 0..49 x 8 scenario seeds are all distinct and all >= 2**64 (so never a legacy seed). The teacher context is wired to `tseed` (`learned_global_scheduler.py:239`), and `PolicyContext` uses the `random.Random` as is |

## Round-1 findings: status

| Round 1 | Status | How verified |
|---|---|---|
| [major] the manifest cannot identify the learner | **fixed** | `learner_provenance()` gives the learner sha256 (matches the file), the `rl_task.py` sha, `git_commit` 7435890 and `git_dirty` True (the checkout has a modified `evolve/sim_fork.py`), with `git_learner_dirty` False. With `PATH=/nonexistent` (no git), the git fields are None, the shas are still present, and no exception is raised. `train()` returns it on success and on failure. The driver stores `result=res` → `res["training"]["learner_provenance"]`, so archive rows carry it for default runs as well. `rl_stats` reads the new keys, and old manifests show `?`. `evolved_global_scheduler.py` is not in `_VERSION_EXCLUDE`, so code_version already covers it. |
| [minor] train_seed=0 reuses the default streams | **fixed** | hashed seeds with separate tags; experiment above (z0 ≠ None) |
| [minor] train_seed test too weak | **fixed** | the new test jitters arrivals, compares request ids, arrival times and `who` flags, and has a check with teeth (generator seed 43). The bandit two-episode golden test and the log-on == log-off test were added. |
| [minor] Gamma ceiling depends on n; misnamed key | **fixed** | `ceiling_ev_lower_from_gamma1`, plus `ceiling_lower()` to read old files: rendering the committed old `out/rl_offline.json` through the new `to_markdown` works, and old files show the lower bound from the old key. `noise_n_curve` and `rows_per_episode_mean` are written, and the doc and Markdown caveats are in place. |
| [minor] argmax agreement on r_out only | **fixed** | agreement is reported for both targets. `reward_model_argmax_eq_teacher` is still the main target with the same fit and seed, so old readers get the same number. |
| [minor] width selection and data curve | **fixed (documented)** | `width_selection` block, plus a note in the data curve and in the Markdown |
| [minor] docstring "exactly v3" | **fixed** | `rl_task.py` train() docstring |
| [nit] golden provenance commit | **fixed** | the comment now cites af87600 = 7fb6d38^ |
| [nit] `--rl-log-trace-frac` unvalidated | **fixed** | `driver.py:386`: `not 0 < f <= 1` also rejects nan |

**Correction to round 1.** Round 1's example of the legacy s / s+1 overlap ("episodes p4 and p5") was wrong. The protocol seeds are 401, 501, ..., 901, so no
two episodes differ by 1, and the quirk cannot occur in any current protocol (see the nit below).

## New findings

### [minor] `learner_provenance()` runs `git status`, which rewrites `.git/index` in the shared checkout
`evolve/rl_task.py:163-190` (`_git`, `learner_provenance`); also exercised by `tests/test_learned_scheduler.py::test_manifest_identifies_the_learner`.
- **Failure scenario.** `git status` refreshes the index as a side effect: it takes `.git/index.lock` and rewrites the index whenever stat data is stale.
  `train()` now calls this once per candidate, whatever the options, plus twice more per `write_manifest`. The driver runs candidates in parallel threads (`driver.py:225, 261`)
  from the same checkout on the Mac, where commits are made while jobs run. A `git commit`/`add` that coincides with the lock fails with
  "Unable to create '.git/index.lock': File exists". The collision window is short, so this is rare, but it is a real cross-process side effect of a function
  whose job is only to observe. The provenance itself stays correct, because git skips the optional refresh when the lock is held.
- **Evidence.** Scratch repo with git 2.50.1: after `touch f`, the mtime of `.git/index` changes after `git status --porcelain --untracked-files=no` (1791635829 → 1791635830).
  With `GIT_OPTIONAL_LOCKS=0` it does not change.
- **Fix.** Pass `env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}` in `_git` (or `git --no-optional-locks`).
- **Hashed file:** no.

### [minor] The recorded learner is the one on disk when `train()` starts, not the one each episode subprocess imported
`evolve/rl_task.py:327` (`learner = learner_provenance()`), and `write_manifest` recomputing it after episode 0.
- **Failure scenario.** Each training episode is a new simulator process that imports `learned_global_scheduler.py` at that moment. A long RL
  run (8–12 episodes, hours) launched from the working checkout picks up any edit to the learner made mid-run. This is the workflow that produced
  7435890 while other jobs were running. The result row then shows the learner sha from the start. The manifest (rewritten after episode 0) can show a
  different sha, and episodes after the edit used a third learner. Nothing flags the mix. The same applies to `git_commit`.
- **Evidence.** By reading the code: `learner_provenance()` reads the file at call time in the harness process, and the learner never records its own file hash.
- **Fix (cheap, harness-only).** Recompute `learner_provenance()["learner_sha256"]` after every episode, store it in the episode row, and set
  `learner_changed_mid_run` when the values differ. Better: have the learner put `hashlib.sha256(Path(__file__).read_bytes())` into its stats JSON under
  `log=True` (additive, inside the excluded file). That would also need `_digest` to drop it, since the golden digests hash the stats JSON.
- **Hashed file:** no (the learner is excluded).

### [nit] Stale wording on episode streams: shared scenario seeds, and a legacy-quirk example that cannot occur
`learned_global_scheduler.py:147-157` (`rng_seeds` docstring) and `RL_LOGGING_SPEC.md` ("train_seed ... done" row).
- The spec says that with train_seed the streams differ "across episodes", with no qualifier. Every protocol, however, reuses scenario seeds: v2 pairs 401/401,
  501/501, 601/601 and 701/701, and the loto protocols have triples at 601 and 701. Episodes with the same scenario seed therefore start the **same** learner and teacher streams,
  with or without train_seed, because the hash key is (tag, train_seed, scenario seed). Draws desynchronise quickly: epsilon continues across episodes, and
  `randrange` consumes bits that depend on the buffer length. So this is a correlation, not repeated decisions. The docstring qualifies it correctly ("with different scenario
  seeds"); the spec does not.
- The docstring's example for the legacy s/s+1 quirk, "(e.g. two episodes of one LOTO protocol)", is false: there is no s/s+1 pair among the protocol seeds. This came from round 1's mistaken
  example.
- **Fix.** Reword the spec row and drop the example. Optionally (train_seed runs only, so legacy stays byte-identical), add the loaded decision count or
  the trace path to the hash key so that every episode of a chain gets its own stream.
- **Hashed file:** no.

### [nit] `git_commit` can name the wrong repository
`rl_task.py:180`. `git -C REPO rev-parse HEAD` succeeds inside *any* enclosing repository. Two cases: (a) an rsync'd tree without `.git` that sits under a
parent git repo records the parent's HEAD. In a scratch demo, a non-git child directory reported the parent's commit. (b) A git clone that is later
overwritten by rsync (the `launch_v4_campaign.sh` pattern, if the Alpine dir was ever cloned) records an old commit with `git_dirty` True. I could not check the
Alpine dir (no Alpine access in this review). `learner_sha256` stays authoritative in both cases. Fix: compare `git rev-parse --show-toplevel` with
`REPO`, and record None otherwise. **Hashed file:** no.

### [nit] The committed `evolve/analysis/out/rl_offline.{json,md}` still use the old key and label
The fixes are in the code, but the artifacts were not regenerated. The `.md` still says "Fitted reward model's argmax = rule's action on 0.490" (the r_out model) and has
no lower bound or n-curve. Anyone who quotes the committed `.md` gets the round-1 wording. Re-run `python -m evolve.analysis.rl_offline` when the Mac is
free, or add a one-line header to the `.md` pointing to round 1. **Hashed file:** no.

## Other new code checked (no finding)
- `rng_seeds`: deterministic across processes (sha256, not `hash()`). The 65-bit seeds are accepted by `random.Random`. `train_seed` validation still rejects
  bool, negative and non-int values. `torch.manual_seed` runs only for a fresh net, so this is unchanged.
- `rl_offline`: in the argmax loop for target r, the regressor is fitted on `r` with the price feature among its inputs, so `argmax r_hat` is the bandit's greedy rule.
  The main-target value is computed exactly as before. The `noise_n_curve` row counts are capped at T, and light mode skips them. `width_selection` is fine
  (heuristic bound; it excludes linear from the spread, which matters only if linear were ever selected, and it is not on either finalist).
- `rl_stats`: the manifest reader and `write_train_dirs_md` tolerate missing keys and manifest parse errors. `tests/test_rl_stats.py` passes.
- Test `test_analyse_data_end_to_end_on_a_synthetic_buffer` runs the full non-light path, including the Markdown output, so the new keys are covered.
