# RL logging specification (plan E0, 2026-10-10)

What every future RL training run must record so that `evolve.analysis.rl_stats` and `evolve.analysis.rl_offline`
can answer, without re-running anything: does learning converge, is capacity sufficient, is the amount of data
sufficient, how noisy is the target, and does the policy differ from (and beat) its teacher.

**Where each item lives.** `[v]` = needs an edit in `vidur/` (the learner,
`vidur/scheduler/global_scheduler/learned_global_scheduler.py`). These go into the **v4 release** (plan A4/A9) and
must be behaviour-preserving when off (golden set byte-identical). `[h]` = harness only (`evolve/rl_task.py`,
`evolve/driver.py`, `evolve/analysis/*`); no `code_version` change.

## 0. Run identity and seeds

| Item | Where | Notes |
|---|---|---|
| `train_seed` separate from the scenario seed | [v]+[h] | Today `vidur/main.py` calls `set_seeds(config.seed)`, so torch initialisation, exploration and the teacher RNG all follow the *scenario* seed: a new training seed changes the arrivals. Add `learned_global_scheduler_config_train_seed` (default `None` = current behaviour); when set, the learner calls `torch.manual_seed(train_seed)` before `_init_net` and seeds its `random.Random` from it. The harness passes it per run. |
| ≥ 5 training seeds for every finalist and every key ablation | [h] | `rl_task.train(..., train_seed=s)`; directory `training/<id>/seed<s>/`. Report mean ± CI across seeds, never the best seed. |
| Run manifest | [h] | `training/<id>/manifest.json`: code_version, design sha256, CONFIG after `resolve_config`, teacher path + sha, episode list (tags, seeds), train_seed, host, torch/numpy versions, prediction-bundle key. |

## 1. Per gradient update (aggregated per logging interval, e.g. every 200 updates)

Stored as a compact array file `net_ep<k>.pt.updates.npz` (one row per interval) rather than in the stats JSON.

| Item | Where | Why |
|---|---|---|
| Loss (mean, p95) split into TD/regression part and imitation part | [v] | The current `loss` summary mixes the DQfD margin term with the TD term. |
| Gradient norm before clipping (`clip_grad_norm_` return value, discarded today) and fraction of updates clipped | [v] | Divergence / clip-limited learning. |
| TD error distribution: mean, sd, p5/p50/p95 of `y - Q(s,a)` | [v] | Convergence: shrinking sd at a stable mean. |
| Q spread: mean over batch of std across replicas, and best-vs-next gap | [v] | Signal for action choice; compare with the TD-error sd (signal-to-noise; today best-vs-next ≈ 0.6 × fit error). |
| Target drift (DQN): mean \|Q_target(s') change\| at each target sync | [v] | Moving-target instability. |
| Value-head explained variance on the batch | [v] | Cheap online fit-quality signal. |
| Policy entropy and KL(old‖new) per update (PPO / softmax policies) | [v] | Standard PPO health; collapse detection. |
| Replay age of sampled transitions (mean decisions since stored) and buffer fill | [v] | Off-policyness; verifies the A9 double-store fix (each decision stored once). |
| Learning rate, ε, teacher_mix, imitation weight at the interval | [v] | Schedules actually applied. |

## 2. Per episode (extends today's `net_ep<k>.pt.stats.json`)

| Item | Where | Notes |
|---|---|---|
| Everything logged today (decisions, steps, ε, explore/teacher fractions, teacher agreement, action share, reward/loss/q summaries, train/policy seconds, buffer sizes) | [v] exists | Keep the keys stable; `rl_stats` reads them. |
| Episode tag, scenario seed, train_seed | [h] | Already in the archive row for the tag; add seeds. |
| Buffer index range of this episode (`buffer_start`, `buffer_end`) | [v] | `rl_offline` currently infers it from `buffer_size_end`, which breaks once the buffer wraps. |
| Held-out greedy evaluation of this checkpoint on 2–3 fixed **unseen** windows (P99 E2E/TTFT/TPOT, TBT) | [h] | Learning curves on held-out draws (plan E3). Cost: ~3 short sims per episode. |
| Greedy agreement with the teacher on a fixed probe set of 5k saved states | [h] | Imitation index over training, comparable across runs (`rl_offline` probe set). |
| Fraction of greedy actions that changed vs the previous checkpoint on the probe set | [h] | Policy churn; convergence of the *policy*, not only the loss. |

## 3. Per decision (raw traces; sampled to keep size bounded)

| Item | Where | Notes |
|---|---|---|
| Persist the **whole** replay buffer at every episode end, not only the last (or keep a 10 % row sample of each episode) | [v]+[h] | `rl_task.train` deletes the previous buffer today. Offline refits need every episode's rows. |
| Per transition: features of all replicas, action, reward, the reward's components (e.g. TTFT/E2E/TPOT terms and the deterministic price subtracted), teacher action, who acted (greedy / ε / teacher), candidate mask, routing time, request id | [v] | The "who acted" flag removes the confound in `rl_offline`'s beyond-rule test (outcomes of ε-random actions are unbiased counterfactuals). The reward components let the outcome part be separated exactly (today only via the clipped price feature). |
| Inference decision latency (p50/p99 µs per decision) and features time | [v] | Router overhead metric (readiness report). |
| Per-request outcome joined to the decision (TTFT, E2E, TPOT, replica) | [v] | Already used for rewards; write it with the buffer for credit-assignment studies. |

## 4. What rl_stats / rl_offline compute from this

- Convergence: TD-error sd and Q spread trends, gradient-norm stability, policy churn → 0, held-out learning curve flat.
- Capacity: `rl_offline` width curve (train vs held-out-episode EV) against the Gamma-test noise ceiling.
- Data sufficiency: `rl_offline` row-fraction curve; plus held-out learning curve vs episodes.
- Beyond the teacher: EV gain of all features over state + is-teacher-action, using only ε-explored rows once the "who acted" flag exists; greedy-vs-teacher agreement.
- Seeds: every number above as mean ± bootstrap CI over ≥ 5 train_seeds.

## 5. Acceptance

- With all new options off, the golden set is byte-identical to v3 (plan §2 procedure).
- A smoke run (`driver --scenarios smoke`) produces `manifest.json`, `updates.npz`, per-episode buffers, and
  `rl_stats` / `rl_offline` read them without warnings.

## Implementation status (2026-10-10)

**How options reach the learner (deviation from the `[v]` notes above).** `vidur/main.py`, `vidur/config/config.py`
and the four hashed harness functions are untouched, so `code_version` stays `c16266a8ef49`. New learner options are
CONFIG keys validated by `resolve_config` (`learned_global_scheduler.py` is in `_VERSION_EXCLUDE`): `train_seed`
(None), `log` (False), `log_interval` (200 updates per row), `log_trace_frac` (1.0). `rl_task.train(..., train_seed=,
log=)` passes them by writing `<work_dir>/design_run.py` = the design + one appended `CONFIG = dict(CONFIG, ...)`
line; the original design file and hash are untouched and both hashes go into the manifest. With every option off,
`train` runs the original design file exactly as before (no new file, no new key).

| Item | Status | Where |
|---|---|---|
| `train_seed` | done, learner-local | `_init_net` calls `torch.manual_seed(train_seed)` before building a fresh net; the learner RNG (exploration, batch sampling) and the teacher's `ctx.rng` use `train_seed * 1000003 + scenario seed` (episodes keep distinct streams). The design's own `ctx.rng` and the simulator still follow the scenario seed, so arrivals are unchanged (smoke check: identical `request_arrived_at`). |
| ≥ 5 seeds | hook | `rl_task.train_seeds(design, dir, seeds)` -> `dir/seed<s>/`, sequential. Not yet run for any finalist. |
| Manifest | done | `<work_dir>/manifest.json` when any option is on (`rl_task.write_manifest`), bundle added after episode 0. |
| §1 per-update | done except entropy/KL | `<ckpt>.updates.npz`: loss total/TD/imitation (mean, p95), grad norm (mean, p95, max) and clip fraction, TD error mean/sd/p5/p50/p95, Q mean, target mean, Q spread (std across replicas) and top-2 gap, batch EV, target syncs and drift, replay age, buffer fill, lr, ε, teacher_mix, imitation weight. Entropy/KL: n/a (no PPO/softmax policy in the learner). |
| §2 per-episode | done | stats JSON gets an additive `logging` block (only with `log=True`): `buffer_start`, `buffer_rows_added`, `buffer_end` (slots `(start + j) % capacity`, exact after a wrap), who-acted counts, decision/features/act latency (p50/p99/mean/max µs), file row counts; `train_seed` key when set. Episode tag, scenario seed and train seed are in the manifest and the returned log rows. |
| Held-out greedy eval per checkpoint | hook, off by default | `train(heldout=[Scenario...])` -> `<ckpt>.heldout.json` (P99 E2E/TTFT/TPOT/TBT, mean E2E). One simulation per scenario and episode; tested on one 200-request window. |
| Probe agreement and churn | hook, off by default | `make_probe_set(traces, out, n=5000)` once, then `train(probe=file)` -> `<ckpt>.probe.json` (greedy vs teacher agreement, fraction of greedy actions changed vs the previous checkpoint). No simulation. |
| §3 per decision | done (sampled) | `<ckpt>.trace.npz` per episode instead of persisting the whole buffer each episode: x, a, r, nx, γ, teacher action, `who` (0 greedy / 1 ε / 2 teacher), candidate mask, next mask, request id, buffer slot, store id, t, dt, `r_outcome`, `r_interval`, decision latency; plus every finished request's outcome (`out_*`: TTFT, E2E, TPOT, replica, tokens, cached, routed_at). Row sample `log_trace_frac`, deterministic in the request id (no RNG draw). `keep_buffers=True` keeps every episode's buffer as well. |
| Reward components | partial | Outcome and interval rewards are logged separately, with the raw outcome. A design's internal terms (e.g. e413's TTFT/E2E/TPOT terms and price) are not visible to the learner; the price part is still recoverable from the price feature. A design hook (e.g. `reward_components(outcome)`) is left for later. |
| Readers | done | `rl_stats`: logging fields per episode, `candidate['logging']` (updates/probe/held-out summaries), exact chunks from logged ranges, `--train-dir` for standalone dirs. `rl_offline`: reads traces when every episode has one (else the buffer, labelled by logged ranges, else `buffer_size_end` as before); beyond-rule test on ε rows only when the flag exists (≥ 100 rows); `--train-dir`, `--light`. Old runs: `rl_stats` output identical apart from the new `logging: null` key and the updated "not available" texts; `rl_offline` gives the same T / dedup / folds on both finalists. `driver --task rl --rl-log [--rl-train-seed s]`. |

**Verification.** (1) Smoke training (`rl_task.train(..., episodes=SMOKE_EPISODES)`, two designs: e413 bandit with
teacher and seed_dqn_teacher, small `learning_starts`), before vs after the change with options off: checkpoint
net and optimizer tensors `torch.equal`, buffer arrays equal, stats JSON equal on every key except the wall-clock
`train_seconds`/`policy_seconds`, `request_metrics.csv` and `token_gaps.npz` byte-identical (the pre-change
reference was itself deterministic on a rerun). (2) Same with `log=True`: identical, only the additive `logging` key
and new files. (3) `tests/test_learned_scheduler.py::test_logging_off_matches_golden_and_logging_is_observe_only`
checks a digest of a tiny DQN+teacher run against the pre-change learner (torch 2.10.0 / numpy 1.26.4; skipped on
other builds) and log on == log off. (4) `train_seed`: same seed twice identical, a different seed changes the
network, arrivals unchanged.

**Left.** Run the ≥ 5-seed protocol for finalists; decide the probe set and held-out windows for real runs (cost:
about 3 short simulations per episode); a design-level reward-components hook; entropy/KL once a softmax/PPO learner
exists; a full `driver --scenarios smoke --task rl --rl-log` run (needs a proposer; the training path it calls was
run directly).
