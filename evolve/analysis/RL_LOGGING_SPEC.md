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
