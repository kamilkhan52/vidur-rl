# Review loop log, 2026-10-10

Base `1823e99`, head at round 1 `7fb6d38`. Brief: `BRIEF.md`. Stop rule per chunk: a fresh reviewer round with no
blocking or major findings. Fixes touching `code_version`-hashed files need an explicit decision (running jobs and
cached baselines depend on `c16266a8ef49`).

| Chunk | Scope | Round 1 | Triage | Fixes | Round 2 |
|---|---|---|---|---|---|
| A | simulator v4 (`vidur/` minus learner) | changes requested: 4 major, 6 minor | all accepted; hashed-file fixes deferred to branch `v4.1-sim-fixes` (defaults unaffected, nothing running uses them); test-only fixes now | branch `v4.1-sim-fixes`: f16fc4a (tests; cherry-picked to main as a4fb66a), 559218e (vidur; branch code_version c8bb68b7636b; golden v3+v4 30/30 byte-identical) | round 2 running (A_r2) |
| B | harness, traces, driver/bar_eval --bar | approve with minors: 2 major, 5 minor | majors accepted (table values folded into code_version only when they differ from today's, so c16266a8ef49 stays valid; behavioural test); minors accepted except bundle mid-run swap (noted) | 8dab76b, 16ffa55 | approve with minors (BD_r2); unhashable-table minor fixed; deferred: bundle-fingerprint parts 3-4, warmup_frac, old-run-dir budget count, nits. **Done.** |
| C | golden-set tool | changes requested: 2 major, 5 minor | all accepted; extended cells (preemption, route delay, view refresh, observe) as an opt-in set | 97e18c4 | approve with minors (C_r2); --force minor fixed 29ca349; nits noted. **Done.** |
| D | floor/tunable baselines, ES tuner, head_to_head, campaign launcher | changes requested: 1 blocking, 4 major, 5 minor | all accepted except #10 (p2c n=1; editing changes the baseline key); tuner #4-6 fixed and synced to Alpine (068fd0b) | 068fd0b, 8dab76b, 7cb131c | round 2: 1 new major (tuner retry changed trajectory) fixed in 7cb131c and synced to Alpine; round 3 running (D_r3) |
| E | sim_fork | changes requested: 5 major, 2 minor, 1 nit | all accepted; headroom agent warned (forced-decision policy memory confounds lookahead numbers) | bc0b369 | round 2 running (E_r2) |
| F | RL learner, rl_task, rl_offline, rl_stats, logging | approve with minors: 1 major, 6 minor, 2 nit | all accepted; train_seed=None behaviour kept (byte-identity), quirk documented | 7435890, 2ab8918 | approve with minors (F_r2); git-lock and mid-run learner-edit minors fixed; nits noted (rl_offline outputs not regenerated). **Done.** |

Not yet reviewed: `evolve/rl_plan/` (headroom pilot; agent still running, uncommitted).

## Decisions
- 2026-10-10 A: simulator fixes all touch hashed `vidur/` files. Running jobs (Alpine tuning, Mac heldout3 baselines)
  and cached search3b baselines are keyed to `c16266a8ef49`, and no current or planned v4 job uses the broken paths
  (stuck-request crash only on failed runs; route_delay_s / view_refresh_s default 0). Fix on branch
  `v4.1-sim-fixes`, verify with the golden set (defaults byte-identical), merge at the next planned version bump.
- 2026-10-10 G (new chunk): `evolve/rl_plan/` headroom pilot (uncommitted). Its Tool H=64 run is confounded by E #4
  (π_e's attended-context memory not updated on forced decisions); Conv run stopped. The agent added
  `commit_policy_state` in headroom.py with a test. Review G after the sim_fork fixer finishes (headroom keeps its own
  force code; unify with sim_fork's mechanism if possible), then rerun Tool and Conv at H=64.
| G | rl_plan headroom pilot (uncommitted) | round 1 running (G_r1) | | | |
