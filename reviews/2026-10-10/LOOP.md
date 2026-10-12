# Review loop log, 2026-10-10

Base `1823e99`, head at round 1 `7fb6d38`. Brief: `BRIEF.md`. Stop rule per chunk: a fresh reviewer round with no
blocking or major findings. Fixes touching `code_version`-hashed files need an explicit decision (running jobs and
cached baselines depend on `c16266a8ef49`).

| Chunk | Scope | Round 1 | Triage | Fixes | Round 2 |
|---|---|---|---|---|---|
| A | simulator v4 (`vidur/` minus learner) | changes requested: 4 major, 6 minor | all accepted; hashed-file fixes deferred to branch `v4.1-sim-fixes` (defaults unaffected, nothing running uses them); test-only fixes now | branch `v4.1-sim-fixes`: f16fc4a (tests; cherry-picked to main as a4fb66a), 559218e (vidur; branch code_version c8bb68b7636b; golden v3+v4 30/30 byte-identical) | approve with minors (A_r2; own golden v4 30/30). Agg-in-sim_fork and fork write_output already fixed on main (bc0b369); tracked .pyc untracked; view_refresh leaves the prefix probe live (document in paper); bundle_source sha unused by evaluate (after merge). **Done (branch ready for the next version bump).** |
| B | harness, traces, driver/bar_eval --bar | approve with minors: 2 major, 5 minor | majors accepted (table values folded into code_version only when they differ from today's, so c16266a8ef49 stays valid; behavioural test); minors accepted except bundle mid-run swap (noted) | 8dab76b, 16ffa55 | approve with minors (BD_r2); unhashable-table minor fixed; deferred: bundle-fingerprint parts 3-4, warmup_frac, old-run-dir budget count, nits. **Done.** |
| C | golden-set tool | changes requested: 2 major, 5 minor | all accepted; extended cells (preemption, route delay, view refresh, observe) as an opt-in set | 97e18c4 | approve with minors (C_r2); --force minor fixed 29ca349; nits noted. **Done.** |
| D | floor/tunable baselines, ES tuner, head_to_head, campaign launcher | changes requested: 1 blocking, 4 major, 5 minor | all accepted except #10 (p2c n=1; editing changes the baseline key); tuner #4-6 fixed and synced to Alpine (068fd0b) | 068fd0b, 8dab76b, 7cb131c, f7a8810, 4cc65a5 | r2: retry changed trajectory; r3: keep every record (kills leave none; Alpine KillWait 30 s, cgroup, no child signalling); r4 approve with minors, all fixed in 4cc65a5 and synced 06:18 before link 2. **Done.** Tuner file FROZEN on Alpine until both chains finish (meta.json records its sha). |
| E | sim_fork | changes requested: 5 major, 2 minor, 1 nit | all accepted; headroom agent warned (forced-decision policy memory confounds lookahead numbers) | bc0b369, 1ad201a, 5c36f7b, final | r2, r3 changes requested; r4 approve with minors, minors applied. **Done.** |
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
| G | rl_plan headroom pilot | r1-r7 changes requested (method); r8 approve with minors. **Done.** | | | |
- 2026-10-10 G round 1: changes requested (3 major: horizon too short to see window P99 effects; lex hinge-first score
  hinge-driven on Tool; stale summary reuse; 5 minor). All accepted. Horizon and score are pilot-methodology choices
  that also inform the MDP discussion (Q-list); the fixer implements a time horizon with arrivals continuing and an
  H sweep, keeps lex as an option, and switches forcing to sim_fork (record-only mode) after E round 2 settles.
- 2026-10-10 E round 3: changes requested (1 major: 'independent' check shallow -> allowlist by source sha + recursive guard; 3 minor: ctx-attribute state, module-state leak between forks, os.fork RNG helper). Fixer resumed. E_r2 issues all fixed.
- 2026-10-10 G round 2: changes requested (3 major: pure time cutoff blind to later arrivals -> T+drain default; summary key lacks headroom/sim_fork sha; len-600 sweep cannot separate horizons -> open-loop horizon selection; 2 minor). Fixer resumed.
- 2026-10-10 E round 4: approve with minors (l3_48a0 rollouts: forks byte-identical, forced = self-chosen, 104 real overrides raise nothing). Stop rule met. Final minor (alias false refusal) + docs being applied.
- 2026-10-10 G round 3: changes requested on method (drain bias; open-loop agreement inflated by truncation; no CIs/noise floor/placebo). Decision: headline = full horizon; proxies validated against it; stats added. Fixer resumed. a11 notes updated.
- 2026-10-10 New finding (from the G fixer): RequestView.prefix_hashes exposes the full block_hash_ids, whose length is
  (prefill + decode) // block_size in v4 traces, so routers could infer output length. Finalists (incl. l3_48a0) never
  read it; system baselines use it only for prefix matching (output ids are row-unique). Kamil chose to fix now as
  v4.1 (bundled with the approved A fixes): router-visible hashes cut to prompt blocks, table digest folded into
  code_version. Alpine v4 tuning/bars/gen70 jobs cancelled; v4 baselines superseded on merge. v4.1 build running on
  branch v4.1-sim-fixes; then a review round on the v4.1 delta, recompute baselines, restart tuning.
- 2026-10-10 G round 4: mechanics and statistics code correct; method changes requested (noise floor identically 0 because π_e tie-breaks occur only before warm-up; w=25m trades tail for mean; verdict pooling; block bootstrap; control code in log key). Fixer resumed; all must land before launch.
- 2026-10-10 G round 5: mechanics approved; rule-level changes requested (trade counted as negative; weight rule can pick a never-deviating weight; TPOT guard knife edge; null without effect size). Cross-fit real-vs-chaos statistic adopted. Fixer resumed.
- 2026-10-10 v4.1 built on branch (3cbf477 merge, 62b5ea5 leak fix, 27adc34 code_version fold -> cbeb5161f046 / v3 cead88d7d5d3, a2595a4, f9c1917 manifests): golden v3/v4 30/30 and v4ext 5/5 byte-identical; v4.1ext 8 cells reproduce; 240 tests pass. Legacy RL_CACHE_AWARE StateTracker feeds true decode length into RL state (oracle leak; legacy path only) - flagged. Review V41_r1 running before merge.
- 2026-10-10 G round 6: run code approved; verdict code matches docstring; 2 major post-processing rule fixes (no-headroom unreachable; effect bounds not data-based) + minors. Fixer resumed.
- 2026-10-10 v4.1 review (V41_r1): ready to merge, minors only. Merged as 4098ec5 (code_version cbeb5161f046, v3-data
  cead88d7d5d3). Minors fixed in 6324d69: sim_fork observe pass-through; legacy rl_cache_aware refuses without
  VIDUR_ALLOW_ORACLE_RL=1 (oracle decode-length features incl. replica*_pending_decode_*; historical results from it
  were oracle-informed); static candidate screen in evaluate_policy (all 306 existing policies pass). **Done.**
  Mac BAR_V4 baselines recomputing (search3b, heldout3); Alpine tuning chains t41_* 33743012-19 and golden_v41
  33743020 submitted.
- 2026-10-10 G round 7: runs may launch; 1 major post-processing fix (pairwise noise reference: coverage 88% -> 95%, true-null 'no headroom' 51% -> 77%, false 'headroom' 0 in ~2000 null pilots) + minors. Fixer resumed.
- 2026-10-10 G round-7 fix e153a6c (pairwise noise reference). Headroom pilot launched on Alpine on v4.1: 4 chains x 8 links (10 CPUs, 4 h), windows Tool/Conv x start 0/2000, out evolve/rl_plan/out/v41_*; last jobs 33743599/607/615/623. Round 8 (strict) running on post-processing.
- 2026-10-10 G round 8: approve with minors (stop rule met). Null-rate statements aligned to ~70 %. Optional pooled-null refinement and 'single negative window' caveat noted for the write-up. ALL CHUNKS DONE.
- 2026-10-11 Step B (headroom planner with sampled lengths, K=4): review B_step_r1 approve with minors; minors fixed (1183084); launched on Alpine.
- 2026-10-11 Step B2 (lcb deviation rule, K=8, true-future audit; Kamil approved B2): B ran with the mean rule and overrode pi_e on 77-81 % of decisions (worse than pi_e). Review B2_r1: changes requested (2 major, design: power/interpretability, skew); r2: 1 major (audit reading at the noise level) -> audit analysis pre-registered as code (audit_stats, --mode audit); r3: approve with minors (stop rule met); r3 minors applied before launch (workers 18, three readings, block bootstrap, audit tied to the gated row, predicted precision labelled an upper bound). 33 tests pass. Launched on Alpine in a separate tree vidur-rl-b2.
