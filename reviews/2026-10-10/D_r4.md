# Chunk D, round 4: tuner resume logic (fix f7a8810)

**Verdict: approve with minors (1 major in the tests only, 3 minor, 2 nits).** D_r3's major and both minors are fixed in the code. 180 of 180 randomized chains (link 1 on 1f9fde4, then HEAD links killed at random points, with torn lines and missing newlines, then two post-finish links) match the uninterrupted run exactly. In those chains a post-finish link makes 0 evaluations and leaves `best.json` byte-identical.
**The premise holds.** At the time limit, Slurm cannot in practice make the tuner write a failed record. The repurposed end-to-end test does not exercise the fix, though: it also passes on 7cb131c. The trajectory refusal fires only after the whole budget has been spent. The `tuner_sha` refusal is safe, but a comment-only edit trips it, and its message does not say what differs or what to do.
No finding touches a `code_version`-hashed file. `tests/test_tune_baselines.py`: 7 passed. `code_version` is `c16266a8ef49`.

Repro scripts are in this session's scratchpad, `r4/` (`harness.py` is D_r3's stub harness; `tb_old.py` = `git show 1f9fde4:...`, checked identical):
- `s1.py`: randomized chains.
- `s2.py`: SIGTERM race.
- `s3.py`: wrong seed.
- `s4.py`: forced duplicates.
- `s5.py`: tuner edit.
- `test_disc*.py`: the tests run against 7cb131c.

## Requested checks

### 1. D_r3 findings
- **M1 (resume retried genuine failures): fixed.**
  - `s1.py`, 60 chains for each failure model (transient = fails on its first attempt only, permanent, none), budgets 125/25/29/31/40, `--parallel 2`. Link 1 runs on 1f9fde4 and is killed at a random point. HEAD links are killed at random points until done. After each kill the file is damaged at random (torn last line, missing final newline, or clean). Two post-finish links follow.
    `transient exact: {True: 60}`, `permanent exact: {True: 60}`, `none exact: {True: 60}`. Exact means:
    - same keys, valid set and best as the uninterrupted run;
    - exactly `budget` history lines;
    - the post-finish links make 0 new calls, and `best.json` is byte-identical.
  - D_r3's `s4.py`, rerun on HEAD (the final generation's best candidate failed once, then a post-finish link): `new evals 0, best before -0.0000 after -0.0000, history lines 125`. Same for mooncake 125, dynamo 25 and mooncake 29.
- **m1 (missing final newline): fixed.** Covered by the `nonl` damage in `s1.py` and by the new unit test, which does discriminate.
- **m2 (no trajectory check; meta lacks template and ES): fixed, but late.** See minor m-r4-1 for the timing. `template_sha` and `tuner_sha` are in `meta`; the tuner sha covers `FAMILIES`, `ask_tell`, `params_of` and `lam_of`.
- **N1: fixed.**
  - `meta.json` and `best.json` are written atomically.
  - `best` is computed over valid records only, and "no valid evaluation" is refused.
  - The install header counts distinct keys.
- **Side effect, good.** A dir touched by 068fd0b or 7cb131c code has a `meta.json` without `template_sha`/`tuner_sha`, so f7a8810 refuses it. The D_r3 precondition (no 068fd0b link ran) is therefore now enforced rather than assumed.

### 2. The premise: can a time-limit kill leave a failed record?
**How simulations launch.** `run_one` (`evolve/evaluate.py:486`) calls `subprocess.run(cmd, capture_output=True, timeout=3600)`:
- There is no `start_new_session`/`preexec_fn`, so each simulation stays in the tuner's process group.
- The return code is negative on a signal death: `"exit -15"` for SIGTERM, `"exit -9"` for SIGKILL.
- Nothing in `vidur/` or `evolve/` installs a SIGTERM handler. `grep signal` finds only atexit hooks, and those do not run on a default SIGTERM.

`run_many` runs 10 search3b scenarios on 8 threads, and `evaluate_policy` returns only after all 10 return. A failed record therefore needs every scenario of the candidate to have finished or died, and the tuner to survive long enough to write. `submit_python_job.sbatch` runs `eval "$CMD"` from the batch shell, with no `srun`, no `--signal` and no trap.

**What Slurm does** (current SchedMD source):
- `slurmd/req.c`, `_rpc_terminate_job`: SIG_TIME_LIMIT (it only prints "CANCELLED ... DUE TO TIME LIMIT"), then SIGCONT, then SIGTERM via `_kill_all_active_steps(..., SIGTERM, 0 /*flags*/, ..., batch=true)`, then a wait of `MAX(KillWait, 5)` s, then SIGKILL to all steps.
- `slurmstepd/req.c`, `_handle_signal_container`: it calls `killpg(pgid)` on the batch step only with `KILL_FULL_JOB`/`KILL_JOB_BATCH`, which are not set here. Otherwise it calls `proctrack_g_signal`.
- `proctrack/cgroup` (`proctrack_p_signal`) signals a pid only if `signal_children_processes || slurm_task == 1 || signal == SIGKILL`. With the default `SignalChildrenProcesses=no`, **SIGTERM reaches only the batch shell**, not the tuner or the simulations.
- SIGKILL then goes to everything at once. On cgroup v2 it uses `cgroup.kill`, which is atomic. On v1 it is a tight `kill()` loop over the cgroup's pids.

**Consequences:**
- *Default configuration.* The batch shell dies at SIGTERM. The tuner and its simulations run on for KillWait seconds (often 30). A candidate that finishes in that window writes a valid, correct record. Then all of them die in one SIGKILL, and no failed record can be written.
- *`SignalChildrenProcesses=yes`, or `proctrack/pgid`.* SIGTERM goes to every pid, in one loop or in one `killpg`. In cgroup order (fork order), the tuner precedes its simulations. A default-action SIGTERM is fatal at generation time (`complete_signal` group-kills the process), so the tuner cannot run Python after its own `kill()`. For a record to be written, all 10 simulations of a candidate would have to exit, be reaped and be scored inside the few microseconds between the first `kill()` in the loop and the tuner's own. With 10 scenarios on 8 threads, a candidate's last 2 simulations even start only after 2 earlier ones finish.
- *SIGKILL loop on cgroup v1.* Same argument, for `exit -9`.

So the premise is right in effect: no failed record from a time-limit kill. Its wording ("takes the tuner down with its simulations") is slightly off. Under the default configuration the tuner survives SIGTERM, runs for KillWait seconds and dies with SIGKILL; this is harmless.

Unverified: Alpine's `ProctrackType`, `KillWait` and `SignalChildrenProcesses`. A login-node `scontrol show config | grep -iE 'proctrack|killwait|signalchildren'` would settle it. The conclusion holds for every combination above.

**If a kill did leave a failed record:**
- *What the new rule does.* It keeps the record, and the run silently diverges. `s2.py` forces the race (in-flight candidates write `exit -15` before the tuner dies), 40 chains: `(same keys, same best, has exit -15 records): {(False, True, True): 1, (False, False, True): 39}`.
- *How to detect it.* HEAD records carry `failures` reasons:
  - `exit -15` is a SIGTERM, and only a job cancel or time limit sends one to a simulation;
  - `exit -9` is ambiguous with a cgroup OOM kill, which is a genuine failure and is correctly kept;
  - link-1 records (1f9fde4) have no `failures` field and cannot be classified.
- *Suggested guard* (nit N-r4-2): warn, or refuse with instructions, when an invalid record's reasons are all `exit -15`.

### 3. False positives of the trajectory refusal
None found:
- *Partially recorded last generation from link 1.* That generation's keys are all requested by the replay (`s1.py`: 180/180 completed, with no `SystemExit`).
- *Resume after completion.* Every key is requested: 0 calls, no refusal (`s1.py`, D_r3 `s4.py`).
- *Duplicate params within the ES.* `s4.py` forces them by quantising `to_value` (21 duplicate lines over 40 reference runs). The refusal never trips: 39/39 chains exact, with no `SystemExit`. A concurrent duplicate writes two lines with one key; `loaded` holds the key once, and both requests discard it.
  - The 40th chain crashed in the stub with `KeyError: 'QUEUE_W'`. The likely cause is two threads writing the same content-named candidate file at once (one reads it while the other truncates it). That needs a concurrent duplicate key, which is practically impossible with 6-digit rounding; nit N-r4-1.
- *Link-1 key format.* Same `params_of` and `json.dumps(sort_keys)`, and float JSON round-trips are exact.
- *`--parallel` or `--workers` changed between links.* Neither affects the trajectory, and neither is in `meta`; correct.

### 4. `tuner_sha`
- *Behaviour.* `s5.py`: link 2 creates `meta.json`; a comment-only edit of the tuner; link 3 exits with `run dir was started with {...10 keys...}, not {...}`. The difference is in the last key of two dicts of about 400 characters each, and the message gives no remedy. The refusal happens before `load_history`, so no data is touched; later `afterany` links refuse too, and the chain stops.
- *Is it desired?* Failing closed is right for trajectory-relevant edits (`FAMILIES`, `ask_tell`, `params_of`, `lam_of`, rounding). It also blocks benign edits: comments, `install`, `load_history`, or a better error message.
- *Operational consequence.* Sync any tuner fix to Alpine either before link 2 starts (link 1 ends about 08:35, and link 2 writes `meta.json` with the sha it runs) or after the chain ends. Otherwise links 3–4 refuse, and someone has to restore the file or edit `meta.json` by hand.
- *Suggested fix* (minor m-r4-2):
  - Print only the differing keys, plus the remedy: "restore the tuner/template of the run (meta tuner_sha X), or, if the edit cannot change the trajectory or the scoring, set tuner_sha in meta.json to Y".
  - Optionally, hash only the trajectory-defining source (`inspect.getsource` of `ask_tell`, `reflect`, `lam_of`, `to_value`, `render`, a module-level `params_of`, and `repr(FAMILIES)`) instead of the whole file.

---

### [major] M-r4-1: the repurposed end-to-end test does not exercise the fix; it passes on 7cb131c
`tests/test_tune_baselines.py:155-191`

**Failure scenario.** The transient failure is the 5th distinct key: index 4, generation 0. The cuts are 3, 12 and 25, so the failure is either not yet recorded (cut 3) or sits in an earlier generation than the last recorded one (cuts 12 and 25). 7cb131c kept such failures too. Only the "last generation" case changed, and no test covers it. Nothing tests the trajectory refusal, the "no valid evaluation" exit, or a template/tuner-sha mismatch either. D_r3 asked for "a post-finish link with a failure in the final generation" and "a genuine failure in an interrupted last generation is kept".

**Evidence.** The test file run with `tune_baselines` replaced by the 7cb131c version (`r4/test_disc.py`): `2 passed` for `-k resume`.
- The variant `r4/test_disc2.py` puts the failure at the 21st distinct key (generation 3, the final one at budget 25) and uses cuts `(3, 12, 22, 25)`.
- On 7cb131c it fails: `assert key(resumed) == key(full)`.
- On HEAD it passes (`r4/test_head2.py`: `1 passed`).

**Suggested fix:**
- Apply that change to the test: `len(seen) == 21`, cuts `(3, 12, 22, 25)`.
- Add a refusal case: history from seed 0, resume with a `meta.json` holding seed 1 copied in, and expect `SystemExit`. Better, test the early refusal of m-r4-1.

**Hashed file:** no.

### [minor] m-r4-1: the trajectory refusal fires only after the replay has spent the whole budget
`evolve/analysis/tune_baselines.py:219-220`

**Failure scenario.** Link 1 (1f9fde4) leaves no `meta.json`, so link 2 cannot compare its arguments with link 1's. If link 2 is started with another `--seed`, or anything else that changes the trajectory, it evaluates the new trajectory, writes `meta.json` with its own arguments, and is killed at 4 h before it reaches the check. Link 3 agrees with that `meta.json`. Only the link that finishes `ask_tell` refuses, after about 12 h of Alpine time, with 124 off-trajectory lines mixed into the history.

**Evidence** (`s3.py`, mooncake 125, link 1 killed after 40):
```
link2 (seed 1): killed | new evaluations 46 | history lines 85
link3 (seed 1): killed | new evaluations 46 | history lines 130
link4 (seed 1): SystemExit: 39 recorded evaluations are not on the replayed ES trajectory | new evaluations 34 | history lines 164
```

**Suggested fix.** A legitimate history is a generation prefix: generations before g are complete and g is partial, because `ex.map` blocks before g+1 is sampled. So at the end of the first generation that had a cache miss, `loaded` must be empty.
- In `score`, set `miss[0] = True` after the cache check fails.
- In `evaluate`, after `ex.map`: `if miss[0] and loaded: raise SystemExit(...)`.
- Keep the final check.

Checked as a patched copy (`r4/tb_early.py`): `s1` transient and none, 40+40 chains, all exact and with no false refusal. `s3` refuses in link 2 after one generation (`new evaluations 7`).

**Hashed file:** no.

### [minor] m-r4-2: the `tuner_sha` refusal trips on any byte change and does not say what differs or what to do
`evolve/analysis/tune_baselines.py:171-175`. See check 4 for the evidence and the fix. Operationally: sync tuner edits only before link 2 starts or after the chain ends.

**Hashed file:** no.

### [minor] m-r4-3: duplicate-key lines resolve last-wins, but the ES consumed the first
`evolve/analysis/tune_baselines.py:130`

**Failure scenario.** Two lines with one key and different results. The concurrent-duplicate case is practically unreachable. The other case, a 068fd0b-style retry, is now refused by the `meta` check. If it did happen, the replay would feed the ES the later fitness where the original run consumed the earlier one.

**Evidence.** D_r3 `s7.py` rerun on HEAD (a hand-made 068fd0b retry with no `meta.json`): `link2 done | valid-set equal False`.

**Suggested fix.** Use `done.setdefault(key, r)` (first wins), or refuse when two records of one key disagree on `valid` or `fitness`.

**Hashed file:** no.

### [nit] N-r4-1: concurrent identical keys share one content-named candidate file
`evolve/analysis/tune_baselines.py:192-193`. Two threads scoring the same key at once both `write_text` the same path while the other thread's simulations may be reading it. This needs a concurrent duplicate key, which is unreachable with 6-digit rounding. The `s4.py` stub crash is the likely instance. Accept as is, or move the in-flight check under the lock.

### [nit] N-r4-2: no detection of kill-made failures, should a configuration ever produce them
See check 2. Print each invalid record's reasons at load time, and refuse, with instructions, a last-generation record whose reasons are all `exit -15`. This costs a few lines.

## Before link 2 (about 08:35 PDT)
- Confirm that neither run dir has a `meta.json`. If one exists, f7a8810 refuses, and correctly so.
- Sync f7a8810, or any follow-up fix, before link 2 starts. Do not sync between links 2 and 4 (m-r4-2).
- Link 1's invalid records, if any, will be kept by design. Under the Slurm analysis above they are genuine failures.
