# Code review loop, 2026-10-10: brief for every reviewer

You are reviewing one chunk of a research codebase as if it were a pull request. Repo: `/Users/kamil/vidur-rl-working`
(branch `openevolve-rl-search`). Python: `.venv/bin/python` (system python lacks pandas). The diff under review is
`git diff 1823e99..HEAD -- <your chunk's paths>`; read the surrounding code as needed.

## Project context (enough to judge correctness)
- Vidur is a discrete-event LLM-serving simulator. We study request routers across replicas with prefix caching,
  scored on P99 tails (E2E, TTFT, TPOT per request, TBT per token) against a bar of baseline routers.
- Results feed a paper. What matters most: (1) wrong numbers (a bug that silently changes simulated outcomes or
  scores), (2) broken reproducibility (cache keys, `code_version`, prediction-bundle keys, seeds), (3) claims in
  commit messages or docstrings that the code does not deliver (e.g. "behaviour-preserving", "byte-identical",
  "exact fork"), (4) tests that would pass even if the property they name were broken.
- `code_version` (`evolve/evaluate.py`) is a content hash of `vidur/**/*.py` except `_VERSION_EXCLUDE`, plus
  `scripts/tail_metrics.py`, `data/profiling/**.csv` and four harness functions. Baseline results are cached under it.
  Current value `c16266a8ef49`; long jobs are running against it.
- v4 = a simulator release today. Changes claimed behaviour-preserving were checked with a golden set (6 scenarios x
  5 policies, byte-identical outputs on v3 data). The trace change (unpadded prompts, 128K context cap) is the
  intended behaviour change.

## Rules
- READ-ONLY except your review file. Do not edit code, do not commit, do not stash/checkout/reset.
- The Mac is busy with long simulations: run only targeted unit tests (`-q`, a few files), never the full suite, at
  most 2 simulator processes, never `pkill`/kill anything. Do not run anything on Alpine.
- Verify before reporting: for each bug, give the concrete input/state that triggers it and what goes wrong. Prefer a
  tiny reproduction (a few lines run with `.venv/bin/python -c`) over speculation. Mark unverified suspicions as such.
- Severity: **blocking** (wrong results, data loss, broken reproducibility, a false claim of equivalence), **major**
  (a real bug on a plausible path, a test that does not test its claim, a significant robustness gap), **minor**
  (edge case, clarity that could mislead), **nit**. Do not pad with style nits; this codebase's style is terse.
- For each finding, say whether the fix would touch a `code_version`-hashed file (it would invalidate cached baselines).

## Output
Write `reviews/2026-10-10/<chunk>_r<round>.md` with: a 3-line verdict (approve / approve with minors / changes
requested), then findings as `### [severity] title`, `file:line`, failure scenario, evidence (repro output), suggested
fix, hashed-file yes/no. Save the file as you go. Return a summary of at most 300 words listing each finding with
severity.
