# Root-level working artifacts archived on 2026-09-21

Moved out of the repository root during the publication-prep cleanup so that the
root holds only the package, scripts, data and documentation. Nothing was deleted.

## Tracked files (moved with `git mv`, history preserved)
- `comparison_*.png`, `exp02_*_heatmap.png`, `progress_table_*.png`,
  `graduation_progress_dense.png`, `grad_7b-8R-Synth_*_training_progress.{csv,png}`,
  `graduation_{fresh,latest,summary}.csv` — February/March plot outputs. Canonical copies
  of the figures live in the vault under `plots/Vidur-RL/`.
- `debug_stuck_requests.py`, `debug_trace.py`, `run_local_baseline.py`, `simulator_fix.py`,
  `test_simulator_*.{py,sh}`, `test_checkpoint_bug.sh`, `test_ckpt_bug/` — one-off February
  debugging scripts and their outputs (the checkpoint-saving bug and the "simulator stalls with
  pending requests" investigation). The stall was diagnosed on 2026-09-21 as requests larger
  than a replica's KV cache wedging the FCFS queue; the simulator now refuses such traces
  (`vidur/simulator.py::_validate_request_capacity`). Regression tests live in `tests/`.
- `CHECKPOINT_BUG_FIX.md`, `CHECKPOINT_FIX_README.md`, `SIMULATOR_FIX_NOTES.md`,
  `PRIORITY2_JOBS.md`, `PRIORITY2_TASK_ANALYSIS.md` — notes from those investigations.
- `memory/2026-02-05_session_summary.md` — an agent session dump.
- `scripts/vidur_config_alpine.py` — a stale near-copy of `vidur/config/config.py`
  (nothing referenced it).

## Untracked files (`untracked/`, gitignored)
Superseded plot variants (`graduation_progress_*`), the two directories named `*.png`,
`graduation_summary_{prefix,ttft}.csv`, `graduation_postfix_summary.csv`, `plot_data_dump.json`,
`check_missing_iterations.py`, and the January/February training logs. Kept on disk only.
