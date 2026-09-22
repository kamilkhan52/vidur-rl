# Scripts archived on 2026-09-22

These scripts were moved out of `scripts/` because nothing live references them: no vault
note, repo doc, dev note, test, package module, or still-live script names or imports them.
The reference map was computed over every text file in the repo and in
`hermes-memories/20_Research/Vidur-RL`, iterating until no archived script was reachable from a
live one. They are kept, with history, for provenance of the February/March figures.

Most are one-off iterations of the same figures (`plot_graduation_progress_*`, eight
`patch_*` label/threshold tweaks, `plot_main_dense_colors*`), early sweep launchers
superseded by `run_baseline_sweep.sh` and the `submit_*_array.sbatch` files, and
resource-probe scripts (`test_4core*.sh`).

Note: 28 of these (the `patch_*`, most `plot_graduation_progress_*`, `csv_to_json.py`,
`dump_plot_data.py`, `analyze_results_robust.py`, etc.) were untracked working files until
commit `b643560` swept them into git with a `git add scripts`; that commit's message only
describes the partition change. They were March plot-iteration artifacts, never part of a
pipeline.

Kept in `scripts/` although only January dev notes cite them: `analyze_training_progress.py`,
`fix_trace_formats.py`, `train_mooncake.sh`, `train_rl_continuous.sh`, `train_rl_long.sh`.

## Archived files
- `analyze_comprehensive_sweep.py`
- `analyze_grad_progress.py`
- `analyze_results_robust.py`
- `analyze_stalled_tasks.py`
- `analyze_sweep.py`
- `cleanup_results.py`
- `csv_to_json.py`
- `diagnose_rl_performance.py`
- `dump_plot_data.py`
- `evaluate_mooncake_scaling.py`
- `exp03_params.csv`
- `find_qps_limits.py`
- `fix_graduation_pointers.py`
- `generate_baseline_tables.py`
- `generate_progress_table.py`
- `generate_trace_heatmaps.py`
- `identify_rl_env_targets.py`
- `patch_enhanced_labels.py`
- `patch_graduation.py`
- `patch_graduation_comparisons.py`
- `patch_labels.py`
- `patch_labels_all.py`
- `patch_rl_vs_baseline.py`
- `patch_threshold.py`
- `patch_threshold_main.py`
- `plot_comparisons.py`
- `plot_grad_progress.py`
- `plot_graduation_comparison.py`
- `plot_graduation_comparisons_patched.py`
- `plot_graduation_from_slurm.py`
- `plot_graduation_progress_enhanced.py`
- `plot_graduation_progress_postfix.py`
- `plot_graduation_progress_postfix_final.py`
- `plot_graduation_progress_postfix_threshold.py`
- `plot_graduation_progress_with_baselines.py`
- `plot_graduation_progress_with_baselines2.py`
- `plot_graduation_progress_with_baselines2_threshold.py`
- `plot_graduation_ttft.py`
- `plot_main_dense_colors.py`
- `plot_main_dense_colors_fixed.py`
- `plot_priority2_postfix.py`
- `plot_priority_1_heatmaps.py`
- `plot_trace_heatmaps_from_csv.py`
- `plot_training.py`
- `plot_ttft_vs_e2e_comparison.py`
- `plot_unified_colors.py`
- `rl_training_configs.csv`
- `rl_vs_baseline_postfix.py`
- `run_global_sweep.py`
- `run_priority2_sweep.sh`
- `run_priority_sequential.sh`
- `submit_EXP03.sbatch`
- `submit_alpine_sweep.sbatch`
- `submit_alpine_training.sbatch`
- `submit_priority2_missing.sh`
- `submit_priority2_toolconv.sh`
- `submit_rl_training_array.sbatch`
- `test_4c_32gb.sh`
- `test_4core.sh`
- `test_4core_v2.sh`
- `train_16gpu_toolagent.sh`
