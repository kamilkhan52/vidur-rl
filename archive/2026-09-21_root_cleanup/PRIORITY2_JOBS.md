# Priority 2 Baseline Sweep - Job Documentation

## Overview
Priority 2 sweep evaluates 4 selected schedulers across all model configurations and traces to validate tension zone patterns and identify best config for RL training.

## Jobs

### Job 1: Main Sweep (23901516) - 176 tasks
**Status**: Running
**Config**: 9 cores, 32GB per task
**Submitted**: 2026-02-15 17:07 MST

**Tasks included**:
- 7b-8R: 56 tasks (Synth 24 + Tool 16 + Conv 16) ✓
- 7b-16R: 8 tasks (Synth scales 0.75, 1.0 only) ⚠️
- 70b-8R: 56 tasks (Synth 24 + Tool 16 + Conv 16) ✓
- 70b-16R: 56 tasks (Synth 24 + Tool 16 + Conv 16) ✓

**Output**: `/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/{SCHEDULER}/{MODEL}-{REPLICAS}-{TRACE}/scale{SCALE}/`

**Note**: Tasks 177-224 failed (array out of range). This was expected - only 176 task definitions exist.

---

### Job 2: Missing Tasks (TBD) - 32 tasks
**Status**: To be submitted
**Config**: 9 cores, 32GB per task
**Purpose**: Fill gaps from Job 1

**Tasks included**:
- 7b-16R-Tool: 16 tasks (4 schedulers × 4 scales: 1.5, 3.0, 6.0, 12.0)
- 7b-16R-Conv: 16 tasks (4 schedulers × 4 scales: 1.5, 3.0, 6.0, 12.0)

**Note**: 7b-16R-Synth scales 1.5-12.0 already completed in Priority 1 (Job 23851270), so NOT included here.

**Output**: Same directory structure as Job 1

---

## Complete Task Matrix

### Configuration: 4 schedulers × 4 model configs × 3 traces
- **Schedulers**: STICKY_LOR, ROUND_ROBIN, LOR, RANDOM
- **Model configs**: 7b-8R, 7b-16R, 70b-8R, 70b-16R
- **Traces**: Synth, Tool, Conv

### Scale ranges per trace
- **Synth**: {0.75, 1.0, 1.5, 3.0, 6.0, 12.0} - 6 scales
- **Tool**: {1.5, 3.0, 6.0, 12.0} - 4 scales
- **Conv**: {1.5, 3.0, 6.0, 12.0} - 4 scales

### Task count breakdown
| Config | Synth | Tool | Conv | Total | Job 1 | Job 2 | Priority 1 |
|--------|-------|------|------|-------|-------|-------|------------|
| 7b-8R  | 24    | 16   | 16   | 56    | 56    | -     | -          |
| 7b-16R | 24    | 16   | 16   | 56    | 8     | 32    | 16         |
| 70b-8R | 24    | 16   | 16   | 56    | 56    | -     | -          |
| 70b-16R| 24    | 16   | 16   | 56    | 56    | -     | -          |
| **Total** | **96** | **64** | **64** | **224** | **176** | **32** | **16** |

**Explanation**:
- Priority 1 (Job 23851270): Ran 7b-16R-Synth scales 1.5-12.0 (16 tasks)
- Job 1 (23901516): Main sweep, includes 7b-16R-Synth scales 0.75, 1.0 (8 tasks)
- Job 2 (missing): 7b-16R-Tool and 7b-16R-Conv (32 tasks)
- Combined: All 224 unique configs covered

## Resource Optimization
**Original plan**: 18 cores, 64GB → ~18 parallel jobs
**Optimized**: 9 cores, 32GB → ~80+ parallel jobs (4.4x improvement!)

**Alpine behavior**: Memory request auto-determines core allocation
- 32GB ÷ 3.75 GB/core = 8.53 → rounds to 9 cores
- Cannot request fewer cores than memory requires

## Expected Timeline
- Job 1 (176 tasks): ~2-3 hours with high parallelism
- Job 2 (32 tasks): ~1 hour running in parallel
- **Total completion**: ~3 hours from start

## Output Structure
```
/scratch/alpine/kamil@colostate.edu/vidur-rl/baselines/
├── STICKY_LOR/
│   ├── 7b-8R-Synth/
│   │   ├── scale0.75/
│   │   ├── scale1.0/
│   │   ├── scale1.5/
│   │   └── ...
│   ├── 7b-16R-Tool/
│   │   ├── scale1.5/
│   │   └── ...
│   └── ...
├── ROUND_ROBIN/
├── LOR/
└── RANDOM/
```

## Notes
- Chrome traces DISABLED (prevent disk bloat)
- CPU overhead modeling SKIPPED (no cpu_overheads.csv)
- 70b models use TP=4, 7b models use TP=1
- Each config runs 5 iterations (NUM_RUNS=5)
- Outputs to SCRATCH not PROJECT directory (quota limits)
