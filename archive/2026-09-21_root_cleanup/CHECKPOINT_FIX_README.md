# Checkpoint Save Fix

## Problem
Checkpoints were not being saved at the end of each training iteration, causing all iterations to load the same stale checkpoint (from request 2000 instead of the full 3994-request trace).

## Root Cause
- Commit 728d1df removed `atexit.register()` for final checkpoint saving
- `_save_checkpoint()` only called every 2000 requests (periodic saves)
- Training from request 2000→3994 was never saved
- Next iteration loaded stale checkpoint from request 2000

## Solution (Option 2 - Clean Architecture)
1. **Added `save_final_checkpoint()` method** to `RLCacheAwareGlobalScheduler`
   - Public method that can be called explicitly
   - Wraps `_save_checkpoint()` with logging and safety checks
   
2. **Modified `main.py`** to call `save_final_checkpoint()` after `simulator.run()`
   - Ensures checkpoint is saved WITH full trace training
   - Only called for RL schedulers in training mode

## Files Changed
- `vidur/scheduler/global_scheduler/rl_cache_aware_global_scheduler.py` - added public method
- `vidur/main.py` - call checkpoint save after simulation

## Testing
See `test_checkpoint_bug.sh` for reproduction and verification.

Expected behavior:
- Iteration 1: Saves checkpoint after full trace (3994 requests)
- Iteration 2: Loads checkpoint from iteration 1 (not stale 2000-request checkpoint)
- "Resuming from update" number should INCREASE between iterations
