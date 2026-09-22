"""
Checkpoint Bug Fix - Proposed Changes
=====================================

PROBLEM:
- atexit handler was removed in commit 728d1df
- _save_checkpoint() only called every 2000 requests
- Final training (requests 2000-3994) never gets saved
- Next iteration loads stale checkpoint from request 2000

SOLUTION:
Call scheduler.save_final_checkpoint() explicitly after simulation completes,
before the Python process exits.

CHANGES:

1. vidur/scheduler/global_scheduler/rl_cache_aware_global_scheduler.py
   - Add public method: save_final_checkpoint()
   - This wraps _save_checkpoint() but can be called externally

2. vidur/simulator.py
   - After run() completes, call self._scheduler.save_final_checkpoint()
   - Only for RL schedulers in training mode

ALTERNATIVE (simpler):
- Just call _save_checkpoint() at the end of Simulator.run()
- No need for a new public method
"""