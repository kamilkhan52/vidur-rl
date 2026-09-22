# Research Log: RL Cache-Aware Scheduling for LLM Inference

**Date:** February 5, 2026
**Project Goal:** Achieve state-of-the-art results for RL-based global scheduling in LLM serving and publish findings.

## Session Updates (2026-02-05)

### 1. Grid Sweep Stability (Resolution)
- **Problem:** Silent terminations and `TIMEOUT` errors in the 240-run grid sweep.
- **Root Cause Identified:** 
    1. **WSL/TTY Hangups:** Spawning background simulations directly from an agent session sometimes triggers termination when the session disconnects.
    2. **Script Timeouts:** `run_global_sweep.py` had a hardcoded 30-minute timeout. Under high load (6 workers), overloaded simulations (2-replica/4-replica) frequently exceeded 60 minutes.
- **Solution:** 
    - Launched sweep inside a detached **`tmux` session** (`sweep_session`).
    - **Removed all timeouts** from the Python `subprocess.run` calls in the sweep script.
- **Verification:** System has run stable for 5+ hours. Successfully survived runs taking **5302s (~88 minutes)**.

### 2. Comprehensive Research Sweep (In Progress)
- **Status:** **22 / 240 unique configurations completed.**
- **Observations:** 
    - **2-Replica Collapse:** Data for 2-replica clusters shows total system collapse (E2E latency > 10,000s) even at low QPS. These data points are likely "junk" due to extreme overloading.
    - **4-Replica Transition:** System is starting to show the "Tension Zone" at 4 replicas.
- **Parallelism:** Holding steady at **6 workers** locally. 
- **ETA:** Expected completion late afternoon 2026-02-05.

### 3. Infrastructure & Alpine
- **Scripting:** Created `scripts/submit_alpine_sweep.sbatch` and updated `run_global_sweep.py` to respect `MAX_WORKERS` environment variable.
- **Alpine Config:** Confirmed requirements for **128GB RAM** and workspace-local **`TEMP_DIR`** to avoid cluster node failures.

### 4. Analysis Tooling
- **Script:** Created `scripts/analyze_comprehensive_sweep.py`.
- **Functionality:** Generates PDF/PNG plots (QPS vs. Latency) normalized per-model (7B vs 70B) across all schedulers (`lor`, `sticky_lor`, `random`).
- **Initial Result:** Sent 7B E2E Latency plot to Kamil via Telegram; confirmed the layout and hue-by-scheduler logic.

## Phase 2: High-Pressure Sweep (2026-02-05)
- **Learnings from Phase 1:** 
    - 2-replica configurations are consistently overloaded (Latency > 1000s).
    - Scale 0.5 is under-utilized; 8 and 16 replica clusters can handle much more than Scale 8.0.
- **Decision:** Pivot the grid to focus on the "Breaking Point."
    - **Replicas:** [4, 8, 16]
    - **Scales:** [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    - **Threshold for Analysis:** Focus on the zone where E2E latency is between 100s and 500s.
- **Goal:** Identify the exact QPS where `sticky_lor` (and eventually RL) significantly outperforms `lor`.

---
*Status: 6 workers active in tmux. Stability verified.*
