# evolve/ — automated router discovery for Vidur

An end-to-end research loop that searches for a request router with lower **tail latency**
(P99 E2E, TTFT, TPOT) than the heuristic routers, on Mooncake traces. It evolves the router
**code** inside a fixed, audited harness; the simulator and the metrics never change.

## Why this design (2026-09-27 survey)

Three surveys (DeepMind's AlphaEvolve/FunSearch/AlphaDev line; systems-specific work such as
Berkeley's ADRS, OpenEvolve, ShinkaEvolve, SkyDiscover, MIT's Glia and Engram, AIChilles; and a
fit analysis of this repo) converged on:

- **Evolve a small routing function, not an RL reward.** AlphaEvolve's production Borg win
  evolved a scoring heuristic and chose it over deep RL for interpretability and deployability;
  FunSearch evolves a priority function in a fixed skeleton. Our earlier OpenEvolve run
  (EXP04) evolved the DQN reward, so every candidate needed a noisy multi-hour RL training run;
  it collapsed.
- **Closest prior work: Glia (arXiv 2510.27176) and Engram (arXiv 2603.21321)** evolved Vidur
  routers with LLM agents and beat OpenEvolve on **mean** latency (ShareGPT, 4×A10). Our
  contribution must be **tail** latency on Mooncake with prefix caching at 8/16 replicas,
  with held-out generalisation, paired statistics and robustness checks.
- **Cascaded evaluation, held-out scenarios, several perturbations, and adversarial
  robustness.** AIChilles (arXiv 2606.15834) showed reported gains of evolved systems code
  collapsing (19 % → 0 %, 49 % → 3.9 %) once robustness was tested. ArchAgent reported
  "simulator escapes". Hence the static guard, the completion gate and the held-out set.
- **Engine:** a small tournament loop of our own (`driver.py`). A simple (1+1)/tournament loop
  was competitive with FunSearch/EoH/ReEvo (arXiv 2407.10873), and OpenEvolve's LLM layer
  speaks the OpenAI API while our proposer uses the official Anthropic SDK. OpenEvolve,
  ShinkaEvolve (native SLURM) or SkyDiscover can replace the driver later; the harness,
  guard and evaluator are engine-independent.

## Pieces

| File | Role |
|---|---|
| `vidur/scheduler/global_scheduler/evolved_global_scheduler.py` | Harness: loads a policy file (`--global_scheduler_config_type evolved --evolved_global_scheduler_config_policy_path P`), gives it frozen views, validates every decision. Output lengths, future arrivals and simulator objects are withheld. |
| `evolve/policies/*.py` | Seeds: ROUND_ROBIN, LOR, STICKY_LOR, LOP_UNCACHED ported to the contract. Tests prove they reproduce the built-in routers' `request_metrics.csv` exactly. |
| `evolve/guard.py` | Stage 0a: static checks (import allowlist; no files, clock, network, global RNG, dunder access, eval/exec, global state). |
| `evolve/evaluate.py` | Stage 0b crash check, stage 1 screen, scoring. Deterministic cached windows, fresh process per simulation, completion gate, baselines keyed by code version + trace hash + scenario. |
| `evolve/proposer.py` | `claude-opus-5` via the Anthropic SDK: adaptive thinking, effort `high`, streaming, server-side refusal fallback (`fallbacks: "default"`). No sampling parameters (rejected by the model); diversity comes from parent/inspiration selection. |
| `evolve/driver.py` | Tournament selection over the archive → prompt with parent code, per-scenario feedback, two inspirations and recent failures → guard → crash check → stage 1. Append-only `archive.jsonl` with lineage and proposer usage. |

## Score

Per scenario *s* and metric *m* ∈ {E2E, TTFT, TPOT}:
`r[s,m] = ln(P99_candidate / min over baselines P99)`;
`fitness = −(mean r + 0.5·max_s mean_m r) − penalty`, where the penalty adds
`ln(mean_E2E / best mean_E2E)` for every scenario with mean E2E > 5 % worse than the best
heuristic. 0 = ties the per-scenario best heuristic on every tail; higher is better.
Requests in the first 20 % of a window are simulated but not scored (warm-up).

## Scenario sets

- **Stage 1 (search):** Tool and Conv, 8 replicas, scale 1.5 and 2.0, two 1,000-request
  windows each (mid and late trace), perturbation 1 (seed 101, 3 s in-bucket jitter).
  8 simulations per candidate, ~1 min each on the Mac.
- **Held out, never shown to the search:** all of Synth, 16 replicas, scale 3.0,
  perturbations 2–3, other windows. Candidates are frozen before held-out evaluation, and the
  total number of candidates tried is reported.

## Gates before a real run

1. **Frontier v1 — done (2026-09-28), and it changes the scenario set.** All 288 runs completed
   cleanly. Most of the old grid is lightly loaded once replicas stop wedging (LOR queueing
   <= 7 %, LOR tail-best almost everywhere); 8R Tool/Conv at scale 1.5-2.0 collapses for every
   router. At 8R, scale 2.0 LOR is **metastable**: P99 E2E 16-74 s in some perturbations and
   ~2.9e4 s in others. That knee is where a better router can win big, so the search scenarios
   must sit at each scenario's knee. The knee-calibration sweep
   (`scripts/submit_load_calibration_array.sbatch`, 432 tasks) locates it.
2. **Stage-1 redesign and fidelity.** The current STAGE1 (1,000-request windows starting empty,
   8R at 1.5/2.0) is a placeholder: collapse builds over an hour, so empty-start windows cannot
   see it. After calibration, pick knee scenarios and either use long enough windows or start
   windows from a realistic backlog; then check that heuristic rankings on the screen agree with
   full-trace rankings (Spearman per metric) before trusting it.
3. **Credentials:** the proposer needs Anthropic API credentials where the driver runs.

## Running

```bash
.venv/bin/python -m pytest tests -q                      # 70+ tests, includes this loop
.venv/bin/python -m evolve.evaluate --policy evolve/policies/lor.py --workers 8
.venv/bin/python -m evolve.driver --run-dir evolve/runs/r1 --iterations 20 --batch 4 --workers 8
```
On Alpine, run the driver inside one allocation (`--ntasks=16 --cpus-per-task=1`) with
`--srun` so each simulation gets its own task slot.
