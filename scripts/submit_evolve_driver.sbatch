#!/bin/bash
# Evolved-router search (EXP11) on one node: the driver proposes policies and runs every
# simulation as a local subprocess (up to --workers at a time). Resumable: rerun with the
# same RUN_DIR and the archive continues (seeds and baselines are not recomputed). ITERATIONS is the run's
# TOTAL across resubmissions (the driver counts what the archive holds); other settings must match run.json.
# Without BAR the driver scores protocol v2 against the v2 bar and warns; v4 runs pass BAR=v4t.
#   sbatch scripts/submit_evolve_driver.sbatch                      # DeepSeek via OpenRouter
#   PROPOSER=claude-code sbatch ...   (only where the claude CLI is logged in; not Alpine)
#   TASK=rl PROPOSER=relay RELAY_DIR=... sbatch ...   EXP12: evolve RL designs; proposals are
#       answered by `python -m evolve.relay serve --remote alpine:$RELAY_DIR` on the workstation.
# The OpenRouter key is read from ~/.config/vidur-evolve/openrouter.env (mode 600).
#SBATCH --job-name=vidur_evolve
#SBATCH --account=csu65_alpine1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve/slurm/evolve-%j.out
#SBATCH --error=/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve/slurm/evolve-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamil@colostate.edu,zeeraktya@gmail.com

set -euo pipefail
REPO_DIR="${REPO_DIR:-/projects/kamil@colostate.edu/vidur-rl}"
SCRATCH="/scratch/alpine/kamil@colostate.edu/vidur-rl/evolve"
TASK="${TASK:-router}"
RELAY_DIR="${RELAY_DIR:-}"
MODEL="${MODEL:-}"
OBJECTIVE="${OBJECTIVE:-balanced}"
SEED_POLICY="${SEED_POLICY:-}"  # space-separated list of policy files (repo-relative)
PROTOCOL="${PROTOCOL:-v1}"   # router task: v1 (EXP11) | v2 (two draws, bar = heuristics + Mooncake + Dynamo)
SEED="${SEED:-0}"
BAR="${BAR:-}"  # router v2: evaluate.BARS name (v4t = v4 bar + tuned Mooncake/Dynamo)
SEARCH_STAGE="${SEARCH_STAGE:-}"  # router v2: evaluate.STAGES set to search on (e.g. search3 = Llama-3-8B)
RL_PROTOCOL="${RL_PROTOCOL:-v2}"
RUN_DIR="${RUN_DIR:-$SCRATCH/runs/deepseek_r1}"
PROPOSER="${PROPOSER:-openrouter}"
ITERATIONS="${ITERATIONS:-60}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-16}"
MAX_COST_USD="${MAX_COST_USD:-8}"

source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate vidur
cd "$REPO_DIR"
export EVOLVE_CACHE_DIR="${EVOLVE_CACHE_DIR:-$SCRATCH/cache}"
mkdir -p "$RUN_DIR"
COMMIT=$( (git rev-parse --short HEAD 2>/dev/null) || cat COMMIT 2>/dev/null || echo unknown)
echo "commit $COMMIT task $TASK proposer $PROPOSER run_dir $RUN_DIR cache $EVOLVE_CACHE_DIR $(date)"
ls cache | grep prediction_bundle || echo "WARNING: no prediction bundle; results will not match other machines"
EXTRA=()
[ -n "$RELAY_DIR" ] && EXTRA+=(--relay-dir "$RELAY_DIR")
[ -n "$MODEL" ] && EXTRA+=(--model "$MODEL")
[ -n "$SEARCH_STAGE" ] && EXTRA+=(--search-stage "$SEARCH_STAGE")
[ -n "$BAR" ] && EXTRA+=(--bar "$BAR")
for sp in $SEED_POLICY; do EXTRA+=(--seed-policy "$sp"); done
EXTRA+=(--objective "$OBJECTIVE" --rl-protocol "$RL_PROTOCOL" --protocol "$PROTOCOL" --seed "$SEED")
python -m evolve.driver --run-dir "$RUN_DIR" --task "$TASK" --proposer "$PROPOSER" --scenarios stage1 \
    --iterations "$ITERATIONS" --batch "$BATCH" --workers "$WORKERS" --max-cost-usd "$MAX_COST_USD" ${EXTRA[@]+"${EXTRA[@]}"}
