#!/bin/bash
# v4 evolved-router campaign (plan step 2; evolve/SELECTION.md §4): three independent search runs on search3b
# against bar v4t (BAR_V4 + the tuned Mooncake/Dynamo control), objective tail4, seeded with the v3 pilot finalists.
# Budget per run: 60 iterations x batch 2 = 120 proposals + seeds, about the tuning control's 125 evaluations.
#
# Run from the Mac after both tuning runs are installed (tune_baselines --install), so evolve/policies/tuned/ exists.
# Each run is a chain of 4 h, 22-CPU jobs (small jobs backfill on acpu; 48-CPU jobs waited hours). The driver resumes
# from its archive, so a chain link that times out loses at most the iteration in flight.
#   scripts/launch_v4_campaign.sh            # submit
#   DRY=1 scripts/launch_v4_campaign.sh      # print the sbatch commands only
# The Claude run needs the relay served from the Mac:
#   .venv/bin/python -m evolve.relay serve --remote alpine:$R/relay/v4_claude_r1
set -euo pipefail
R=/scratch/alpine/kamil@colostate.edu/vidur-rl-v4
cd "$(dirname "$0")/.."
for f in dynamo_kv_cost_tuned_best.py mooncake_min_ttft_tuned_best.py; do
  [ -f "evolve/policies/tuned/$f" ] || { echo "missing evolve/policies/tuned/$f: install the tuning results first"; exit 1; }
done
CV=$(.venv/bin/python -c "import evolve.evaluate as ev; print(ev.code_version())")
SEEDS="evolve/finalists/l3_48a067a7df63.py evolve/finalists/l3_1dafac434ec8.py evolve/finalists/l3_6595477664aa.py"
run() { if [ -n "${DRY:-}" ]; then echo "$*"; else "$@"; fi; }

# Code, tuned policies and the Mac's v4 baseline cache (one cache per run avoids write races).
run rsync -a --exclude __pycache__ evolve/ "alpine:$R/evolve/" --exclude '.cache*' --exclude runs --exclude 'evcache*'
run rsync -a --exclude __pycache__ vidur/ scripts/ "alpine:$R/" --relative
REMOTE_CV=$(ssh alpine "cd $R && source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate vidur \
  && python -c 'import evolve.evaluate as ev; print(ev.code_version())'" 2>/dev/null || echo unknown)
[ -n "${DRY:-}" ] || [ "$REMOTE_CV" = "$CV" ] || { echo "code_version Mac $CV vs Alpine $REMOTE_CV"; exit 1; }

submit_chain() {  # name, links, extra env
  local name=$1 links=$2 env=$3 prev="" dep
  run ssh alpine "mkdir -p $R/evolve/evcache_$name && cp -n $R/evolve/evcache_tune_dynamo/baselines.json $R/evolve/evcache_$name/"
  for i in $(seq 1 "$links"); do
    dep=${prev:+--dependency=afterany:$prev}
    cmd="cd $R && sbatch --parsable $dep --cpus-per-task=22 --mem=88G --time=04:00:00 --job-name=${name}_$i \
      --export=ALL,REPO_DIR=$R,RUN_DIR=$R/evolve/runs/$name,EVOLVE_CACHE_DIR=$R/evolve/evcache_$name,PROTOCOL=v2,BAR=v4t,\
SEARCH_STAGE=search3b,OBJECTIVE=tail4,ITERATIONS=60,BATCH=2,WORKERS=10,SEED_POLICY='$SEEDS',$env \
      scripts/submit_evolve_driver.sbatch"
    if [ -n "${DRY:-}" ]; then echo "ssh alpine \"$cmd\""; prev="DRY$i"; else prev=$(ssh alpine "$cmd"); echo "$name link $i: $prev"; fi
  done
}
submit_chain v4_ds_r1 4 "SEED=1,MAX_COST_USD=8"
submit_chain v4_ds_r2 4 "SEED=2,MAX_COST_USD=8"
submit_chain v4_claude_r1 4 "SEED=3,PROPOSER=relay,RELAY_DIR=$R/relay/v4_claude_r1,MAX_COST_USD=1000"  # relay spend is not metered; a cap of 0 would stop at once
echo "code_version $CV. Start the relay for v4_claude_r1 on the Mac (see header)."
