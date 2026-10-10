#!/bin/bash
# v4 evolved-router campaign (plan step 2; evolve/SELECTION.md §4): three independent search runs on search3b
# against bar v4t (BAR_V4 + the tuned Mooncake/Dynamo control), objective tail4, seeded with the v3 pilot finalists.
# Budget per run: ITERATIONS=60 x BATCH=2 = 120 proposals + 3 seeds = 123 candidate evaluations, about the tuning
# control's 125. --iterations is a TOTAL: the driver counts the iterations already in the archive, so the chain
# links together stop at 120 proposals and links that start after the budget is spent exit at once. A run can end
# below the budget if the chain's 4 x 4 h runs out (re-launch with RESUME=1 to add links) or, for the DeepSeek
# runs, at MAX_COST_USD; report the archive's proposal count with the results.
#
# Run from the Mac after both tuning runs are installed (tune_baselines --install), so evolve/policies/tuned/ exists.
# Each run is a chain of 4 h, 22-CPU jobs (small jobs backfill on acpu; 48-CPU jobs waited hours). The driver resumes
# from its archive, so a chain link that times out loses at most the proposals in flight.
#   scripts/launch_v4_campaign.sh            # submit
#   DRY=1 scripts/launch_v4_campaign.sh      # print what would run; opens no ssh connection
#   RESUME=1 scripts/launch_v4_campaign.sh   # add chains to runs that already have an archive
# Refuses to submit when: the tuned policies were frozen under another code_version; Alpine's tree at $R has another
# code_version than the Mac (checked BEFORE anything is copied: syncing hashed code into a tree that running jobs use
# would change their simulator mid-run; sync deliberately, with no jobs running from $R, then re-run); a job of the
# same chain is already queued; or a run dir already has an archive (unless RESUME=1).
# The Claude run needs the relay served from the Mac:
#   .venv/bin/python -m evolve.relay serve --remote alpine:$R/relay/v4_claude_r1
set -euo pipefail
R=/scratch/alpine/kamil@colostate.edu/vidur-rl-v4
RUNS="v4_ds_r1 v4_ds_r2 v4_claude_r1"
LINKS=4
cd "$(dirname "$0")/.."
CV=$(.venv/bin/python -c "import evolve.evaluate as ev; print(ev.code_version())")
for f in dynamo_kv_cost_tuned_best.py mooncake_min_ttft_tuned_best.py; do
  p=evolve/policies/tuned/$f
  if [ ! -f "$p" ]; then
    echo "missing $p: install the tuning results first"; [ -n "${DRY:-}" ] || exit 1
  elif ! head -3 "$p" | grep -q "code_version $CV"; then
    echo "$p was frozen under another code_version than the current $CV: re-tune or re-install"; [ -n "${DRY:-}" ] || exit 1
  fi
done
SEEDS="evolve/finalists/l3_48a067a7df63.py evolve/finalists/l3_1dafac434ec8.py evolve/finalists/l3_6595477664aa.py"
run() { if [ -n "${DRY:-}" ]; then echo "$*"; else "$@"; fi; }
remote_cv() {
  ssh alpine "cd $R && source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate vidur \
    && python -c 'import evolve.evaluate as ev; print(ev.code_version())'" 2>/dev/null || echo unknown
}

# 1. Read-only checks on Alpine, before anything is copied (one ssh call for the queue and the run dirs).
if [ -n "${DRY:-}" ]; then
  echo "[dry] would check: Alpine code_version at $R == $CV; no queued job named <run>_1..$LINKS; no archive in" \
       "$R/evolve/runs/<run> unless RESUME=1"
else
  REMOTE_CV=$(remote_cv)
  [ "$REMOTE_CV" = "$CV" ] || { echo "code_version Mac $CV vs Alpine $REMOTE_CV at $R: refusing to copy or submit"; exit 1; }
  STATE=$(ssh alpine "squeue -h -u \$USER -o %j; for n in $RUNS; do [ -s $R/evolve/runs/\$n/archive.jsonl ] && echo ARCHIVE:\$n; done; true")
  bad=0
  for name in $RUNS; do
    if printf '%s\n' "$STATE" | grep -Eq "^${name}_[0-9]+$"; then echo "a job of chain $name is already queued"; bad=1; fi
    if printf '%s\n' "$STATE" | grep -qx "ARCHIVE:$name" && [ -z "${RESUME:-}" ]; then
      echo "$R/evolve/runs/$name already has an archive (RESUME=1 to continue it)"; bad=1
    fi
  done
  [ "$bad" = 0 ] || exit 1
fi

# 2. Copy code and the tuned policies (code_version equal, so no hashed file changes; caches and runs are excluded).
run rsync -a --exclude __pycache__ evolve/ "alpine:$R/evolve/" --exclude '.cache*' --exclude runs --exclude 'evcache*'
run rsync -a --exclude __pycache__ vidur/ scripts/ "alpine:$R/" --relative
if [ -z "${DRY:-}" ]; then
  REMOTE_CV=$(remote_cv)
  [ "$REMOTE_CV" = "$CV" ] || { echo "after the copy: code_version Mac $CV vs Alpine $REMOTE_CV"; exit 1; }
fi

submit_chain() {  # name, links, extra env
  local name=$1 links=$2 env=$3 prev="" dep
  # One baseline cache per run avoids write races. It is seeded from the tuning run's cache on Alpine, which holds
  # the BAR_V4 baselines on search3b; each run then computes the two tuned members itself (20 simulations).
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
submit_chain v4_ds_r1 $LINKS "SEED=1,MAX_COST_USD=8"
submit_chain v4_ds_r2 $LINKS "SEED=2,MAX_COST_USD=8"
submit_chain v4_claude_r1 $LINKS "SEED=3,PROPOSER=relay,RELAY_DIR=$R/relay/v4_claude_r1,MAX_COST_USD=1000"  # relay spend is not metered; a cap of 0 would stop at once
echo "code_version $CV. Start the relay for v4_claude_r1 on the Mac (see header)."
