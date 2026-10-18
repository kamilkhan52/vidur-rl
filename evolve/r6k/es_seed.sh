#!/bin/bash
# r6k RL recipe (PREREG section 6) for one seed. Run through scripts/submit_python_job.sbatch with
# CMD="bash evolve/r6k/es_seed.sh <seed>": sbatch --export splits on commas, so the comma lists live here.
set -euo pipefail
SEED=$1
CFG=""
for t in Tool3 Conv3; do
  for s in 2.0 2.5 3.0 4.0 5.6; do CFG+="$t:4:$s,"; done
  for s in 1.0 1.25 1.5; do CFG+="$t:8:$s,"; done
done
EVOLVE_CACHE_DIR="$PWD/evolve/.cache_r6k" python -m evolve.rl_plan.rl_es --out "evolve/rl_plan/out/es/r6k_gen8_s$SEED" \
  --seed "$SEED" --workers "${WORKERS:-16}" --device rtxpro6000 --arch linear --fitness gen8 --pairs 6 --sigma 0.03 \
  --lr 0.05 --eval-every 10 --bound 0 --length 2000 --windows 16 --hours 15.5 --configs "${CFG%,}" \
  --exclude Tool3:4000:2000,Tool3:20000:3608,Conv3:0:2004,Conv3:10000:2031 \
  --eval-len 2000 --eval-starts Tool3:0/8000/14000,Conv3:2500/5000/7500
