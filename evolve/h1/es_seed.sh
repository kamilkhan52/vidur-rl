#!/bin/bash
# h1 RL recipe (evolve/h1/PREREG.md section 4) for one seed; "oracle" as the second argument gives the section 5b arm
# (true output lengths, a diagnostic). The comma lists live here because sbatch --export splits on commas.
#   bash evolve/h1/es_seed.sh 40            -> evolve/rl_plan/out/es/h1_gen8_s40
#   bash evolve/h1/es_seed.sh 40 oracle     -> evolve/rl_plan/out/es/h1_gen8_s40_oracle
set -euo pipefail
SEED=$1
MODE=${2:-}
CFG=""
for t in Tool3 Conv3; do
  for s in 0.4 0.5982 0.8973 1.256; do CFG+="$t:8:$s,"; done
  for s in 0.2 0.2991 0.4487 0.628; do CFG+="$t:16:$s,"; done
done
OUT="evolve/rl_plan/out/es/h1_gen8_s$SEED"
EXTRA=()
if [ "$MODE" = oracle ]; then OUT+="_oracle"; EXTRA=(--oracle); fi
EVOLVE_CACHE_DIR="$PWD/evolve/.cache_h1" python -m evolve.rl_plan.rl_es --out "$OUT" \
  --seed "$SEED" --workers "${WORKERS:-30}" --device h100df --arch linear --fitness gen8 --pairs 6 --sigma 0.03 \
  --lr 0.05 --eval-every 10 --bound 0 --length 2000 --windows 16 --hours 15.5 --configs "${CFG%,}" \
  --exclude Tool3:4000:2000,Tool3:20000:3608,Tool3:14000:4000,Conv3:0:2004,Conv3:10000:2031,Conv3:6000:2000 \
  --eval-len 2000 --eval-starts Tool3:0/8000/11000,Conv3:2004/4000/8000 ${EXTRA[@]+"${EXTRA[@]}"}
