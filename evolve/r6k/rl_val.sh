#!/bin/bash
# r6k RL checkpoints (PREREG section 6): best gen8 on each seed's learning curve, frozen and run on r6k_val vs 4873.
# Learning-curve eval at iteration k scores params_{k+1}; picks: s20 iter 49, s21 iter 9, s22 iter 79, s23 iter 19.
set -euo pipefail
export EVOLVE_CACHE_DIR="$PWD/evolve/.cache_r6k"
for x in 20:50 21:10 22:80 23:20; do
  s=${x%%:*}; k=${x##*:}
  python -m evolve.rl_plan.es_validate --params "evolve/rl_plan/out/es/r6k_gen8_s$s/params_$k.npz" \
    --name "r6k_gen8_s${s}_$k" --stage r6k_val --workers "${WORKERS:-16}" --out "evolve/r6k/out/rl_val_s${s}_$k.json"
done
