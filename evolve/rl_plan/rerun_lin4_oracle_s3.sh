#!/bin/bash
# EXP32 (a): rerun of es/lin4_oracle_s3 (2026-10-14, Alpine 34129012), which never read true lengths (--oracle bug,
# fixed in 90f68c2). Arguments copied from its meta.json; only the output directory differs.
set -euo pipefail
python -m evolve.rl_plan.rl_es --mode train --out evolve/rl_plan/out/es/lin4_oracle_s3_fixed --seed 3 --workers 12 \
  --hours 11.5 --configs Tool:8:0.75,Conv:8:0.75,Tool:8:1.0,Conv:8:1.0 --pairs 6 --windows 4 --length 4000 \
  --start-grid 0 --sigma 0.05 --lr 0.05 --wd 0.0 --opt sgd --arch linear --oracle --eval-every 10 --eval-len 4000 \
  --probe 1,3,10 --probe-draws 22,23,24,25 --fitness tails4
