#!/bin/bash
# Build the golden-set manifest on Alpine, to compare with the Mac's (evolve/golden/v3_<mac-host>.json).
# Prepared 2026-10-10; not yet submitted. Run from the Mac (one SSH multiplexed connection; no login-node compute).
#
# 1. The manifest needs the same prediction bundles as the Mac. Bundle keys used by the golden set (Mac, 2026-10-10):
#      7b 96bd5ae2116525e5, 70b d90ff80eaad08336, l3_8b 8e6e78e985b87b2a.
#    As of 10-09 Alpine's cache/ listed no d90ff80eaad08336 bundle: copy it (270 MB), or the build exits with code 2
#    (it refuses to train machine-specific forests unless --allow-train).
# 2. Sync evolve/analysis/golden.py and the policies, then submit with the generic CPU job script.
set -euo pipefail
R=${R:-/scratch/alpine/kamil@colostate.edu/vidur-rl-l3}
cd "$(dirname "$0")/.."
rsync -a cache/prediction_bundle_d90ff80eaad08336.pkl "alpine:$R/cache/"
rsync -a evolve/analysis/golden.py "alpine:$R/evolve/analysis/"
ssh alpine "cd $R && sbatch --cpus-per-task=8 --mem=48G --time=02:00:00 --job-name=golden_v3 \
  --export=ALL,REPO_DIR=$R,CMD='python -m evolve.analysis.golden build --workers 6 --out evolve/golden/v3_alpine.json' \
  scripts/submit_python_job.sbatch"
# Afterwards (from the Mac):
#   rsync -a alpine:$R/evolve/golden/v3_alpine.json evolve/golden/
#   .venv/bin/python -m evolve.analysis.golden check --manifest evolve/golden/v3_<mac-host>.json \
#       --against evolve/golden/v3_alpine.json
