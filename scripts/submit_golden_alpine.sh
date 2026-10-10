#!/bin/bash
# Build the v3-data golden-set manifest on Alpine, to compare with the Mac's (evolve/golden/v3_<mac-host>.json).
# Prepared 2026-10-10. Run from the Mac (one SSH multiplexed connection; no login-node compute).
#
# 1. The manifest needs the same prediction bundles as the Mac. Bundle keys used by the golden set (Mac, 2026-10-10):
#      7b 96bd5ae2116525e5, 70b d90ff80eaad08336, l3_8b 8e6e78e985b87b2a.
#    As of 10-09 Alpine's cache/ listed no d90ff80eaad08336 bundle: copy it (270 MB), or the build exits with code 2
#    (it refuses to train machine-specific forests unless --allow-train).
# 2. The simulator (vidur/), the harness (evolve/evaluate.py) and the policy files must match the Mac's commit:
#    update the Alpine checkout with git first. This script syncs only golden.py, the golden test policies and the
#    bundle; any remaining drift shows up as env diffs (code_version, policies) and differing hashes.
# 3. EVOLVE_V3_DATA=1 selects the v3 traces (without it the build runs on v4 data and the default name is v4_<host>);
#    EVOLVE_KEEP_DIR is cleared because the build refuses to run with it set. The build refuses to overwrite an
#    existing v3_alpine.json: remove it on Alpine first (or add --force) when rebuilding.
set -euo pipefail
R=${R:-/scratch/alpine/kamil@colostate.edu/vidur-rl-l3}
cd "$(dirname "$0")/.."
rsync -a cache/prediction_bundle_d90ff80eaad08336.pkl "alpine:$R/cache/"
rsync -a evolve/analysis/golden.py "alpine:$R/evolve/analysis/"
rsync -a evolve/golden/policies "alpine:$R/evolve/golden/"
ssh alpine "cd $R && sbatch --cpus-per-task=8 --mem=48G --time=02:00:00 --job-name=golden_v3 \
  --export=ALL,REPO_DIR=$R,EVOLVE_V3_DATA=1,EVOLVE_KEEP_DIR=,CMD='python -m evolve.analysis.golden build --workers 6 --out evolve/golden/v3_alpine.json' \
  scripts/submit_python_job.sbatch"
# Afterwards (from the Mac):
#   rsync -a alpine:$R/evolve/golden/v3_alpine.json evolve/golden/
#   .venv/bin/python -m evolve.analysis.golden check --manifest evolve/golden/v3_<mac-host>.json \
#       --against evolve/golden/v3_alpine.json
