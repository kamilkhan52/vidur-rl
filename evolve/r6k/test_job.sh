#!/bin/bash
# r6k_test (PREREG amendment 3), run once on Alpine through scripts/submit_python_job.sbatch with
# CMD="bash evolve/r6k/test_job.sh" (sbatch --export splits on commas, so the command lives here).
set -euo pipefail
export EVOLVE_CACHE_DIR="$PWD/evolve/.cache_r6k"
python -c "from evolve import evaluate as ev; v = ev.code_version(); print('code_version', v); assert v == 'c4557954b77b'"
python -m evolve.r6k.test_run --workers "${WORKERS:-12}" --out evolve/r6k/out/r6k_test.json
