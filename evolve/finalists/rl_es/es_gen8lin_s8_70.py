"""Frozen ES policy es_gen8lin_s8_70 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_gen8lin_s8_70.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "fbe285be0725c604688a000394fa0519cb0c60254a5cea85bae2f4ddaea42042"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
