"""Frozen ES policy h1_gen8_s41_100 (evolve.rl_plan.es_validate): 4873 + the learned price correction in h1_gen8_s41_100.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "95d6c94267b693eb1bb2fb06b218c69aab4df7f6ae830f218940bac84fcdde06"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
