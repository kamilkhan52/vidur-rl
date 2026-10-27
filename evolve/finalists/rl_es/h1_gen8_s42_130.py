"""Frozen ES policy h1_gen8_s42_130 (evolve.rl_plan.es_validate): 4873 + the learned price correction in h1_gen8_s42_130.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "6964ccb0eed93cb005c6aecec47e14352d39f73d35282fe426b2dd07a0197f95"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
