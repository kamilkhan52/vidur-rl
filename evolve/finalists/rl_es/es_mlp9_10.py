"""Frozen ES policy es_mlp9_10 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_mlp9_10.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "b7fbb02e6234c06a451265d7c34b0f6bfa9f85ca00445ba049267a754036067d"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
