"""Frozen ES policy es_gen8lin_20 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_gen8lin_20.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "0f0f6015cef18a689712448185bae1c77ba06e679ea5c821de975b0e9f7fcb55"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
