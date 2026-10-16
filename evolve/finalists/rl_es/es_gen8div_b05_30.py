"""Frozen ES policy es_gen8div_b05_30 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_gen8div_b05_30.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "3c876327d01658f51e45e2bf180b5bc67898f09b45f69274b41262ae839974c3"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
