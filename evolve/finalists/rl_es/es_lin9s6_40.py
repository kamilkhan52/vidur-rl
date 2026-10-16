"""Frozen ES policy es_lin9s6_40 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_lin9s6_40.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "e42ce1ff2321f4b28bb3ababd53523fe82b81e85076ba6a72d7926b477948b0c"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
