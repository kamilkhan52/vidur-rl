"""Frozen ES policy es_gen8lin_s9_20 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_gen8lin_s9_20.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "32ae09b4e8c1ab4cb9bb331048a38809a15efba35275bb2dec4b16883d9a450c"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
