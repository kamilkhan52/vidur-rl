"""Frozen ES policy es_gen8div_b0_40 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_gen8div_b0_40.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "d192c9acb4f2bf87fa846aea5c38135119e0affc26e6644c13c0c2ffafea268f"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
