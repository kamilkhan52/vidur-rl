"""Frozen ES policy es_lin9s5_50 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_lin9s5_50.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "a885b2e32fd5db9d8de8b76746ebfafd15d2458ce9ad4c191cbd34e2782b0e5d"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
