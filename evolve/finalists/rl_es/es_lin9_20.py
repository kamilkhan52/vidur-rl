"""Frozen ES policy es_lin9_20 (evolve.rl_plan.es_validate): 4873 + the learned price correction in es_lin9_20.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "c624f1ee8efe5057b63da39f6aec2a7b548331b23b62551248f2280f8f2ea209"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
