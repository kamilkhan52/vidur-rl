"""Frozen ES policy h1_gen8_s43_120 (evolve.rl_plan.es_validate): 4873 + the learned price correction in h1_gen8_s43_120.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "8583f85b6cab04a2ee95ea5f97e10f404b7eee2506dc58227fb2d1b42703e29c"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
