"""Frozen ES policy r6k_gen8_s20_50 (evolve.rl_plan.es_validate): 4873 + the learned price correction in r6k_gen8_s20_50.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "47e39c65d96d81bcb5f8e506afbfd54120d5a3d155135ed9ce253292b3188164"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
