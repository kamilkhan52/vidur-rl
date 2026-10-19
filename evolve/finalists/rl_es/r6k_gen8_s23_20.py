"""Frozen ES policy r6k_gen8_s23_20 (evolve.rl_plan.es_validate): 4873 + the learned price correction in r6k_gen8_s23_20.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "146beadde423b1e6e8b4d740ec8c1158d14c3adec00831d828d02db28aa8a05e"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
