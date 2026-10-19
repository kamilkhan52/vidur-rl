"""Frozen ES policy r6k_gen8_s22_80 (evolve.rl_plan.es_validate): 4873 + the learned price correction in r6k_gen8_s22_80.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "4ba5f6f6b2db8cbdbb8c77489d28946a9b68b4bf4ee3f02c80a5bab6d6c9e30e"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
