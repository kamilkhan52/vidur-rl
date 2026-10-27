"""Frozen ES policy h1_gen8_s40_80 (evolve.rl_plan.es_validate): 4873 + the learned price correction in h1_gen8_s40_80.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "85b56cecb5dbf69bb17c6639791e4d4e6f266843b18f99c31c0807c578516ddb"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
