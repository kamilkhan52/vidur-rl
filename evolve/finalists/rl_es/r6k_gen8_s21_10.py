"""Frozen ES policy r6k_gen8_s21_10 (evolve.rl_plan.es_validate): 4873 + the learned price correction in r6k_gen8_s21_10.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "be0064596df90024056068df2ef52f9b4e0652bfa2df4c770962614203745535"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{_P} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
