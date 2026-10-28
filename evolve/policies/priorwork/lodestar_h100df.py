"""Lodestar (evolve/priorwork/lodestar_core.py) with its offline rows from the h1 training windows
(evolve/priorwork/lodestar_collect.py --device h100df), trained at start-up and then online."""
import hashlib
import pickle
from pathlib import Path

from evolve.priorwork.lodestar_core import build

_ROWS = Path(__file__).resolve().parents[2] / "priorwork/data/lodestar_offline_h100df.pkl"
ROWS_SHA256 = "d4301a3e4e53b57c6bc0a352a3857f5937ccb6f60709823f87d9e655bf62a963"
_b = _ROWS.read_bytes()
if hashlib.sha256(_b).hexdigest() != ROWS_SHA256:
    raise RuntimeError(f"{_ROWS} is not the frozen offline row file")
choose, observe, _inst = build(pickle.loads(_b))
