"""Lodestar (evolve/priorwork/lodestar_core.py) with its offline rows from the r6k search windows (4R, high loads)
(evolve/priorwork/lodestar_collect.py --device rtxpro6000), trained at start-up and then online."""
import hashlib
import pickle
from pathlib import Path

from evolve.priorwork.lodestar_core import build

_ROWS = Path(__file__).resolve().parents[2] / "priorwork/data/lodestar_offline_rtxpro6000.pkl"
ROWS_SHA256 = "28df757978b0048fd37dd558240e2b88f450d0fac56de1636875a214fe653e4a"
_b = _ROWS.read_bytes()
if hashlib.sha256(_b).hexdigest() != ROWS_SHA256:
    raise RuntimeError(f"{_ROWS} is not the frozen offline row file")
choose, observe, _inst = build(pickle.loads(_b))
