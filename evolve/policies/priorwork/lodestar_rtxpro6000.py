"""Lodestar (evolve/priorwork/lodestar_core.py) with its offline rows from the r6k search windows (4R, high loads)
(evolve/priorwork/lodestar_collect.py --device rtxpro6000), trained at start-up and then online."""
import hashlib
import pickle
from pathlib import Path

from evolve.priorwork.lodestar_core import build

_ROWS = Path(__file__).resolve().parents[2] / "priorwork/data/lodestar_offline_rtxpro6000.pkl"
ROWS_SHA256 = "6d99e623641ddc4b2b48176b6d4e6de88a36098d00a998d351165b5213c7f662"
_b = _ROWS.read_bytes()
if hashlib.sha256(_b).hexdigest() != ROWS_SHA256:
    raise RuntimeError(f"{_ROWS} is not the frozen offline row file")
choose, observe, _inst = build(pickle.loads(_b))
