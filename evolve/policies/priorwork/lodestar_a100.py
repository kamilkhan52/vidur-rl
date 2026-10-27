"""Lodestar (evolve/priorwork/lodestar_core.py) with its offline rows from search3b (Tool and Conv full traces, 8R,
load 0.75, draw 1; evolve/priorwork/lodestar_collect.py --device a100), trained at start-up and then online."""
import hashlib
import pickle
from pathlib import Path

from evolve.priorwork.lodestar_core import build

_ROWS = Path(__file__).resolve().parents[2] / "priorwork/data/lodestar_offline_a100.pkl"
ROWS_SHA256 = "1329a8c0ac12b7a6486e96bd1f18c1a8ec24094ee8082ad98e2a05c2fafbe5a8"
_b = _ROWS.read_bytes()
if hashlib.sha256(_b).hexdigest() != ROWS_SHA256:
    raise RuntimeError(f"{_ROWS} is not the frozen offline row file")
choose, observe, _inst = build(pickle.loads(_b))
