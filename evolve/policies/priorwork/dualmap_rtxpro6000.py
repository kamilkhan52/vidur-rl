"""DualMap (dualmap.py) on the rtxpro6000 deployment: the same rule with this device's idle prefill time per uncached token
(evolve/priorwork/calib_prefill.py). dualmap.py is executed in this module's namespace, so its functions read the
PREFILL_TPOT set below; its sha256 is checked so this file's hash (the run cache key) also pins the rule."""
import hashlib
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "dualmap.py"
SRC_SHA256 = "6ca9b933aa2d318c49b144827afb8fb78df10c465494843c4e72a94c91e468fd"
_b = _SRC.read_bytes()
if hashlib.sha256(_b).hexdigest() != SRC_SHA256:
    raise RuntimeError(f"{_SRC} is not the frozen DualMap port")
exec(compile(_b, str(_SRC), "exec"))
PREFILL_TPOT = 6.732e-05  # rtxpro6000 (calib_prefill.py)
