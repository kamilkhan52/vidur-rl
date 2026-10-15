"""Freeze an ES checkpoint as a policy file and validate it against 4873 (2026-10-14; the "working RL policy" test).

Freeze: ``evolve/finalists/rl_es/<name>.npz`` (the parameters) and ``<name>.py`` (loads them; records their sha256,
so the baseline cache, which keys a policy by its file's contents, cannot mix two parameter sets).

Validate: the frozen policy and 4873 on a stage (default heldout3: 27 scenarios, 9 configurations; never used in ES
training or checkpoint selection), through ``evaluate.ensure_baselines`` (same simulator, bundle and cache as every
other result). Reported per metric: geometric-mean ratio policy / 4873 with a 95 % cluster-bootstrap CI over
configurations (``cluster_ci``), and wins. The pre-declared success test (plan §10, 2026-10-14 night): the four-tail
geometric mean (P99 E2E, TTFT, TPOT, TBT) below 1 with the CI excluding 1, and no single P99's CI above 1.

    python -m evolve.rl_plan.es_validate --params evolve/rl_plan/out/es/lin9_s2/params_40.npz --name es_lin9_40 \\
        --workers 12 --out evolve/rl_plan/out/es_validate_es_lin9_40.json
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import boot, clusters
from evolve.rl_plan import c0

FINAL_DIR = ev.REPO / "evolve" / "finalists" / "rl_es"
TAILS4 = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt")
METS = TAILS4 + ("mean_e2e", "mean_ttft", "p50_e2e", "p50_ttft")
TEMPLATE = '''"""Frozen ES policy {name} (evolve.rl_plan.es_validate): 4873 + the learned price correction in {name}.npz."""
import hashlib
from pathlib import Path

from evolve.rl_plan.rl_es import build_policy as _build

_P = Path(__file__).with_suffix(".npz")
PARAMS_SHA256 = "{sha}"
if hashlib.sha256(_P.read_bytes()).hexdigest() != PARAMS_SHA256:
    raise RuntimeError(f"{{_P}} is not the frozen parameter file")
choose, observe = _build(str(_P), None)
'''


def freeze(params: Path, name: str) -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    npz = FINAL_DIR / f"{name}.npz"
    if npz.exists() and npz.read_bytes() != Path(params).read_bytes():
        raise SystemExit(f"{npz} exists with other parameters")
    shutil.copyfile(params, npz)
    py = FINAL_DIR / f"{name}.py"
    py.write_text(TEMPLATE.format(name=name, sha=hashlib.sha256(npz.read_bytes()).hexdigest()))
    return py


def compare(pol: str, ref: str, scen: list, res: dict, n_boot: int = 10000) -> dict:
    groups = clusters([sc.tag for sc in scen])
    out = {}
    for m in METS:
        d = np.log(np.array([res[(pol, sc)][m] / res[(ref, sc)][m] for sc in scen], float))
        g, lo, hi = boot(d, groups, n_boot)
        out[m] = {"geo": float(g), "lo": float(lo), "hi": float(hi), "wins": int((d < 0).sum()), "n": len(d)}
    d = np.array([np.mean([np.log(res[(pol, sc)][m] / res[(ref, sc)][m]) for m in TAILS4]) for sc in scen])
    g, lo, hi = boot(d, groups, n_boot)
    out["tails4"] = {"geo": float(g), "lo": float(lo), "hi": float(hi), "wins": int((d < 0).sum()), "n": len(d)}
    out["success"] = bool(hi < 1.0 and all(out[m]["lo"] <= 1.0 for m in TAILS4))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", default="heldout3", choices=sorted(ev.STAGES))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-simulation limit, seconds (70B runs need more)")
    a = ap.parse_args(argv)
    py = freeze(Path(a.params), a.name)
    scen = list(ev.STAGES[a.stage])
    pol, ref = f"policy:{py.resolve()}", f"policy:{Path(c0.BASE).resolve()}"
    res = ev.ensure_baselines(scen, [ref, pol], workers=a.workers, timeout=a.timeout)
    out = {"name": a.name, "params": a.params, "stage": a.stage, "code_version": ev.code_version(),
           "vs_4873": compare(pol, ref, scen, res),
           "per_scenario": {sc.tag: {m: res[(pol, sc)][m] / res[(ref, sc)][m] for m in TAILS4} for sc in scen}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for m, r in out["vs_4873"].items():
        if isinstance(r, dict):
            print(f"{m:10s} {r['geo']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] wins {r['wins']}/{r['n']}")
    print("success:", out["vs_4873"]["success"])


if __name__ == "__main__":
    main()
