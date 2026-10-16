"""Choose the load gate for an ES policy: below the gate it falls back to plain 4873 (plan 11, 2026-10-16).

The gate is in mean outstanding requests per replica (router-owned, exact). Each gate value is written as a copy of
the parameter file with a ``gate`` entry, frozen under ``finalists/rl_es/gate_sweep/``, and run with 4873 on
validation scenarios: Tool/Conv/Synth x {8R x 1.0, 16R x 0.5, 24R x 0.333, 8R x 0.5, 16R x 0.75, 16R x 1.0} at the
learning-curve draws p40-p41, never used for training, heldout3, test3b or the final regime map.

Selection rule (fixed before the sweep): the gate with the best mean log gen8 ratio over the six configurations,
among gates whose gen8 is at most 1.0 in every configuration; if none qualifies, the gate with the smallest worst-
configuration gen8.

    python -m evolve.rl_plan.gate_sweep --params evolve/finalists/rl_es/es_gen8lin_20.npz --name es_gen8lin_20 \\
        --gates 1 2 4 8 16 32 --workers 8 --out evolve/rl_plan/out/gate_sweep_gen8.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan.es_validate import COMBINED, FINAL_DIR, METS, TEMPLATE

CONFIGS = [(8, 1.0), (16, 0.5), (24, 0.333), (8, 0.5), (16, 0.75), (16, 1.0)]
TRACES = ("Tool", "Conv", "Synth")
DRAWS = (40, 41)
SWEEP_DIR = FINAL_DIR / "gate_sweep"


def scenarios():
    return [ev.Scenario(t, R, s, -1, 0, d, model="l3_8b", jitter_s=ev.SYNTH_JITTER_S if t == "Synth" else -1.0)
            for t in TRACES for (R, s) in CONFIGS for d in DRAWS]


def write_gated(params: Path, name: str, gate: float) -> Path:
    """Freeze a copy of ``params`` with the gate set (gate 0 = ungated) as <dir>/<name>_g<gate>.{npz,py}."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(params) as f:
        P = {k: f[k] for k in f.files}
    P["gate"] = np.array(float(gate))
    stem = SWEEP_DIR / f"{name}_g{gate:g}"
    np.savez(stem.with_suffix(".npz"), **P)
    npz = stem.with_suffix(".npz")
    py = stem.with_suffix(".py")
    py.write_text(TEMPLATE.format(name=stem.name, sha=hashlib.sha256(npz.read_bytes()).hexdigest()))
    return py


def summarise(scen, res, pol, ref) -> dict:
    out = {}
    for (R, s) in CONFIGS:
        sel = [sc for sc in scen if sc.replicas == R and sc.scale == s]
        row = {}
        for name, ms in list(COMBINED.items()) + [(m, (m,)) for m in METS]:
            d = [np.mean([np.log(res[(pol, sc)][m] / res[(ref, sc)][m]) for m in ms]) for sc in sel]
            row[name] = float(np.exp(np.mean(d)))
        out[f"{R}:{s}"] = row
    return out


def choose_gate(table: dict) -> float:
    def mean_log(g):
        return float(np.mean([np.log(r["gen8"]) for r in table[g].values()]))

    ok = [g for g in table if all(r["gen8"] <= 1.0 for r in table[g].values())]
    if ok:
        return min(ok, key=mean_log)
    return min(table, key=lambda g: max(r["gen8"] for r in table[g].values()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gates", type=float, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=10800.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    scen = scenarios()
    ref = f"policy:{Path(c0.BASE).resolve()}"
    gates = [0.0] + [g for g in a.gates if g > 0]
    pols = {g: f"policy:{write_gated(Path(a.params), a.name, g).resolve()}" for g in gates}
    res = ev.ensure_baselines(scen, [ref] + list(pols.values()), workers=a.workers, timeout=a.timeout)
    table = {g: summarise(scen, res, p, ref) for g, p in pols.items()}
    pick = choose_gate(table)
    out = {"code_version": ev.code_version(), "draws": DRAWS, "params": a.params, "chosen_gate": pick,
           "table": {f"{g:g}": t for g, t in table.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for g, t in table.items():
        print(f"gate {g:5g}  " + "  ".join(f"{k} {r['gen8']:.3f}" for k, r in t.items()))
    print("chosen gate:", pick)


if __name__ == "__main__":
    main()
