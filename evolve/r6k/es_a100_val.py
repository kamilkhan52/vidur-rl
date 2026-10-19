"""Descriptive (not pre-registered): the A100 RL router es_gen8div_b0_40 vs 4873 on r6k_val, scored like es_validate.

  python -m evolve.r6k.es_a100_val --workers 16 --out evolve/r6k/out/es_gen8div_b0_40_r6k_val.json
"""
import argparse
import json
from pathlib import Path

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan.es_validate import compare

POLICY = ev.REPO / "evolve/finalists/rl_es/es_gen8div_b0_40.py"

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
scen = list(ev.STAGES["r6k_val"])
pol, ref = f"policy:{POLICY.resolve()}", f"policy:{Path(c0.BASE).resolve()}"
res = ev.ensure_baselines(scen, [ref, pol], workers=a.workers)
out = {"name": POLICY.stem, "stage": "r6k_val", "descriptive": True, "code_version": ev.code_version(),
       "vs_4873": compare(pol, ref, scen, res)}
Path(a.out).write_text(json.dumps(out, indent=1))
for m, r in out["vs_4873"].items():
    if isinstance(r, dict):
        print(f"{m:10s} {r['geo']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] wins {r['wins']}/{r['n']}")
