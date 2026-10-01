"""
Ablation study of a frozen evolved router: switch off one mechanism at a time (by
overriding its module-level constants) and re-score on a fixed scenario set against the
same-machine cached heuristic baselines.

    python -m evolve.analysis.ablate --policy evolve/finalists/claude_r1_76be082eb0f5.py \
        --out evolve/analysis/ablation_claude_r1.json --workers 12

Variants are constant overrides, so every variant is the finalist's own code with one
term zeroed; "core_only" keeps just load + miss price with the decode floor. Results are
per scenario and metric: P99 ratio to the per-scenario best heuristic (same as fitness).
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

from evolve import evaluate as ev

ZERO_GUARDS = {"IMBALANCE_SCALE": "1e30", "REL_IMBALANCE": "0.0", "CONC_SCALE": "0.0",
               "INTERFERENCE": "0.0", "KV_PENALTY": "0.0", "SHARED_PREFIX_FRAC": "0.0",
               "HYSTERESIS_FRAC": "0.0", "HYSTERESIS_MIN": "0.0"}

VARIANTS = {
    "full": {},
    "no_decode_floor": {"DECODE_MISS_PRICE": "0.0"},
    "no_queue_externality": {"MISS_EXTERNALITY": "0.0"},
    "miss_price_1": {"DECODE_MISS_PRICE": "0.0", "MISS_EXTERNALITY": "0.0"},  # pure least-work + guards
    "decode_floor_x2": {"DECODE_MISS_PRICE": "4.0"},
    "decode_floor_half": {"DECODE_MISS_PRICE": "1.0"},
    "no_decode_weight": {"DECODE_WEIGHT": "0.0"},
    "no_imbalance_guards": {"IMBALANCE_SCALE": "1e30", "REL_IMBALANCE": "0.0"},
    "no_concurrency_guard": {"CONC_SCALE": "0.0"},
    "no_interference_surcharge": {"INTERFERENCE": "0.0"},
    "no_kv_penalty": {"KV_PENALTY": "0.0"},
    "no_inflight_credit": {"SHARED_PREFIX_FRAC": "0.0"},
    "no_hysteresis": {"HYSTERESIS_FRAC": "0.0", "HYSTERESIS_MIN": "0.0"},
    "core_only": dict(ZERO_GUARDS),
}

# Search set + a spread of held-out scenarios (Synth knee, 16R, other perturbation).
SCENARIOS = ev.STAGE1 + [
    ev.Scenario("Synth", 8, 0.75, -1, 0, 1),
    ev.Scenario("Synth", 8, 1.0, -1, 0, 1),
    ev.Scenario("Tool", 16, 0.5, -1, 0, 1),
    ev.Scenario("Conv", 16, 0.5, -1, 0, 1),
]


def make_variant(src: str, overrides: dict) -> str:
    for name, value in overrides.items():
        src, n = re.subn(rf"^{name}\s*=\s*[^#\n]+", f"{name} = {value}  ", src, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"constant {name} not found")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    src = Path(a.policy).read_text()
    names = a.variants or list(VARIANTS)
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    base = ev.ensure_baselines(SCENARIOS, workers=a.workers)
    with tempfile.TemporaryDirectory(prefix="ablate_") as tmp:
        paths = {}
        for n in names:
            if n in results:
                continue
            p = Path(tmp) / f"{n}.py"
            p.write_text(make_variant(src, VARIANTS[n]))
            paths[n] = p
        jobs = [("evolved", sc, str(paths[n])) for n in paths for sc in SCENARIOS]
        runs = ev.run_many(jobs, a.workers)
    for i, n in enumerate(paths):
        rs = runs[i * len(SCENARIOS):(i + 1) * len(SCENARIOS)]
        bad = {sc.tag: r.get("reason") for sc, r in zip(SCENARIOS, rs) if not r.get("ok")}
        entry = {"overrides": VARIANTS[n], "failures": bad}
        if not bad:
            entry.update(ev.score(dict(zip(SCENARIOS, rs)), base))
            entry["runs"] = {sc.tag: {k: r[k] for k in ("p99_e2e", "p99_ttft", "p99_tpot", "mean_e2e",
                                                        "prefill_hit_ratio", "queue_delay_p99")}
                             for sc, r in zip(SCENARIOS, rs)}
        results[n] = entry
        out_path.write_text(json.dumps(results, indent=1))
    print(json.dumps({n: round(results[n].get("combined_score", float("nan")), 4) for n in results}, indent=1))


if __name__ == "__main__":
    main()
