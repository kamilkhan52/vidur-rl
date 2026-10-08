"""Robustness of the ratio-to-bar to Vidur's execution-time model: re-run the bar and the given policies with a
perturbed predictor and report each policy's P99 ratio to the bar. Variants:
  crude          random forest restricted to depth 4 / 50 trees (the default grid-searches up to 32 / 750)
  reseed         default grid, forest trained with another random seed (training noise of the predictor)
Each variant trains into its own cache directory: Vidur's model and bundle cache keys ignore the grid-search
parameters and the seed, so with the shared cache a variant silently reuses the default forest (2026-10-08: the
first crude run did exactly that). The attention batching-overhead knobs are not a usable variant: Vidur applies
them to grouped-query-attention models only, and Llama-2-7B is not one.
  linear         linear-regression predictor; UNUSABLE here (2026-10-08: its polynomial fits extrapolate to
                 P99 E2E ~1e8 s for every policy), kept only to document that

Results go to their own cache file (the baseline cache key does not include the predictor), so they never mix
with default-predictor runs. Run all of it on one machine: each predictor trains its own bundle per machine.

  python -m evolve.analysis.predictor_sens --variant crude --policy evolve/finalists/tail4_ni.py --workers 10
"""
import argparse
import json
import math
from pathlib import Path

import evolve.evaluate as ev
from evolve.analysis.bar_eval import BAR

RF = "--random_forest_execution_time_predictor_config_"
VARIANTS = {
    "crude": [RF + "max_depth", "4", RF + "num_estimators", "50", RF + "cache_dir", "cache/predictor_crude"],
    "reseed": [RF + "seed", "7", RF + "cache_dir", "cache/predictor_reseed"],
    "linear": ["--execution_time_predictor_config_type", "linear_regression"],
}
SCEN = [ev.Scenario(t, 8, sc, -1, 0, p) for t in ("Tool", "Conv") for sc in (1.0, 0.75) for p in (2, 3)]
METRICS = ("e2e", "ttft", "tpot", "tbt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    a = ap.parse_args()
    EXTRA = VARIANTS[a.variant]
    out = f"evolve/analysis/out/predictor_sens_{a.variant}.json"
    cache_path = ev.CACHE_DIR / ("predictor_sens_lr.json" if a.variant == "linear" else f"predictor_sens_{a.variant}.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    runs = [(b, None) for b in BAR] + [("evolved", str(Path(p).resolve())) for p in a.policy]

    def key(name, pol, sc):
        who = name if pol is None and not name.startswith("policy:") else ev.file_sha(Path(pol or name[7:]))[:12]
        return f"{a.variant}2|{ev.code_version()}|{sc.tag}|{who}"

    todo = []
    for name, pol in runs:
        if name.startswith("policy:"):
            name, pol = "evolved", name[7:]
        for sc in SCEN:
            if key(name, pol, sc) not in cache:
                todo.append((name, pol, sc))
    # warm the predictor cache with one run first so parallel workers do not all train it
    if todo:
        first, rest = todo[:1], todo[1:]
        for batch in (first, rest):
            if not batch:
                continue
            import concurrent.futures as cf
            with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
                res = list(ex.map(lambda j: ev.run_one(j[0], j[2], j[1], 3600 * 3, None, EXTRA), batch))
            for (name, pol, sc), r in zip(batch, res):
                if not r.get("ok"):
                    raise RuntimeError(f"{name} {pol} {sc.tag}: {r}")
                cache[key(name, pol, sc)] = r
            cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))

    def get(name, pol, sc):
        if name.startswith("policy:"):
            name, pol = "evolved", name[7:]
        return cache[key(name, pol, sc)]

    report = {}
    for p in a.policy:
        rows = {sc.tag: {m: get("evolved", str(Path(p).resolve()), sc)[f"p99_{m}"]
                         / min(get(b, None, sc)[f"p99_{m}"] for b in BAR) for m in METRICS} for sc in SCEN}
        geo = {m: math.exp(sum(math.log(r[m]) for r in rows.values()) / len(rows)) for m in METRICS}
        report[Path(p).name] = {"geo": geo, "per_scenario": rows}
        print(Path(p).name, " ".join(f"{m} {geo[m]:.3f} ({sum(r[m] < 1 for r in rows.values())}/{len(rows)})" for m in METRICS))
    Path(out).write_text(json.dumps({"predictor": a.variant, "extra": EXTRA, "bar": BAR, "policies": report}, indent=1))


if __name__ == "__main__":
    main()
