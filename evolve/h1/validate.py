"""h1 checkpoint choice, validation, selection and the one test (evolve/h1/PREREG.md section 4), written before any
h1 result.

    python -m evolve.h1.validate val        # best learning-curve gen8 checkpoint per seed, each vs 4873 on h1_val
    python -m evolve.h1.validate select     # the declared selection -> out/h1_selection.json (frozen)
    python -m evolve.h1.validate test       # the selected policy on h1_test (+ h1_test_mod), the pass rule, once
    python -m evolve.h1.validate replicate  # every seed on h1_test, descriptive; needs the frozen selection
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("EVOLVE_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache_h1"))  # before evaluate
from evolve import evaluate as ev  # noqa: E402
from evolve.rl_plan import es_validate  # noqa: E402

SEEDS = (40, 41, 42, 43)
ES_DIR = ev.REPO / "evolve/rl_plan/out/es"
OUT = ev.REPO / "evolve/h1/out"
SELECTION = OUT / "h1_selection.json"
GUARD = ("mean_e2e", "mean_ttft", "mean_tpot", "mean_tbt", "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt")


def checkpoint(seed: int) -> dict:
    """Best gen8 on the seed's learning curve; the evaluation at iteration k scores params_{k+1}."""
    run = ES_DIR / f"h1_gen8_s{seed}"
    evals = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines() if '"eval"' in line]
    evals = [e for e in evals if "eval" in e]
    if not evals:
        raise SystemExit(f"{run}: no learning-curve evaluation yet")
    best = min(evals, key=lambda e: e["eval"]["gen8"]["ratio"])
    k = best["iter"] + 1
    return {"seed": seed, "params": str(run / f"params_{k}.npz"), "name": f"h1_gen8_s{seed}_{k}",
            "lc_iter": best["iter"], "lc_gen8": best["eval"]["gen8"], "lc_mean_e2e": best["eval"]["mean_e2e"],
            "lc_points": len(evals)}


def run_stage(ck: dict, stage: str, workers: int) -> dict:
    out = OUT / f"{ck['name']}_{stage}.json"
    if not out.exists():
        es_validate.main(["--params", ck["params"], "--name", ck["name"], "--stage", stage, "--workers", str(workers),
                          "--out", str(out)])
    return json.loads(out.read_text())


def passes_test(v: dict) -> dict:
    """PREREG section 4: all four conditions."""
    r = {"mean_e2e_upper_below_1": v["mean_e2e"]["hi"] < 1.0,
         "gen8_upper_below_1": v["gen8"]["hi"] < 1.0,
         "no_guard_metric_worse": all(v[m]["lo"] <= 1.0 for m in GUARD),
         "p50_e2e_ttft_not_worse": v["p50_e2e"]["lo"] <= 1.0 and v["p50_ttft"]["lo"] <= 1.0}
    r["pass"] = all(r.values())
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("val", "select", "test", "replicate"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 30)))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "val":
        rows = []
        for s in SEEDS:
            ck = checkpoint(s)
            v = run_stage(ck, "h1_val", a.workers)["vs_4873"]
            rows.append({**ck, "val_gen8": v["gen8"], "val_mean_e2e": v["mean_e2e"], "val_pass": v["gen8"]["hi"] < 1.0})
        (OUT / "h1_val_summary.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            g = r["val_gen8"]
            print(f"s{r['seed']} it{r['lc_iter']}: val gen8 {g['geo']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] "
                  f"mean E2E {r['val_mean_e2e']['geo']:.4f} pass {r['val_pass']}")
    elif a.step == "select":
        if SELECTION.exists():
            raise SystemExit(f"{SELECTION} is frozen")
        rows = json.loads((OUT / "h1_val_summary.json").read_text())
        passing = [r for r in rows if r["val_pass"]]
        sel = min(passing, key=lambda r: r["val_gen8"]["geo"]) if passing else None
        SELECTION.write_text(json.dumps({"selected": sel, "k_pass": len(passing), "n_seeds": len(rows),
                                         "code_version": ev.code_version()}, indent=1))
        print("selected:", sel and sel["name"], f"({len(passing)} of {len(rows)} seeds pass validation)")
    elif a.step == "test":
        sel = json.loads(SELECTION.read_text())["selected"]
        if sel is None:
            raise SystemExit("no seed passed validation: the method finds no policy (PREREG section 4)")
        os.environ["EVOLVE_KEEP_DIR"] = str(OUT / "keep")  # amendment 1: request-level outputs for goodput
        v = run_stage(sel, "h1_test", a.workers)["vs_4873"]
        mod = run_stage(sel, "h1_test_mod", a.workers)["vs_4873"]
        verdict = {"selected": sel["name"], "test": v, "rule": passes_test(v), "moderate_reported": mod}
        from evolve.h1 import goodput  # reported, never judged (amendment 1)
        goodput.main(["--keep", str(OUT / "keep" / ev.code_version()), "--policy", sel["name"],
                      "--out", str(OUT / "h1_goodput.json")])
        (OUT / "h1_test_verdict.json").write_text(json.dumps(verdict, indent=1))
        print(json.dumps(verdict["rule"], indent=1))
    else:
        if not SELECTION.exists():
            raise SystemExit("freeze the selection first")
        rows = []
        for s in SEEDS:
            ck = checkpoint(s)
            v = run_stage(ck, "h1_test", a.workers)["vs_4873"]
            rows.append({"name": ck["name"], "test_mean_e2e": v["mean_e2e"], "test_gen8": v["gen8"],
                         "rule": passes_test(v)})
        (OUT / "h1_test_replication.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(r["name"], f"mean E2E {r['test_mean_e2e']['geo']:.4f}", "pass", r["rule"]["pass"])


if __name__ == "__main__":
    main()
