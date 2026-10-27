"""EXP33: prior-work routers vs 4873 on one scenario set (evolve/priorwork/PREREG.md).

    python -m evolve.priorwork.compare --set h1 --workers 4 --out evolve/priorwork/out/h1.json

Every router runs on the same scenarios, code version and prediction bundle. Per router and metric: the geometric
mean of router / 4873 over runs with a 95 % cluster bootstrap over configurations (draws of one configuration
together; ``es_validate.compare``), plus goodput capacity at absolute TTFT/TPOT SLOs from the kept per-request
outputs (h1 PREREG amendment 1; EXP30's levels).
"""
import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("EVOLVE_CACHE_DIR", str(HERE / ".cache"))  # before evaluate is imported
os.environ.setdefault("EVOLVE_KEEP_DIR", str(HERE / "keep"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evolve import evaluate as ev  # noqa: E402
from evolve.rl_plan.es_validate import compare  # noqa: E402

REPO = ev.REPO
REF = "4873"
# name -> policy file (relative to the repo) or a built-in scheduler name
ROUTERS = {
    "4873": "evolve/finalists/v4/v4_487363424749.py",
    "es": "evolve/finalists/rl_es/es_gen8div_b0_40.py",
    "mooncake0": "evolve/policies/systems/mooncake_min_ttft.py",
    "mooncake_tuned": "evolve/policies/tuned/mooncake_min_ttft_tuned_best.py",
    "dynamo": "evolve/policies/systems/dynamo_kv_cost.py",
    "lor": "lor",
    "sticky_lor": "sticky_lor",
}
PRIOR = {}  # filled from evolve/priorwork/routers.json (name -> policy file) once the ports exist
_ROUTERS_JSON = HERE / "routers.json"
if _ROUTERS_JSON.exists():
    PRIOR = json.loads(_ROUTERS_JSON.read_text())

SLOS = {"2s_50ms": (2.0, 0.050), "5s_100ms": (5.0, 0.100)}  # (TTFT s, TPOT s); EXP30 / h1 amendment 1
ATTAIN = 0.90


def scenarios(name: str) -> list:
    if name == "h1":  # EXP31's test windows at heavy, high and moderate load (h1_test + h1_test_mod)
        return ev.STAGES["h1_test"] + ev.STAGES["h1_test_mod"]
    if name == "h1_smoke":  # small check: one Tool3 test window, 8R, heavy load, draw 1
        return [sc for sc in ev.STAGES["h1_test"] if sc.replicas == 8 and sc.trace == "Tool3"
                and sc.window_start == 14000 and sc.perturbation == 1 and sc.scale == ev.H1_LOADS_8R[0]]
    return ev.STAGES[name]


def ref_of(name: str) -> str:
    p = {**ROUTERS, **PRIOR}[name]
    return p if not p.endswith(".py") else f"policy:{(REPO / p).resolve()}"


def kept_csv(sc, ref: str) -> Path:
    if ref.startswith("policy:"):
        p = Path(ref[len("policy:"):])
        who = f"{p.stem}_{ev.file_sha(p)[:8]}"
    else:
        who = ref
    d = Path(os.environ["EVOLVE_KEEP_DIR"]) / ev.code_version() / sc.tag / who
    files = list(d.rglob("request_metrics.csv"))
    return files[0] if files else None


def attainment(sc, ref: str) -> dict:
    """Share of scored requests meeting each SLO (TTFT and TPOT both), same warm-up cut as run_one."""
    f = kept_csv(sc, ref)
    if f is None:
        return {}
    df = pd.read_csv(f)
    if sc.window_start >= 0 and sc.warmup_frac > 0:
        a = df["request_arrived_at"]
        df = df[a >= a.min() + sc.warmup_frac * (a.max() - a.min())]
    ttft = df["prefill_e2e_time"].to_numpy()
    n = df["request_num_decode_tokens"].to_numpy()
    tpot = np.where(n > 1, (df["request_e2e_time"].to_numpy() - ttft) / np.maximum(n - 1, 1), 0.0)
    return {k: float(((ttft <= t) & (tpot <= p)).mean()) for k, (t, p) in SLOS.items()}


def capacity(points) -> tuple:
    """points: [(rate, attainment)], rate = 1/scale. Highest rate with attainment >= ATTAIN, log-linear in rate
    between tested loads. Returns (value, kind) with kind 'interp', '>=' (meets at the highest rate) or '<' (fails
    at the lowest)."""
    pts = sorted(points)
    ok = [a >= ATTAIN for _, a in pts]
    if all(ok):
        return pts[-1][0], ">="
    if not ok[0]:
        return pts[0][0], "<"
    for (r0, a0), (r1, a1) in zip(pts, pts[1:]):
        if a0 >= ATTAIN > a1:
            x0, x1 = math.log(r0), math.log(r1)
            return math.exp(x0 + (a0 - ATTAIN) / (a0 - a1) * (x1 - x0)), "interp"
    return pts[0][0], "<"  # non-monotone: first crossing above already handled


def goodput(names, scen) -> dict:
    groups = {}
    for sc in scen:  # one capacity per (trace, window, replicas, draw) across its loads
        groups.setdefault((sc.trace, sc.window_start, sc.replicas, sc.perturbation), []).append(sc)
    out = {}
    for n in names:
        ref = ref_of(n)
        per = {}
        for key, scs in groups.items():
            att = {sc: attainment(sc, ref) for sc in scs}
            if not all(att.values()):
                continue
            for slo in SLOS:
                per.setdefault(slo, {})["/".join(map(str, key))] = capacity([(1 / sc.scale, att[sc][slo]) for sc in scs])
        out[n] = per
    rel = {}
    for n in names:
        for slo in SLOS:
            a, b = out[n].get(slo, {}), out[REF].get(slo, {})
            ks = [k for k in a if k in b]
            if ks:
                rel.setdefault(n, {})[slo] = {
                    "geo_ratio_vs_4873": float(np.exp(np.mean([np.log(a[k][0] / b[k][0]) for k in ks]))),
                    "n": len(ks), "bounds": sum(a[k][1] != "interp" or b[k][1] != "interp" for k in ks)}
    return {"per_group": out, "vs_4873": rel}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--routers", default="", help="comma list; default all known")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = [n for n in (a.routers.split(",") if a.routers else [*ROUTERS, *PRIOR]) if n]
    if REF not in names:
        names.insert(0, REF)
    scen = scenarios(a.set)
    refs = {n: ref_of(n) for n in names}
    res = ev.ensure_baselines(scen, list(refs.values()), workers=a.workers)
    by_name = {(n, sc): res[(refs[n], sc)] for n in names for sc in scen}
    report = {"set": a.set, "n_runs": len(scen), "code_version": ev.code_version(),
              "bundles": sorted({by_name[k].get("bundle", "?") for k in by_name}),
              "routers": {n: refs[n] for n in names}, "vs_4873": {}, "abs": {}}
    for n in names:
        report["vs_4873"][n] = compare(n, REF, scen, by_name)
        report["abs"][n] = {m: float(np.mean([by_name[(n, sc)][m] for sc in scen]))
                            for m in ("mean_e2e", "mean_ttft", "mean_tpot", "mean_tbt", "p50_e2e", "p99_e2e")}
    report["goodput"] = goodput(names, scen)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    for n in names:
        v = report["vs_4873"][n]
        print(f"{n:18s} " + "  ".join(f"{m} {v[m]['geo']:.3f} [{v[m]['lo']:.3f},{v[m]['hi']:.3f}]"
                                      for m in ("mean_e2e", "mean_ttft", "mean_tpot", "p99_e2e")))


if __name__ == "__main__":
    main()
