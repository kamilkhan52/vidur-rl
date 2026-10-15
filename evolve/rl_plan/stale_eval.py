"""Do the ES policies keep their edge when the router sees stale engine metrics? (plan §11 A6, 2026-10-15)

A real router scrapes each engine's metrics (waiting, running, KV use) periodically, while counters it owns (requests
it routed and not yet finished, prefill tokens it routed and not yet prefilled) and its prefix index stay exact.
Vidur v4.1 models this with ``--evolved_global_scheduler_config_view_refresh_s``: the engine metrics a policy sees
are snapshots on a fixed grid of that period (a decision at t sees the snapshot at floor(t / period) * period).

Grid: 4873 and the frozen ES policies x refresh period {0 (exact), 0.1, 0.5, 2.0 s} on heldout3's nine
configurations at draws p44-p46 (never used for training or selection). Reported, with configuration-cluster CIs:
- each policy at each period vs 4873 at the same period (does the edge survive?);
- each policy at each period vs itself exact (how much does staleness cost?).

Results are cached per (policy, period, scenario) in ``<out>.cache.json`` so the job can resume.

    python -m evolve.rl_plan.stale_eval --workers 16 --out evolve/rl_plan/out/stale_eval.json
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan.es_validate import FINAL_DIR, compare

CONFIGS = [(t, R, s) for t in ("Tool", "Conv", "Synth") for (R, s) in ((8, 1.0), (8, 0.75), (16, 0.5))]
DRAWS = (44, 45, 46)
PERIODS = (0.0, 0.1, 0.5, 2.0)
POLICIES = {"4873": Path(c0.BASE), "es_lin9_20": FINAL_DIR / "es_lin9_20.py", "es_mlp9_10": FINAL_DIR / "es_mlp9_10.py"}
FLAG = "--evolved_global_scheduler_config_view_refresh_s"


def scenarios():
    return [ev.Scenario(t, R, s, -1, 0, d, model="l3_8b", jitter_s=ev.SYNTH_JITTER_S if t == "Synth" else -1.0)
            for (t, R, s) in CONFIGS for d in DRAWS]


def _job(args):
    name, period, sc, timeout = args
    extra = [FLAG, str(period)] if period > 0 else None
    return name, period, sc.tag, ev.run_one("evolved", sc, str(POLICIES[name].resolve()), timeout, extra=extra)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    scen = scenarios()
    by_tag = {sc.tag: sc for sc in scen}
    cache_f = Path(a.out + ".cache.json")
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    cv = ev.code_version()
    if cache and cache.get("code_version") != cv:
        raise SystemExit(f"{cache_f} was written by another code_version")
    cache["code_version"] = cv
    runs = cache.setdefault("runs", {})
    # exact (period 0) runs come from the shared baseline cache
    exact = ev.ensure_baselines(scen, [f"policy:{p.resolve()}" for p in POLICIES.values()], workers=a.workers,
                                timeout=a.timeout)
    res = {}
    for name, p in POLICIES.items():
        for sc in scen:
            res[(f"{name}@0.0", sc)] = exact[(f"policy:{p.resolve()}", sc)]
    todo = [(n, per, sc, a.timeout) for per in PERIODS if per > 0 for n in POLICIES for sc in scen
            if f"{n}@{per}|{sc.tag}" not in runs]
    with mp.get_context("forkserver").Pool(a.workers) as pool:
        for name, per, tag, r in pool.imap_unordered(_job, todo):
            if not r.get("ok"):
                raise RuntimeError(f"{name} period {per} on {tag}: {r.get('reason')}")
            runs[f"{name}@{per}|{tag}"] = r
            cache_f.write_text(json.dumps(cache))
    for key, r in runs.items():
        k, tag = key.split("|")
        res[(k, by_tag[tag])] = r
    out = {"code_version": cv, "draws": DRAWS, "periods": PERIODS, "vs_4873_same_period": {}, "vs_self_exact": {}}
    for per in PERIODS:
        for n in POLICIES:
            if n != "4873":
                out["vs_4873_same_period"][f"{n}@{per}"] = compare(f"{n}@{per}", f"4873@{per}", scen, res)
            if per > 0:
                out["vs_self_exact"][f"{n}@{per}"] = compare(f"{n}@{per}", f"{n}@0.0", scen, res)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for sec in ("vs_4873_same_period", "vs_self_exact"):
        print("==", sec)
        for k, r in out[sec].items():
            print(f"{k:16s} " + "  ".join(f"{m} {r[m]['geo']:.3f} [{r[m]['lo']:.3f}, {r[m]['hi']:.3f}]"
                                          for m in ("tails4", "means4", "gen8")))


if __name__ == "__main__":
    main()
