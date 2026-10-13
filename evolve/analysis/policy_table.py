"""Where our policies sit among the routers, not only against the bar (Kamil, 2026-10-12).

The bar (``bar_eval``) is a per-scenario, per-metric envelope: in each scenario and for each metric, the best of the
bar's routers. No single router reaches it. This table reports, from the cache only (no simulation), for each of our
policies and each comparison:
- every bar member separately, and the envelope ("bar");
- the tails (P99 E2E, TTFT, TPOT, TBT) and the centre (mean and P50 of E2E and TTFT);
- the geometric-mean ratio ours / comparison over the stage's scenarios, its 95 % cluster-bootstrap CI over
  configurations (draws of one configuration resampled together; ``cluster_ci``), and wins (ratio < 1).
It also reports each bar member's own ratio to the envelope of the bar, which measures how demanding the envelope is.

  python -m evolve.analysis.policy_table --stage heldout3 --bar v4t \\
      --policy evolve/finalists/v4/v4_487363424749.py --policy evolve/finalists/l3_48a067a7df63.py \\
      --out evolve/analysis/out/policy_table_heldout3.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.analysis.cluster_ci import boot, clusters

METS = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e", "mean_ttft", "p50_e2e", "p50_ttft")


def short(b: str) -> str:
    return Path(b[len("policy:"):]).name.removesuffix(".py") if b.startswith("policy:") else b


def values(scheds, scenarios, cache) -> dict:
    """sched -> metric -> array over scenarios (in ``scenarios`` order); a missing or failed run is an error."""
    out = {}
    for s in scheds:
        rows = []
        for sc in scenarios:
            r = cache.get(ev.baseline_key(s, sc))
            if not r or not r.get("ok", True):
                raise KeyError(f"no cached result for {short(s)} on {sc.tag}: run bar_eval first")
            rows.append(r)
        out[s] = {m: np.array([float(r[m]) for r in rows]) for m in METS}
    return out


def compare(a: dict, b: dict, groups, n: int) -> dict:
    res = {}
    for m in METS:
        d = np.log(a[m] / b[m])
        g, lo, hi = boot(d, groups, n)
        res[m] = {"geo": round(float(g), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                  "wins": int((d < 0).sum()), "n": int(len(d))}
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="heldout3", choices=sorted(ev.STAGES))
    ap.add_argument("--bar", default="v4t", choices=sorted(ev.BARS))
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    scen = list(ev.STAGES[a.stage])
    bar = list(ev.BARS[a.bar])
    ours = [f"policy:{Path(p).resolve()}" for p in a.policy]
    cache = ev.load_baseline_cache()
    vals = values(bar + ours, scen, cache)
    env = {m: np.min([vals[b][m] for b in bar], axis=0) for m in METS}
    groups = clusters([sc.tag for sc in scen])
    out = {"stage": a.stage, "bar": a.bar, "bar_members": [short(b) for b in bar], "metrics": list(METS),
           "code_version": ev.code_version(), "n_scenarios": len(scen), "n_configurations": len(groups),
           "policies": {}, "bar_members_vs_envelope": {}}
    for p in ours:
        cmp = {"bar (envelope)": compare(vals[p], env, groups, a.boot)}
        for b in bar:
            cmp[short(b)] = compare(vals[p], vals[b], groups, a.boot)
        out["policies"][short(p)] = cmp
    for b in bar:
        out["bar_members_vs_envelope"][short(b)] = compare(vals[b], env, groups, a.boot)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    for name, cmp in out["policies"].items():
        print(f"\n{name} (ours / comparison; geo over {len(scen)} scenarios, wins)")
        print("comparison".ljust(34) + "".join(m.rjust(14) for m in METS))
        for c, r in cmp.items():
            print(c[:33].ljust(34) + "".join(f"{r[m]['geo']:.3f} ({r[m]['wins']:2d})".rjust(14) for m in METS))
    print("\nbar members / envelope")
    for b, r in out["bar_members_vs_envelope"].items():
        print(b[:33].ljust(34) + "".join(f"{r[m]['geo']:.3f}".rjust(14) for m in METS))
    return out


if __name__ == "__main__":
    main()
