"""Where does a 4-replica rtxpro6000 cluster saturate? Time-scale sweep on windows outside the hardware test windows
(Tool 4000+2000 and Conv 0+2000 are held out). Result cached like any baseline run; prints mean queueing delay and
mean E2E per router and scale.

    python -m evolve.r6k.probe_load --workers 8
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evolve import evaluate as ev

REPO = ev.REPO
WINDOWS = {"Tool": 8000, "Conv": 2000}  # 2000-request windows clear of the held-out ranges
SCALES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.6)
ROUTERS = ("round_robin", "lor", f"policy:{REPO / 'evolve/finalists/v4/v4_487363424749.py'}",
           f"policy:{REPO / 'evolve/policies/tuned/mooncake_min_ttft_tuned_best.py'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--out", default="evolve/r6k/out/probe_load.json")
    a = ap.parse_args()
    scen = [ev.Scenario(t, a.replicas, s, w, 2000, a.draw, model="l3_8b", device="rtxpro6000")
            for t, w in WINDOWS.items() for s in SCALES]
    ev.ensure_baselines(scen, list(ROUTERS), workers=a.workers, timeout=7200)
    cache = ev.load_baseline_cache()
    rows = []
    for sc in scen:
        for b in ROUTERS:
            r = cache.get(ev.baseline_key(b, sc))
            name = Path(b.split(":", 1)[1]).stem if b.startswith("policy:") else b
            rows.append({"trace": sc.trace, "scale": sc.scale, "router": name,
                         **{k: r.get(k) for k in ("queue_delay_mean", "mean_e2e", "mean_ttft", "p99_e2e")}} if r
                        else {"trace": sc.trace, "scale": sc.scale, "router": name, "missing": True})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"replicas": a.replicas, "draw": a.draw, "windows": WINDOWS, "rows": rows}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
