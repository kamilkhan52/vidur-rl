"""Tabulate ``run.py`` summaries and their ratios to a reference policy (default: 4873), per (window, scale, draw).

    python -m evolve.lss2.compare out/lss2/eval --ref v4_487363424749
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = ("p99_e2e", "mean_e2e", "p50_e2e", "p99_ttft", "mean_ttft", "p50_ttft", "p99_tpot", "mean_tpot",
           "p50_tpot", "p99_tbt", "mean_tbt", "p50_tbt")


def policy_name(s: dict) -> str:
    return Path(s["policy"]).stem if s["policy"].endswith(".py") else s["policy"]


def load(root: str):
    out = []
    for p in sorted(Path(root).rglob("summary.json")):
        s = json.loads(p.read_text())
        s["_policy"] = policy_name(s)
        s["_cell"] = (Path(s["window"]).name.rsplit("_", 1)[0], s["scale"], s["perturbation"], s.get("head", 0))
        out.append(s)
    return out


def table(sums, ref: str, metrics=METRICS):
    cells = defaultdict(dict)
    for s in sums:
        cells[s["_cell"]][s["_policy"]] = s
    rows = []
    for cell, by in sorted(cells.items()):
        r = by.get(ref)
        for pol, s in sorted(by.items()):
            row = {"cell": cell, "policy": pol, "ok": s.get("ok"), "hit": s.get("prefill_hit_ratio"),
                   "wall_s": s.get("wall_s")}
            for m in metrics:
                row[m] = s.get(m)
                if r is not None and r.get(m) and s.get(m) is not None:
                    row[f"r_{m}"] = s[m] / r[m]
            rows.append(row)
    return rows


def geo_mean_ratios(rows, metrics=METRICS):
    """Per policy: geometric mean over cells of the ratio to the reference."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in metrics:
            if f"r_{m}" in r and r[f"r_{m}"] and r[f"r_{m}"] > 0:
                acc[r["policy"]][m].append(math.log(r[f"r_{m}"]))
    return {p: {m: (math.exp(sum(v) / len(v)), len(v)) for m, v in d.items()} for p, d in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", default="v4_487363424749")
    ap.add_argument("--metrics", default="p99_e2e,mean_e2e,p50_e2e,p99_ttft,mean_ttft,p99_tpot,p99_tbt")
    a = ap.parse_args(argv)
    ms = tuple(a.metrics.split(","))
    rows = table(load(a.root), a.ref, ms)
    hdr = f"{'cell':44s} {'policy':18s} ok   hit   " + " ".join(f"{m:>10s}" for m in ms)
    print(hdr)
    for r in rows:
        vals = " ".join(f"{r[m]:10.3f}" if r.get(m) is not None else f"{'-':>10s}" for m in ms)
        hit = f"{r['hit']:.3f}" if r.get("hit") is not None and r["hit"] == r["hit"] else "  -  "
        print(f"{str(r['cell']):44s} {r['policy']:18s} {str(r['ok'])[0]}  {hit} {vals}")
    print(f"\nratios to {a.ref}:")
    for r in rows:
        if r["policy"] == a.ref:
            continue
        vals = " ".join(f"{r.get('r_' + m, float('nan')):10.3f}" for m in ms)
        print(f"{str(r['cell']):44s} {r['policy']:18s}          {vals}")
    print(f"\ngeometric mean ratio to {a.ref} over cells (n cells):")
    for p, d in sorted(geo_mean_ratios(rows, ms).items()):
        if p == a.ref:
            continue
        print(f"{p:18s} " + " ".join(f"{m}={d[m][0]:.3f}({d[m][1]})" for m in ms if m in d))


if __name__ == "__main__":
    main()
