"""h1 system throughput, reported next to the test and never judged (evolve/h1/PREREG.md amendment 1). Written before
the test is run.

For each (trace, window, replicas, draw) of h1_test + h1_test_mod and each router (the selected policy, 4873): the
goodput share at each load (heavy, high, moderate) = fraction of scored requests (after the 20 % warm-up) meeting both
TTFT <= T and TPOT <= P; goodput capacity = the highest load (arrival rate, requests/s) with share >= 0.9,
interpolated log-linearly in arrival rate between tested loads (outside the tested range: a bound); served tokens/s per
GPU at each load. SLO levels (EXP30): (2 s, 50 ms) and (5 s, 100 ms). TPOT = (E2E - TTFT) / (decode tokens - 1).

    python -m evolve.h1.goodput --keep <EVOLVE_KEEP_DIR> --out evolve/h1/out/h1_goodput.json
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SLOS = {"2s_50ms": (2.0, 0.05), "5s_100ms": (5.0, 0.10)}
SHARE = 0.9
WARMUP = 0.2
TAG = re.compile(r"^l3_8b-h100df-(\d+)R-(\w+)-s([\d.]+)-w(\d+)\+\d+-p(\d+)$")


def load_run(csv: Path) -> dict:
    df = pd.read_csv(csv)
    a = df["request_arrived_at"]
    df = df[a >= a.min() + WARMUP * (a.max() - a.min())]
    e2e, ttft, n = df["request_e2e_time"].to_numpy(), df["prefill_e2e_time"].to_numpy(), \
        df["request_num_decode_tokens"].to_numpy()
    tpot = np.where(n > 1, (e2e - ttft) / np.maximum(n - 1, 1), 0.0)
    span = float(df["request_arrived_at"].max() - df["request_arrived_at"].min())
    end = float((df["request_arrived_at"] + df["request_e2e_time"]).max() - df["request_arrived_at"].min())
    tokens = float((df["request_num_prefill_tokens"] + df["request_num_decode_tokens"]).sum())
    out = {"rate": len(df) / span, "tokens_per_s": tokens / end}
    for k, (t, p) in SLOS.items():
        out[f"share_{k}"] = float(np.mean((ttft <= t) & (tpot <= p)))
    return out


def capacity(points: list, key: str) -> dict:
    """points: [(rate, share)], any order. Highest rate with share >= SHARE, log-linear between neighbours."""
    pts = sorted(points)
    ok = [r for r, s in pts if s >= SHARE]
    if not ok:
        return {"value": pts[0][0], "bound": "below"}
    if all(s >= SHARE for _, s in pts):
        return {"value": pts[-1][0], "bound": "above"}
    for (r0, s0), (r1, s1) in zip(pts, pts[1:]):
        if s0 >= SHARE > s1:
            x = math.log(r0) + (s0 - SHARE) / (s0 - s1) * (math.log(r1) - math.log(r0))
            return {"value": math.exp(x), "bound": None}
    return {"value": max(ok), "bound": "non-monotone"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", required=True, help="EVOLVE_KEEP_DIR/<code_version> of the test runs")
    ap.add_argument("--policy", required=True, help="stem prefix of the selected policy's kept directory")
    ap.add_argument("--ref", default="v4_487363424749")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = defaultdict(dict)  # (trace, window, R, draw, who) -> {scale: metrics}
    for csv in Path(a.keep).glob("*/*/*/request_metrics.csv"):
        tag, who = csv.parts[-4], csv.parts[-3]
        m = TAG.match(tag)
        if not m or not (who.startswith(a.policy) or who.startswith(a.ref)):
            continue
        r, tr, sc, w, p = int(m[1]), m[2], float(m[3]), int(m[4]), int(m[5])
        runs[(tr, w, r, p, "policy" if who.startswith(a.policy) else "4873")][sc] = load_run(csv)
    rows = []
    for (tr, w, r, p, who), by_scale in sorted(runs.items()):
        row = {"trace": tr, "window": w, "replicas": r, "draw": p, "router": who,
               "tokens_per_s_per_gpu": {str(s): v["tokens_per_s"] / r for s, v in sorted(by_scale.items())}}
        for k in SLOS:
            row[f"capacity_{k}"] = capacity([(v["rate"], v[f"share_{k}"]) for v in by_scale.values()], k)
            row[f"share_{k}"] = {str(s): v[f"share_{k}"] for s, v in sorted(by_scale.items())}
        rows.append(row)
    ratios = {}
    for k in SLOS:
        pairs = defaultdict(dict)
        for row in rows:
            pairs[(row["trace"], row["window"], row["replicas"], row["draw"])][row["router"]] = row[f"capacity_{k}"]
        d = [v["policy"]["value"] / v["4873"]["value"] for v in pairs.values() if {"policy", "4873"} <= set(v)]
        ratios[k] = {"geo_policy_over_4873": float(np.exp(np.mean(np.log(d)))) if d else None, "n": len(d),
                     "censored": sum(1 for v in pairs.values() for x in v.values() if x["bound"])}
    Path(a.out).write_text(json.dumps({"slos": SLOS, "share": SHARE, "capacity_ratio": ratios, "rows": rows},
                                      indent=1))
    print(json.dumps(ratios, indent=1))


if __name__ == "__main__":
    main()
