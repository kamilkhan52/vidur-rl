"""A11 spike experiments: exactness of forks, counterfactual divergence, fork and rollout costs."""

import argparse
import filecmp
import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evolve import evaluate as ev
from evolve.sim_fork import SimHandle, build_simulator, override_next_decision

POLICY = str(ev.REPO / "evolve/finalists/l3_48a067a7df63.py")


def argv_for(sc, out_dir, policy=POLICY):
    cmd = ev.sim_command("evolved", policy, sc, ev.window_csv(sc), Path(out_dir))
    return cmd[3:]  # drop "python -m vidur.main"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes


def outputs(d):
    f = list(Path(d).rglob("request_metrics.csv"))
    assert f, d
    g = f[0].parent / "token_gaps.npz"
    return f[0], g


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def exactness(sc, k, tmp):
    res = {"scenario": sc.tag, "k": k}
    t0 = time.perf_counter()
    a = (build_simulator(argv_for(sc, f"{tmp}/A")))
    a.run()
    a.write_output()
    res["unforked_s"] = time.perf_counter() - t0
    res["decisions_total"] = a.decisions
    del a

    t0 = time.perf_counter()
    o = (build_simulator(argv_for(sc, f"{tmp}/C")))
    assert o.run_to_decision(k), "fewer than k decisions"
    t_fork = time.perf_counter()
    f = o.fork(f"{tmp}/B/out")
    res["fork_s"] = time.perf_counter() - t_fork
    res["fork_time_sim_s"] = o.now
    g = o.fork(f"{tmp}/G/out")
    first = g.sim._scheduler._request_queue[0]
    o.run()
    o.write_output()
    f.run()
    f.write_output()
    res["orig_plus_fork_s"] = time.perf_counter() - t0

    # counterfactual: send the k-th burst's first request somewhere else
    # find what the policy would choose: run a probe fork one step
    p = g.fork()
    p.run(lambda h: h.decisions == k)  # the k-th routing call happens inside
    chosen = p.sim._scheduler._assigned.get(first.id, (None,))[0]
    if chosen is None:  # already finished? look in session map is unreliable; fall back
        chosen = -1
    alt = (chosen + 1) % sc.replicas if chosen >= 0 else 0
    override_next_decision(g, alt)
    g.run()
    g.write_output()
    res["counterfactual"] = {"request_id": first.id, "policy_choice": chosen, "forced": alt}

    A, Ag = outputs(f"{tmp}/A")
    B, Bg = outputs(f"{tmp}/B")
    C, Cg = outputs(f"{tmp}/C")
    G, Gg = outputs(f"{tmp}/G")
    res["sha"] = {n: (sha(x), sha(y) if y.exists() else None) for n, (x, y) in
                  {"unforked": (A, Ag), "fork": (B, Bg), "orig_after_fork": (C, Cg), "counterfactual": (G, Gg)}.items()}
    res["fork_equals_unforked"] = filecmp.cmp(A, B, shallow=False) and filecmp.cmp(Ag, Bg, shallow=False)
    res["orig_equals_unforked"] = filecmp.cmp(A, C, shallow=False) and filecmp.cmp(Ag, Cg, shallow=False)
    da, dg = pd.read_csv(A), pd.read_csv(G)
    m = da.merge(dg, on="Request Id", suffixes=("", "_g"))
    lat = ["request_e2e_time", "prefill_e2e_time"]
    lat = [c for c in lat if c in da.columns]
    diff = np.zeros(len(m), bool)
    for c in lat:
        diff |= ~np.isclose(m[c].to_numpy(), m[c + "_g"].to_numpy(), rtol=0, atol=1e-12)
    t_k = res["fork_time_sim_s"]
    before = m["request_arrived_at"] < t_k
    res["counterfactual"].update({
        "rows": int(len(m)), "rows_differing": int(diff.sum()),
        "differing_arrived_before_fork": int((diff & before.to_numpy()).sum()),
        "first_differing_arrival": float(m.loc[diff, "request_arrived_at"].min()) if diff.any() else None,
        "p99_e2e_unforked": float(np.percentile(da["request_e2e_time"], 99)),
        "p99_e2e_counterfactual": float(np.percentile(dg["request_e2e_time"], 99)),
    })
    res["rss_peak_mb"] = rss_mb()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exact")
    ap.add_argument("--trace", default="Tool")
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=1500)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--out")
    a = ap.parse_args()
    sc = ev.Scenario(a.trace, a.replicas, a.scale, a.start, a.len, a.p, model="l3_8b")
    with tempfile.TemporaryDirectory(prefix="a11_") as tmp:
        r = exactness(sc, a.k, tmp)
    print(json.dumps(r, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
