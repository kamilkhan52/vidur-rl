"""A11 spike: one-step lookahead signal. At sampled decision points, force the burst's first request onto each
replica, roll out H further decisions with the evolved policy (true future = hindsight), and score.

Scores per candidate:
  full:     drain after H (no further arrivals) -> exact E2E of the affected set (in flight at the fork + H new)
  censored: stop after H decisions; unfinished requests count their elapsed time (cheap, biased low)
Rollouts run in os.fork() children, one at a time (the parent waits), so the parent state is never touched.
"""

import json
import os
import sys
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")  # in-process simulators save plots only; no GUI backend

import numpy as np

from evolve import evaluate as ev
from evolve.sim_fork import build_simulator, override_next_decision
from evolve.spikes.a11.spike_a11 import argv_for
from evolve.spikes.a11.spike_a11_costs import capture_requests, inflight_requests, stop_arrivals


def scores(lat):
    lat = np.asarray(lat, float)
    return {"mean": float(lat.mean()), "p90": float(np.percentile(lat, 90)), "max": float(lat.max()),
            "sq": float(np.mean(lat ** 2))}


def child_rollout(h, choice, H):
    """Runs in the child: force the decision, roll out, score. Returns a dict."""
    affected = inflight_requests(h)
    if choice is not None:
        override_next_decision(h, choice)
    first = h.sim._scheduler._request_queue[0]
    made = capture_requests(h)
    t0 = time.perf_counter()
    n0 = h.decisions
    h.run(lambda x: x.decisions > n0)  # the forced/policy decision has just been applied
    chosen = h.sim._scheduler._assigned[first.id][0]
    h.run(lambda x: x.at_decision() and x.decisions >= n0 + H)
    t_adv = time.perf_counter() - t0
    now = h.now
    reqs = affected + made
    cens = [(r.completed_at if r.completed else now) - r.arrived_at for r in reqs]
    stop_arrivals(h)
    t1 = time.perf_counter()
    h.run()
    t_drain = time.perf_counter() - t1
    full = [r.completed_at - r.arrived_at for r in reqs]
    return {"choice": choice, "routed_to": chosen, "adv_s": t_adv, "drain_s": t_drain, "n": len(reqs),
            "full": scores(full), "cens": scores(cens)}


def in_child(h, choice, H):
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(rfd)
        try:
            out = json.dumps(child_rollout(h, choice, H))
        except Exception as e:  # report, never hang the parent
            out = json.dumps({"error": repr(e)})
        os.write(wfd, out.encode())
        os._exit(0)
    os.close(wfd)
    data = b""
    while True:
        c = os.read(rfd, 1 << 16)
        if not c:
            break
        data += c
    os.close(rfd)
    os.waitpid(pid, 0)
    return json.loads(data)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(trace, R, scale, wlen, points, H, out_path, p=1):
    sc = ev.Scenario(trace, R, scale, 0, wlen, p, model="l3_8b")
    res = {"scenario": sc.tag, "H": H, "points": []}
    with tempfile.TemporaryDirectory() as tmp:
        h = build_simulator(argv_for(sc, tmp))
        for k in points:
            assert h.run_to_decision(k)
            t0 = time.perf_counter()
            policy = in_child(h, None, H)  # the policy's own decision
            cands = [in_child(h, c, H) for c in range(R)]
            wall = time.perf_counter() - t0
            pc = policy["routed_to"]
            row = {"k": k, "t": h.now, "inflight": len(inflight_requests(h)), "policy_choice": pc, "wall_s": wall}
            for m in ("mean", "p90", "max", "sq"):
                full = [c["full"][m] for c in cands]
                cens = [c["cens"][m] for c in cands]
                best = int(np.argmin(full))
                row[m] = {"policy": cands[pc]["full"][m], "best": full[best], "best_choice": best,
                          "worst": max(full), "gain_vs_policy": 1 - full[best] / cands[pc]["full"][m],
                          "policy_rank": int(np.argsort(np.argsort(full))[pc]),
                          "spearman_cens_full": spearman(cens, full),
                          "cens_picks_full_best": int(np.argmin(cens)) == best}
            row["policy_matches_forced_same"] = abs(policy["full"]["mean"] - cands[pc]["full"]["mean"]) < 1e-9
            row["adv_s"] = float(np.mean([c["adv_s"] for c in cands]))
            row["drain_s"] = float(np.mean([c["drain_s"] for c in cands]))
            res["points"].append(row)
            print(json.dumps(row), flush=True)
            with open(out_path, "w") as f:
                json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    trace, R, scale, wlen, H = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    pts = [int(x) for x in sys.argv[6].split(",")]
    main(trace, R, scale, wlen, pts, H, sys.argv[7])
