"""A11 spike: fork and rollout costs at mid-run states (light/knee load, 8R/16R)."""

import json
import os
import resource
import statistics
import sys
import tempfile
import time

os.environ.setdefault("MPLBACKEND", "Agg")  # in-process simulators save plots only; no GUI backend

import numpy as np

from evolve import evaluate as ev
from evolve.sim_fork import build_simulator, override_next_decision
from evolve.spikes.a11.spike_a11 import argv_for


def rss_now_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def capture_requests(h):
    """Record the requests the generator creates from now on (they are the rollout's arrivals)."""
    g = h.sim._request_generator
    made = []
    orig = g.get_next_request

    def wrapped():
        r = orig()
        if r is not None:
            made.append(r)
        return r
    g.get_next_request = wrapped
    return made


def inflight_requests(h):
    gs = h.sim._scheduler
    out = list(gs._request_queue)
    for rs in gs._replica_schedulers.values():
        wq = rs._waiting_queue
        out += (wq.to_list() if hasattr(wq, "to_list") else list(wq)) + list(getattr(rs, "_running", []))
    return out


def stop_arrivals(h):
    g = h.sim._request_generator
    g.num_requests = g.num_requests_generated


def rollout(h, H, drain):
    """From a decision point: route the next H bursts (decisions) with the policy, then optionally drain."""
    made = capture_requests(h)
    t0 = time.perf_counter()
    n0 = h.decisions
    h.run(lambda x: x.at_decision() and x.decisions >= n0 + H)
    t_h = time.perf_counter() - t0
    res = {"H": H, "advance_s": t_h, "sim_advance_s": None}
    if drain:
        stop_arrivals(h)
        t1 = time.perf_counter()
        h.run()
        res["drain_s"] = time.perf_counter() - t1
        lat = [r.completed_at - r.arrived_at for r in made if r.completed]
        res["n_scored"] = len(lat)
        res["p99_e2e_rollout_reqs"] = float(np.percentile(lat, 99)) if lat else None
        res["mean_e2e_rollout_reqs"] = float(np.mean(lat)) if lat else None
    return res


def main(trace, R, scale, wlen, k_fork, p=1):
    sc = ev.Scenario(trace, R, scale, 0, wlen, p, model="l3_8b")
    out = {"scenario": sc.tag}
    with tempfile.TemporaryDirectory() as tmp:
        h = build_simulator(argv_for(sc, tmp))
        t0 = time.perf_counter()
        ok = h.run_to_decision(k_fork)
        out["prefix_run_s"] = time.perf_counter() - t0
        out["reached"] = ok
        out["sim_time_at_fork"] = h.now
        out["events_prefix"] = h.events
        out["decisions_prefix"] = h.decisions
        out["inflight_at_fork"] = len(inflight_requests(h))
        gs = h.sim._scheduler
        out["outstanding_per_replica"] = list(gs._outstanding)
        out["rss_before_fork_mb"] = rss_now_mb()
        ft = []
        for _ in range(3):
            t1 = time.perf_counter()
            f = h.fork()
            ft.append(time.perf_counter() - t1)
            del f
        out["deepcopy_fork_s"] = statistics.median(ft)
        f = h.fork()
        out["rss_after_one_fork_mb"] = rss_now_mb()
        del f
        # os.fork
        oft = []
        for _ in range(3):
            t1 = time.perf_counter()
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            oft.append(time.perf_counter() - t1)
        out["os_fork_roundtrip_s"] = statistics.median(oft)
        # rollouts (in deepcopy forks; the parent state is untouched)
        out["rollouts"] = []
        for H in (16, 64, 256):
            f = h.fork()
            r = rollout(f, H, drain=False)
            del f
            f = h.fork()
            rd = rollout(f, H, drain=True)
            r.update({k: v for k, v in rd.items() if k != "advance_s"})
            out["rollouts"].append(r)
            print(json.dumps(r), flush=True)
        # one rollout inside an os.fork child, to check it works and costs the same
        rfd, wfd = os.pipe()
        t1 = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            os.close(rfd)
            r = rollout(h, 64, drain=True)
            os.write(wfd, json.dumps(r).encode())
            os._exit(0)
        os.close(wfd)
        data = b""
        while True:
            chunk = os.read(rfd, 65536)
            if not chunk:
                break
            data += chunk
        os.waitpid(pid, 0)
        out["os_fork_rollout_H64_total_s"] = time.perf_counter() - t1
        out["os_fork_rollout_H64"] = json.loads(data)
        # plain-run throughput for the rest of the window
        t1 = time.perf_counter()
        e0, d0 = h.events, h.decisions
        h.run()
        dt = time.perf_counter() - t1
        out["rest_run_s"] = dt
        out["events_per_s"] = (h.events - e0) / dt
        out["decisions_per_s_plain"] = (h.decisions - d0) / dt
        out["policy_ms_per_call"] = 1e3 * gs.policy_seconds / max(gs.policy_calls, 1)
        out["policy_share_of_wall"] = gs.policy_seconds / (out["prefix_run_s"] + dt)
    return out


if __name__ == "__main__":
    trace, R, scale, wlen, k = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    r = main(trace, R, scale, wlen, k)
    print("RESULT " + json.dumps(r))
    with open(sys.argv[6], "w") as f:
        json.dump(r, f, indent=1)
