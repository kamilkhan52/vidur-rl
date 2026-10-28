"""Per-decision router overhead (EXP33): run one scenario in-process and time every policy call.

    python -m evolve.priorwork.overhead --set h1_val8 --routers glia,lodestar --out evolve/priorwork/out/overhead.json

The harness already counts policy time (EvolvedGlobalScheduler.policy_seconds / observe_seconds) but only as totals;
here its _call is wrapped in-process to keep every call's wall time, so mean and P99 per decision are reported.
Routing calls (choose / route / dispatch) and observe() calls are kept apart. Overhead runs are separate from the
comparison runs and never on test windows. Times are Python wall time on the machine that runs this (recorded).
"""
import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from evolve import evaluate as ev
from evolve.priorwork import compare as cmp

SETS = {  # validation windows only, heavy load (the most decisions per second and the deepest queues)
    "h1_val8": lambda: [sc for sc in ev.STAGES["h1_val"] if sc.replicas == 8 and sc.trace == "Tool3"
                        and sc.window_start == 20000 and sc.perturbation == 2 and sc.scale == ev.H1_LOADS_8R[0]],
    "h1_val16": lambda: [sc for sc in ev.STAGES["h1_val"] if sc.replicas == 16 and sc.trace == "Tool3"
                         and sc.window_start == 20000 and sc.perturbation == 2 and sc.scale == ev.H1_LOADS_16R[0]],
}


def measure(policy: str, sc) -> dict:
    from vidur.config import SimulationConfig
    from vidur.scheduler.global_scheduler import evolved_global_scheduler as egs
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    route, obs = [], []
    orig = egs.EvolvedGlobalScheduler._call

    def timed(self, fn, *args, observe: bool = False):
        t0 = time.perf_counter()
        try:
            return orig(self, fn, *args, observe=observe)
        finally:
            (obs if observe else route).append(time.perf_counter() - t0)

    egs.EvolvedGlobalScheduler._call = timed
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ev.sim_command("evolved", policy, sc, ev.window_csv(sc), Path(tmp), cmp.NO_PLOTS)
            argv = cmd[cmd.index("vidur.main") + 1:]
            saved = sys.argv
            sys.argv = ["vidur.main", *argv]
            cwd = os.getcwd()
            os.chdir(ev.REPO)
            try:
                config = SimulationConfig.create_from_cli_args()
                set_seeds(config.seed)
                t0 = time.perf_counter()
                sim = Simulator(config)
                sim.run()
                wall = time.perf_counter() - t0
            finally:
                sys.argv = saved
                os.chdir(cwd)
    finally:
        egs.EvolvedGlobalScheduler._call = orig
    r, o = np.array(route), np.array(obs)
    s = sim._scheduler
    return {"scenario": sc.tag, "decisions": int(len(r)), "requests_routed": len(s._assigned),
            "route_mean_ms": float(r.mean() * 1e3) if len(r) else 0.0,
            "route_p99_ms": float(np.percentile(r, 99) * 1e3) if len(r) else 0.0,
            "route_max_ms": float(r.max() * 1e3) if len(r) else 0.0,
            "observe_calls": int(len(o)), "observe_mean_ms": float(o.mean() * 1e3) if len(o) else 0.0,
            "observe_p99_ms": float(np.percentile(o, 99) * 1e3) if len(o) else 0.0,
            "observe_total_s": float(o.sum()), "route_total_s": float(r.sum()), "sim_wall_s": wall}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="h1_val8", choices=sorted(SETS))
    ap.add_argument("--routers", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sc = SETS[a.set]()[0]
    out = Path(a.out)
    rep = json.loads(out.read_text()) if out.exists() else {}
    for n in a.routers.split(","):
        ref = cmp.ref_of(n)
        if not ref.startswith("policy:"):
            raise SystemExit(f"{n}: built-in scheduler, not timed here")
        r = measure(ref[len("policy:"):], sc)
        r.update({"host": platform.node(), "code_version": ev.code_version()})
        rep.setdefault(a.set, {})[n] = r
        print(n, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
