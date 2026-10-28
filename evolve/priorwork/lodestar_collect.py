"""Offline rows for Lodestar (EXP33 PREREG): run Lodestar in collection mode on training windows only (h1: the ES
training range, which excludes the validation, test and hardware rows) and pool the logged rows per device.

    python -m evolve.priorwork.lodestar_collect --device h100df --workers 4
"""
import argparse
import concurrent.futures as cf
import hashlib
import os
import pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("EVOLVE_CACHE_DIR", str(HERE / ".cache"))
from evolve import evaluate as ev  # noqa: E402

POLICY = ev.REPO / "evolve/policies/priorwork/lodestar_collect.py"
DATA = HERE / "data"


def h1_train():
    out = []
    for trace, start in (("Tool3", 8000), ("Conv3", 2004)):
        for reps, loads in ((8, ev.H1_LOADS_8R[:2]), (16, ev.H1_LOADS_16R[:2])):
            for scale in loads:
                sc = ev.Scenario(trace, reps, scale, start, ev.H1_LEN, 4, model="l3_8b", device="h100df")
                for t, s0, n in ((t, a, b) for t, rs in ev.H1_EXCLUDE.items() for a, b in rs):
                    assert t != trace or start + ev.H1_LEN <= s0 or start >= s0 + n, (sc.tag, t, s0, n)
                out.append(sc)
    return out


def r6k_train():  # r6k's search windows (R6K_SEARCH_WINDOWS holds window starts, 1000 rows each), 4R, high loads
    out = []
    for trace, starts in ev.R6K_SEARCH_WINDOWS.items():
        for start in starts:
            for scale in ev.R6K_LOADS[trace][:2]:
                sc = ev.Scenario(trace, 4, scale, start, 1000, 4, model="l3_8b", device="rtxpro6000")
                for s0, n in ev.R6K_EXCLUDE[trace]:
                    assert start + 1000 <= s0 or start >= s0 + n, (sc.tag, s0, n)
                out.append(sc)
    return out


def a100_train():  # search3b (SELECTION.md: search fitness / training split), Tool and Conv at the knee, draw 1
    return [sc for sc in ev.STAGES["search3b"] if sc.trace in ("Tool", "Conv") and sc.scale == 0.75
            and sc.perturbation == 1]


SETS = {"h100df": h1_train, "rtxpro6000": r6k_train, "a100": a100_train}


def one(sc, path):
    path.unlink(missing_ok=True)  # rows are appended per flush; start each collection run empty
    env = dict(os.environ, LODESTAR_DUMP=str(path))
    os.environ.update(env)  # run_one passes the environment to the simulator process
    return ev.run_one("evolved", sc, policy=str(POLICY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=sorted(SETS))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    scen = SETS[a.device]()
    parts = [DATA / f"lodestar_rows_{sc.tag}.pkl" for sc in scen]
    with cf.ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, scen, parts))
    rows = []
    for sc, p, r in zip(scen, parts, res):
        assert r.get("ok"), (sc.tag, r)
        part = []
        with open(p, "rb") as f:  # one pickle per flush (lodestar_core appends)
            while True:
                try:
                    part.extend(pickle.load(f))
                except EOFError:
                    break
        print(f"{sc.tag}: {len(part)} rows, bundle {r.get('bundle')}")
        rows.extend(part)
    out = DATA / f"lodestar_offline_{a.device}.pkl"
    out.write_bytes(pickle.dumps(rows))
    print(out, len(rows), hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
