"""Idle prefill time per uncached token on one device (DualMap's PREFILL_TPOT), from 4873 on a validation window at
8x the high-load time scale (nearly no queueing). Never uses test windows.

    python -m evolve.priorwork.calib_prefill --device h100df
"""
import argparse
import glob
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("EVOLVE_CACHE_DIR", str(HERE / ".cache"))
os.environ.setdefault("EVOLVE_KEEP_DIR", str(HERE / "keep"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evolve import evaluate as ev  # noqa: E402

POLICY = "evolve/finalists/v4/v4_487363424749.py"
CASES = {  # device -> scenario on a validation/selection window, light load
    "h100df": ev.Scenario("Tool3", 8, 8 * 0.8973, 20000, 2000, 2, model="l3_8b", device="h100df"),
    "a100": ev.Scenario("Tool", 8, 8 * 0.75, -1, 0, 2, model="l3_8b"),
    "rtxpro6000": ev.Scenario("Tool3", 4, 8 * 4.0, 20000, 1000, 2, model="l3_8b", device="rtxpro6000"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=sorted(CASES))
    a = ap.parse_args()
    sc = CASES[a.device]
    r = ev.run_one("evolved", sc, policy=POLICY)
    assert r.get("ok"), r
    f = glob.glob(f"{os.environ['EVOLVE_KEEP_DIR']}/{ev.code_version()}/{sc.tag}/v4_487363424749_*/**/request_metrics.csv",
                  recursive=True)[0]
    df = pd.read_csv(f)
    unc = df["request_num_prefill_tokens"] - df["request_num_prefill_tokens_cached"]
    d = df[unc >= 1024]
    per = d["prefill_time_execution_plus_preemption"] / (d["request_num_prefill_tokens"] - d["request_num_prefill_tokens_cached"])
    print(f"{a.device} {sc.tag}: n={len(d)} median {per.median():.3e} s/token (p10 {per.quantile(.1):.3e}, p90 "
          f"{per.quantile(.9):.3e}); mean sched delay {df['request_scheduling_delay'].mean():.3f} s; bundle {r.get('bundle')}")


if __name__ == "__main__":
    main()
