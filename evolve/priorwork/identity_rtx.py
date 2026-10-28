"""Identity check (EXP33 PREREG §2): 4873 on ConvHW rows 0-999, 4R, scale 3.914, draw 1, rtxpro6000 must equal
sim6000's EXP24 run rv/rtx-8b-r4/Conv_h1000_s3.914_p1/4873 request for request.
    python evolve/priorwork/identity_rtx.py <keep dir>"""
import glob
import os
import sys

import numpy as np
import pandas as pd

os.environ["EVOLVE_KEEP_DIR"] = sys.argv[1]
sys.path.insert(0, os.getcwd())
from evolve import evaluate as e  # noqa: E402

sc = e.Scenario("ConvHW", 4, 3.914, 0, 1000, 1, model="l3_8b", device="rtxpro6000")
r = e.run_one("evolved", sc, policy="evolve/finalists/v4/v4_487363424749.py")
print({k: r.get(k) for k in ("ok", "reason", "mean_e2e", "bundle", "wall_s")}, e.code_version())
mine = pd.read_csv(glob.glob(f"{sys.argv[1]}/**/request_metrics.csv", recursive=True)[0])
ref = pd.read_csv(glob.glob(os.path.expanduser(
    "~/vidur-rl-sim6000/results/local/rv/rtx-8b-r4/Conv_h1000_s3.914_p1/4873/out/*/request_metrics.csv"))[0])
cols = [c for c in ref.columns if ref[c].dtype.kind == "f"]
print(len(mine), len(ref), "max|diff|", max(float(np.nanmax(np.abs(mine[c].values - ref[c].values))) for c in cols))
