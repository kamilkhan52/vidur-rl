"""Identity check (EXP33 PREREG): 4873 on ToolHW rows 0-999, 8R, scale 0.8973, draw 1, h100df must equal sim6000's EXP24 run
rv/h100-8b-r8-df/Tool_h1000_s0.8973_p1/4873 request for request.  python evolve/priorwork/identity_h1.py <keep dir>"""
import os, sys, glob
import pandas as pd, numpy as np
os.environ["EVOLVE_KEEP_DIR"] = sys.argv[1]
sys.path.insert(0, os.getcwd())
from evolve import evaluate as e
sc = e.Scenario("ToolHW", 8, 0.8973, 0, 1000, 1, model="l3_8b", device="h100df")
r = e.run_one("evolved", sc, policy="evolve/finalists/v4/v4_487363424749.py")
print({k: r.get(k) for k in ("ok", "reason", "mean_e2e", "bundle", "wall_s")})
mine = pd.read_csv(glob.glob(f"{sys.argv[1]}/**/request_metrics.csv", recursive=True)[0])
ref = pd.read_csv(glob.glob(os.path.expanduser("~/vidur-rl-sim6000/results/local/rv/h100-8b-r8-df/Tool_h1000_s0.8973_p1/4873/out/*/request_metrics.csv"))[0])
cols = [c for c in ref.columns if ref[c].dtype.kind == "f"]
print(len(mine), len(ref), "max|diff|", max(float(np.nanmax(np.abs(mine[c].values - ref[c].values))) for c in cols))
