"""Identity check (EXP33 PREREG): 4873 on test3b's Tool full trace, 16R, scale 0.5, draw 9, A100 must equal vidur-working's
cached run (evolve/.cache_v4/baselines.json, key cbeb5161f046|327e8b4c8cdb|l3_8b-16R-Tool-s0.5-full-p9|policy:487363424749)
on every summary field except wall time.  python evolve/priorwork/identity_a100.py"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from evolve import evaluate as e
REF_KEY = "cbeb5161f046|327e8b4c8cdb|l3_8b-16R-Tool-s0.5-full-p9|policy:487363424749"
ref = json.load(open(os.path.expanduser("~/vidur-rl-working/evolve/.cache_v4/baselines.json")))[REF_KEY]
sc = e.Scenario("Tool", 16, 0.5, -1, 0, 9, model="l3_8b")
assert sc.tag == REF_KEY.split("|")[2], sc.tag
r = e.run_one("evolved", sc, policy="evolve/finalists/v4/v4_487363424749.py", extra=["--no-metrics_config_store_plots"])
print("code_version", e.code_version(), {k: r.get(k) for k in ("ok", "reason", "mean_e2e", "completed", "bundle", "wall_s")})
diff = {k: (ref[k], r.get(k)) for k in ref if k != "wall_s" and ref[k] != r.get(k)}
print("fields compared", len([k for k in ref if k != "wall_s"]), "differing", diff)
