"""
Score a TRAINED RL design (design module + final checkpoint) greedily on a scenario set,
with the same fitness and baselines as everything else. Frozen finalists only.

    python -m evolve.analysis.rl_heldout --design runs/rl_claude_r1/programs/<id>.py \
        --checkpoint runs/rl_claude_r1/training/<id>/net_ep7.pt --stage heldout --out x.json
"""

import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path

from evolve import evaluate as ev
from evolve import rl_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stage", default="heldout", choices=sorted(ev.STAGES) + ["loto_heldout"])
    ap.add_argument("--protocol", default="v2", choices=sorted(rl_task.PROTOCOLS),
                    help="reference policies of this protocol join the heuristic baselines")
    ap.add_argument("--objective", default="balanced", choices=sorted(ev.OBJECTIVES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    scen = rl_task.LOTO_HELDOUT if a.stage == "loto_heldout" else ev.STAGES[a.stage]
    refs = tuple(rl_task.PROTOCOLS[a.protocol]["references"])
    baselines = tuple(ev.DEFAULT_BASELINES) + refs
    base = ev.ensure_baselines(scen, baselines, workers=a.workers)
    extra = rl_task.learned_args("inference", ckpt_in=Path(a.checkpoint))
    with cf.ThreadPoolExecutor(a.workers) as ex:
        runs = list(ex.map(lambda sc: ev.run_one("learned", sc, a.design, 3600, None, extra), scen))
    bad = {sc.tag: r.get("reason") for sc, r in zip(scen, runs) if not r.get("ok")}
    out = {"design": a.design, "checkpoint": a.checkpoint, "stage": a.stage, "code_version": ev.code_version()}
    if bad:
        out["failures"] = bad
    else:
        res = ev.score(dict(zip(scen, runs)), base, baselines, a.objective)
        out["result"] = res
        out["references"] = list(refs)
        if refs:  # the policy relative to each reference alone, per scenario and metric
            out["vs_reference"] = {
                r: {sc.tag: {m: run[f"p99_{m}"] / base[(r, sc)][f"p99_{m}"] for m in ev.METRICS}
                    for sc, run in zip(scen, runs)} for r in refs}
        for m in ev.METRICS:
            r = [d["p99"] / d["best_heuristic_p99"] for d in res["details"] if d["metric"] == m]
            out[m] = {"better": sum(x < 1 for x in r), "n": len(r),
                      "geo": math.exp(sum(map(math.log, r)) / len(r)), "min": min(r), "max": max(r)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("failures", "e2e", "ttft", "tpot")}, indent=1))
    print("fitness", out.get("result", {}).get("combined_score"))


if __name__ == "__main__":
    main()
