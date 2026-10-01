"""
Why does an evolved router win, and where? Runs the policy and the heuristic baselines on a
set of scenarios, KEEPS the per-request metrics, and compares the composition of the
latency tails:

  - cache: prefix-cache hit ratio, share of cold misses (no cached tokens), overall and
    among the slowest 1 % of requests (by E2E and by TTFT);
  - where the time goes for tail requests: scheduling (queue) delay vs prefill vs decode;
  - prompt length of tail requests vs all requests;
  - load spread: per-replica request share, and per-replica P99 TTFT spread;
  - preemptions (restarts).

    python -m evolve.analysis.mechanism --policy evolve/finalists/claude_r1_76be082eb0f5.py \
        --out-dir evolve/analysis/mechanism_claude_r1 --workers 16

Outputs <out-dir>/runs/<scenario>/<scheduler>/request_metrics.csv and summary.json.
"""

import argparse
import concurrent.futures as cf
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from evolve import evaluate as ev

SCENARIOS = ev.STAGE1 + [
    ev.Scenario("Synth", 8, 0.75, -1, 0, 1),
    ev.Scenario("Synth", 8, 1.0, -1, 0, 1),
    ev.Scenario("Tool", 16, 0.5, -1, 0, 1),
    ev.Scenario("Conv", 16, 0.5, -1, 0, 1),
]


def simulate(sched, sc, policy, out: Path, timeout=7200):
    dest = out / sc.tag / (sched if not policy else "evolved")
    target = dest / "request_metrics.csv"
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ev.sim_command(sched, policy, sc, ev.window_csv(sc), dest / "sim")
    p = subprocess.run(cmd, cwd=ev.REPO, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{sc.tag} {sched}: {p.stderr[-800:]}")
    src = next((dest / "sim").rglob("request_metrics.csv"))
    src.replace(target)
    subprocess.run(["rm", "-rf", str(dest / "sim")])
    return target


def describe(df: pd.DataFrame) -> dict:
    e2e, ttft = df["request_e2e_time"], df["prefill_e2e_time"]
    decode = e2e - ttft
    cold = df["request_num_prefill_tokens_cached"] == 0
    hit = df["request_num_prefill_tokens_cached"].sum() / df["request_num_prefill_tokens"].sum()
    out = {"n": len(df), "hit_ratio": float(hit), "cold_share": float(cold.mean()),
           "restarts": int(df["request_num_restarts"].sum()),
           "prompt_mean": float(df["request_num_prefill_tokens"].mean())}
    for name, s in (("e2e", e2e), ("ttft", ttft)):
        tail = df[s >= s.quantile(0.99)]
        t_e2e, t_ttft = tail["request_e2e_time"], tail["prefill_e2e_time"]
        out[f"tail_{name}"] = {
            "p99": float(s.quantile(0.99)),
            "cold_share": float((tail["request_num_prefill_tokens_cached"] == 0).mean()),
            "prompt_mean": float(tail["request_num_prefill_tokens"].mean()),
            "decode_tokens_mean": float(tail["request_num_decode_tokens"].mean()),
            "queue_frac": float((tail["request_scheduling_delay"] / t_e2e).mean()),
            "prefill_frac": float(((t_ttft - tail["request_scheduling_delay"]) / t_e2e).mean()),
            "decode_frac": float(((t_e2e - t_ttft) / t_e2e).mean()),
            "restarted_share": float((tail["request_num_restarts"] > 0).mean()),
        }
    per = df.groupby("replica")
    share = per.size() / len(df)
    out["replica_share_max_over_mean"] = float(share.max() * len(share))
    p99_by = per["prefill_e2e_time"].quantile(0.99)
    out["replica_p99_ttft_max_over_median"] = float(p99_by.max() / p99_by.median())
    out["decode_p99"] = float(decode.quantile(0.99))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out_dir)
    jobs = [(b, sc, None) for sc in SCENARIOS for b in ev.DEFAULT_BASELINES] + \
           [("evolved", sc, a.policy) for sc in SCENARIOS]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        paths = list(ex.map(lambda j: simulate(j[0], j[1], j[2], out / "runs"), jobs))
    summary = {}
    for (sched, sc, _), p in zip(jobs, paths):
        summary.setdefault(sc.tag, {})[sched] = describe(pd.read_csv(p))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for tag, by in summary.items():
        print(f"\n=== {tag}")
        for s, d in sorted(by.items(), key=lambda kv: kv[1]["tail_e2e"]["p99"]):
            te, tt = d["tail_e2e"], d["tail_ttft"]
            print(f"  {s:13s} P99 e2e {te['p99']:7.1f} ttft {tt['p99']:7.1f} | hit {d['hit_ratio']:.3f} "
                  f"cold {d['cold_share']:.3f} | e2e-tail: cold {te['cold_share']:.2f} prompt {te['prompt_mean']:6.0f} "
                  f"q/pf/dec {te['queue_frac']:.2f}/{te['prefill_frac']:.2f}/{te['decode_frac']:.2f} | "
                  f"ttft-tail: cold {tt['cold_share']:.2f} q {tt['queue_frac']:.2f} | restarts {d['restarts']} "
                  f"| share max/mean {d['replica_share_max_over_mean']:.2f}")


if __name__ == "__main__":
    main()
