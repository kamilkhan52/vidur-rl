"""Load calibration for a model: P99 tails of LOR / sticky-LOR / LOP_UNCACHED on 8R Tool and Conv (draw p1) across
time-scale factors, to place the knee and moderate-load scenarios. One warm-up run first trains the predictor.

  python -m evolve.analysis.calibrate --model l3_70b --scales 1.5 2 3 4 --workers 16
"""
import argparse

import evolve.evaluate as ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ev.MODEL_NAME))
    ap.add_argument("--scales", type=float, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sc = [ev.Scenario(t, 8, s, -1, 0, 1, model=a.model) for t in ("Tool", "Conv") for s in a.scales]
    first = ev.run_one("lor", sc[0], None, 4 * 3600)
    print("warm-up", sc[0].tag, first.get("ok"), first.get("reason", ""), f"wall {first.get('wall_s', 0):.0f}s", flush=True)
    jobs = [(b, s, None) for s in sc for b in ("lor", "sticky_lor", "lop_uncached")]
    for (b, s, _), r in zip(jobs, ev.run_many(jobs, workers=a.workers, timeout=4 * 3600)):
        vals = " ".join(f"{k} {r[k]:.3f}" for k in ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "prefill_hit_ratio") if k in r)
        print(s.tag, b, r.get("ok"), vals, r.get("reason", ""), f"wall {r.get('wall_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
