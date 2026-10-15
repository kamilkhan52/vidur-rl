"""Launcher: one LLMServingSim 2.0 simulation of a Vidur window under one of our routers.

    python -m evolve.lss2.run --policy evolve/finalists/v4/v4_487363424749.py \
        --window evolve/.cache_v4/windows/Tool_9000_2000_61ff419c699ee9bd.csv --replicas 8 --scale 0.75 \
        --perturbation 1 --out out/lss2/tool_4873 --lss-root /path/to/LLMServingSim

``--policy`` is a policy file (``choose``/``route`` [+ ``observe``]), a built-in baseline (rr, lor, sticky_lor,
lop_uncached), or ``lss:RR`` / ``lss:LOAD`` for LLMServingSim's own router (no adapter; used to check the adapter's
rr against the stock RR).

Replica settings default to Vidur's vllm_v1 replica (``FIDELITY``): 128 sequences, a 512-token step budget, chunked
prefill with no per-request cap, 16-token blocks, prefix caching, no async scheduling, no full-prompt reservation at
admission, and ``mem_util`` chosen so each replica has Vidur's 30,208 KV blocks (483,328 tokens).

Outputs in ``--out``: ``summary.json`` (Vidur's metric names: mean/p50/p95/p99 of e2e, ttft, tpot; mean/p50/p95/p99/
p999 of tbt; over requests arriving after the first 20 % of the window span, as ``evolve.evaluate.run_one``),
``requests.csv`` (per request), ``lss_requests.csv`` (LLMServingSim's own per-request CSV) and ``stdout.log``.
"""
import argparse
import ast
import contextlib
import csv
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evolve.lss2 import convert as cv  # noqa: E402

# npu_mem.mem_util that gives 30,208 blocks of 16 tokens (Vidur's l3_8b A100 capacity, 483,328 tokens) on a 96 GiB
# RTXPRO6000 with Llama-3.1-8B bf16: (weights + 30,208 * 2 MiB) / 96 GiB. Checked against the blocks LLMServingSim
# reports (summary.json: kv_blocks).
DEFAULT_MEM_UTIL = 0.7704
VIDUR_KV_BLOCKS = cv.VIDUR_KV_TOKENS // cv.BLOCK

# LLMServingSim flags matching Vidur's vllm_v1 replica scheduler (vidur/config/config.py: batch_size_cap 128,
# VllmV1SchedulerConfig.chunk_size 512, CacheConfig.block_size 16; evolve.evaluate.sim_command enables prefix caching)
FIDELITY = [
    "--max-num-seqs", "128",
    "--max-num-batched-tokens", "512",
    "--long-prefill-token-threshold", "0",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--block-size", "16",
    "--no-async-scheduling",
    "--no-reserve-full-isl",
]


def summarize(recs, warmup_frac: float = 0.2) -> dict:
    """Vidur's summary (scripts/tail_metrics.summarize_rows + evolve.evaluate.run_one's TBT and warm-up cut)."""
    arr = np.array([r["arrival_s"] for r in recs], float)
    if warmup_frac > 0 and len(arr):
        cut = arr.min() + warmup_frac * (arr.max() - arr.min())
        recs = [r for r, a in zip(recs, arr) if a >= cut]
    e2e = np.array([r["e2e_s"] for r in recs], float)
    ttft = np.array([r["ttft_s"] for r in recs], float)
    tpot = np.array([(r["e2e_s"] - r["ttft_s"]) / (r["num_decode_tokens"] - 1)
                     for r in recs if r["num_decode_tokens"] > 1], float)
    gaps = np.concatenate([np.asarray(r["itl_ns"], float) for r in recs] or [np.zeros(0)]) / 1e9
    out = {"n_requests": len(recs)}
    for name, v in (("e2e", e2e), ("ttft", ttft), ("tpot", tpot)):
        out[f"mean_{name}"] = float(v.mean()) if len(v) else float("nan")
        for q in (50, 95, 99):
            out[f"p{q}_{name}"] = float(np.percentile(v, q)) if len(v) else float("nan")
    if len(gaps):
        out.update({"p50_tbt": float(np.percentile(gaps, 50)), "p95_tbt": float(np.percentile(gaps, 95)),
                    "p99_tbt": float(np.percentile(gaps, 99)), "p999_tbt": float(np.percentile(gaps, 99.9)),
                    "mean_tbt": float(gaps.mean()), "n_tbt": int(len(gaps))})
    pf = sum(r["num_prefill_tokens"] for r in recs)
    ca = [r.get("num_cached_tokens") for r in recs]
    out["prefill_hit_ratio"] = (sum(ca) / pf) if pf and all(c is not None for c in ca) else float("nan")
    out["n_tpot"] = int(len(tpot))
    return out


def read_lss_csv(path: str, rows_by_id: dict) -> list:
    recs = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rid = int(r["request id"])
            meta = rows_by_id[rid]
            recs.append({"id": rid, "row": meta["row"], "replica": int(r["instance id"]),
                         "arrival_s": int(r["arrival"]) / 1e9, "num_prefill_tokens": int(r["input"]),
                         "num_decode_tokens": int(r["output"]), "ttft_s": int(r["TTFT"]) / 1e9,
                         "e2e_s": int(r["latency"]) / 1e9, "itl_ns": ast.literal_eval(r["ITL"]),
                         "num_cached_tokens": None, "session_id": meta["session_id"]})
    return recs


REQ_FIELDS = ("id", "row", "replica", "session_id", "arrival_s", "num_prefill_tokens", "num_decode_tokens",
              "probe_cached", "num_cached_tokens", "first_token_s", "completed_s", "ttft_s", "e2e_s", "preemptions")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one Vidur window in LLMServingSim 2.0 under one of our routers")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--replicas", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--perturbation", type=int, default=0, help="Vidur draw p (seed 100p+1, 3 s jitter on Tool/Conv)")
    ap.add_argument("--jitter", type=float, default=-1.0)
    ap.add_argument("--head", type=int, default=0, help="first N rows of the window only (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="ctx.rng seed (default 100p+1, as Vidur's scheduler)")
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lss-root", default=os.environ.get("LSS_ROOT", "/Users/kamil/ext/LLMServingSim"))
    ap.add_argument("--mem-util", type=float, default=DEFAULT_MEM_UTIL)
    ap.add_argument("--hardware", default="RTXPRO6000")
    ap.add_argument("--no-fidelity", action="store_true", help="use LLMServingSim's defaults instead of FIDELITY")
    ap.add_argument("--lss-arg", action="append", default=[], help="extra LLMServingSim flag (repeatable)")
    ap.add_argument("--local-inputs", default=os.environ.get("LSS2_LOCAL_INPUTS"),
                    help="write ASTRA-Sim's per-batch inputs under this (node-local) directory instead of the "
                         "checkout's astra-sim/inputs/runs (removed after the run)")
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lss_root = Path(a.lss_root).resolve()
    seed = a.seed if a.seed is not None else 100 * a.perturbation + 1
    builtin_lss = a.policy.startswith("lss:")

    # ---- workload and cluster
    trace = cv.trace_name(a.window)
    rows = cv.read_window(a.window)
    if a.head:
        rows = rows[: a.head]
    prefix_violations = cv.check_prefix_unique(rows)
    rows = cv.replay(rows, a.scale, a.perturbation, cv.vidur_jitter(trace, a.perturbation, a.jitter))
    jsonl = out / "workload.jsonl"
    cv.write_jsonl(rows, str(jsonl), "full" if builtin_lss else "compact")
    cluster = out / "cluster.json"
    cluster.write_text(json.dumps(cv.cluster_config(a.replicas, a.mem_util, hardware=a.hardware), indent=2))

    # ---- LLMServingSim, with our router installed
    sys.path.insert(0, str(lss_root))
    cwd = os.getcwd()
    os.chdir(lss_root)
    import serving.__main__ as lss_main
    from serving.core.request import Request
    from serving.core.router import Router
    from evolve.lss2 import router as rt

    if builtin_lss:
        routing = a.policy.split(":", 1)[1]
        policy_name, RouterCls = a.policy, None
    else:
        routing = "RR"
        policy = rt.load_policy(a.policy)
        policy_name = a.policy
        RouterCls = rt.make_router_class(Router, Request, policy, seed)
        lss_main.Router = RouterCls
    dataset = str(jsonl) if not builtin_lss else os.path.relpath(jsonl, lss_root)
    run_id = f"lss2_{uuid.uuid4().hex[:10]}"
    argv_lss = ["serving", "--cluster-config", str(cluster), "--dataset", dataset,
                "--output", str(out / "lss_requests.csv"), "--request-routing-policy", routing,
                "--run-id", run_id, "--log-level", "WARNING"]
    local_inputs = None
    if a.local_inputs:
        local_inputs = Path(a.local_inputs).resolve() / run_id
        argv_lss += ["--inputs-root", str(local_inputs)]
    argv_lss += [] if a.no_fidelity else FIDELITY
    argv_lss += a.lss_arg
    sys.argv = argv_lss
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            lss_main.main()
    finally:
        os.chdir(cwd)
        (out / "stdout.log").write_text(buf.getvalue())
        if local_inputs is not None and local_inputs.name.startswith("lss2_"):
            import shutil
            shutil.rmtree(local_inputs, ignore_errors=True)
    wall = time.time() - t0

    # ---- results
    n = len(rows)
    rows_by_id = {i: r for i, r in enumerate(rows)}  # LLMServingSim ids = file order = arrival order
    info = {"policy": policy_name, "window": str(a.window), "trace": trace, "replicas": a.replicas, "scale": a.scale,
            "perturbation": a.perturbation, "seed": seed, "head": a.head, "n_loaded": n, "wall_s": wall,
            "lss_argv": argv_lss[1:], "mem_util": a.mem_util, "prefix_violations": prefix_violations}
    if RouterCls is not None:
        r = RouterCls.INSTANCE
        recs = [r.records[k] for k in sorted(r.records)]
        done = [x for x in recs if "e2e_s" in x]
        info.update(r.stats())
        # cross-check against LLMServingSim's own CSV (same requests, same latencies)
        lss = {x["id"]: x for x in read_lss_csv(str(out / "lss_requests.csv"), rows_by_id)}
        diffs = [abs(lss[x["id"]]["e2e_s"] - x["e2e_s"]) for x in done if x["id"] in lss]
        info["csv_crosscheck"] = {"n_csv": len(lss), "n_adapter": len(done),
                                  "max_abs_e2e_diff_s": max(diffs) if diffs else None}
    else:
        done = read_lss_csv(str(out / "lss_requests.csv"), rows_by_id)
        for x in done:
            x["first_token_s"] = x["arrival_s"] + x["ttft_s"]
            x["completed_s"] = x["arrival_s"] + x["e2e_s"]
    info["completed"] = len(done)
    info["ok"] = len(done) == n
    info["sim_makespan_s"] = max((x["completed_s"] for x in done), default=float("nan"))
    info["wall_ms_per_request"] = 1e3 * wall / max(n, 1)
    info["preemptions"] = sum(x.get("preemptions", 0) or 0 for x in done)
    # the simulator's own prefix-hit line (all requests, no warm-up cut)
    for line in buf.getvalue().splitlines():
        if "NPU prefix hit ratio" in line:
            try:
                info["lss_prefix_hit_ratio_all"] = float(line.split()[-1]) / 100
            except ValueError:
                pass
    summ = summarize(done, a.warmup_frac)
    summ.update(info)
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    with open(out / "requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQ_FIELDS, extrasaction="ignore")
        w.writeheader()
        for x in sorted(done, key=lambda x: x["id"]):
            w.writerow(x)
    print(json.dumps({k: summ.get(k) for k in ("ok", "completed", "n_requests", "p99_e2e", "p99_ttft", "p99_tpot",
                                                "p99_tbt", "mean_e2e", "prefill_hit_ratio", "wall_s",
                                                "wall_ms_per_request", "kv_blocks", "policy_import")}, default=str))
    return summ


if __name__ == "__main__":
    main()
