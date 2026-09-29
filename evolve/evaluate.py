"""
Cascaded evaluator for evolved routing policies.

Stage 0  contract: the policy loads and routes the bundled tiny trace (seconds).
Stage 1  screen:   trace windows (default 1,000 requests) in a small scenario set,
                   paired against heuristic baselines on the identical workload.
Later stages (full traces, full grid, held-out set) reuse ``run_one``/``score``
with different ``Scenario`` lists; see evolve/README.md.

Every simulation runs in a fresh ``python -m vidur.main`` process (request IDs and
the reward plugin are process-global). A run counts only if every admitted
request completed (completion gate). Metrics come from scripts/tail_metrics.py.

Score: for scenario s and metric m in {e2e, ttft, tpot},
    r[s, m] = ln(P99_candidate / min over baselines of P99_baseline)
    fitness = -(mean over s, m of r + 0.5 * max over s of mean_m r) - mean_penalty
where mean_penalty sums ln(mean_e2e_cand / best mean_e2e) over scenarios in which
the candidate's mean E2E is more than 5 % worse than the best baseline's. Higher
is better; 0 means "matches the per-scenario best heuristic on every tail".

CLI:
    python -m evolve.evaluate --policy evolve/policies/lor.py --stage 1 --workers 8
    python -m evolve.evaluate --policy <frozen finalist> --stage heldout --workers 16
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from tail_metrics import summarize_rows  # noqa: E402

TRACES = {
    "Tool": REPO / "data/processed_traces/mooncake_toolagent_trace.csv",
    # v1 conversation trace: the file every Alpine experiment used (git-tracked; identical on all
    # machines). The untracked v2 differs in prompt padding and inferred sessions.
    "Conv": REPO / "data/processed_traces/mooncake_conversation_trace.csv",
    "Synth": REPO / "data/processed_traces/mooncake_synthetic_trace.csv",
}
BUCKETED = {"Tool", "Conv"}  # Mooncake conv/tool timestamps are 3 s buckets
KV_CAPACITY = {"7b": 122752, "70b": 525920}
MODEL_NAME = {"7b": "meta-llama/Llama-2-7b-hf", "70b": "meta-llama/Llama-2-70b-hf"}
TP = {"7b": 1, "70b": 4}
SEED_FLAG = {
    "random": "random_global_scheduler_config_seed",
    "round_robin": "round_robin_global_scheduler_config_seed",
    "lor": "l_o_r_global_scheduler_config_seed",
    "sticky_lor": "sticky_lor_global_scheduler_config_seed",
    "lop_uncached": "l_o_p_uncached_global_scheduler_config_seed",
    "evolved": "evolved_global_scheduler_config_seed",
}
DEFAULT_BASELINES = ("sticky_lor", "lor", "round_robin", "random", "lop_uncached")
CACHE_DIR = Path(os.environ.get("EVOLVE_CACHE_DIR", REPO / "evolve" / ".cache"))


@dataclass(frozen=True)
class Scenario:
    trace: str
    replicas: int
    scale: float
    window_start: int  # row in the capacity-filtered trace; -1 = full trace
    window_len: int
    perturbation: int  # 0 = exact replay, p>0 = seed 100p+1 and in-bucket jitter
    model: str = "7b"
    warmup_frac: float = 0.2  # requests arriving in the first 20 % of the window are not scored

    @property
    def seed(self) -> int:
        return 100 * self.perturbation + 1

    @property
    def jitter(self) -> float:
        return 3.0 if (self.perturbation > 0 and self.trace in BUCKETED) else 0.0

    @property
    def tag(self) -> str:
        w = "full" if self.window_start < 0 else f"w{self.window_start}+{self.window_len}"
        return f"{self.model}-{self.replicas}R-{self.trace}-s{self.scale}-{w}-p{self.perturbation}"


# Stage-1 search set (2026-09-29, from frontier v2): FULL traces in the band where
# routing matters and different heuristics win — 8R Tool/Conv at scale 1.0 (moderate:
# LOR best on E2E, LOP_UNCACHED 23-24 % better on TTFT) and 0.75 (knee: STICKY_LOR 0.44x
# LOR's P99 E2E on Tool, LOP_UNCACHED 0.87x on Conv). Perturbation 1 only. With the
# restart fix a full Tool/Conv run takes 10-15 min, so no windowing (windows start with
# empty queues and cannot show backlog dynamics).
STAGE1 = [Scenario(t, 8, sc, -1, 0, 1) for t in ("Tool", "Conv") for sc in (1.0, 0.75)]

# Held out from search: never shown to the proposer or used for selection. Same band on
# other perturbations, the Synth trace, and 16 replicas. Evaluate only frozen finalists.
HELDOUT = (
    [Scenario(t, 8, sc, -1, 0, p) for t in ("Tool", "Conv") for sc in (1.0, 0.75) for p in (2, 3)]
    + [Scenario("Synth", 8, sc, -1, 0, p) for sc in (1.0, 0.75) for p in (1, 2, 3)]
    + [Scenario(t, 16, 0.5, -1, 0, p) for t in ("Tool", "Conv", "Synth") for p in (1, 2)]
)
STAGES = {"1": STAGE1, "heldout": HELDOUT}

_WINDOW_LOCK = threading.Lock()


def window_csv(sc: Scenario) -> Path:
    """Deterministic, cached window of the capacity-filtered trace, rebased to t=0."""
    with _WINDOW_LOCK:  # parallel workers may request the same window
        return _window_csv(sc)


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    write(tmp)
    os.replace(tmp, path)


def _window_csv(sc: Scenario) -> Path:
    src = TRACES[sc.trace]
    key = hashlib.sha256(f"{file_sha(src)}|{sc.window_start}|{sc.window_len}|{KV_CAPACITY[sc.model]}".encode()).hexdigest()[:16]
    out = CACHE_DIR / "windows" / f"{sc.trace}_{sc.window_start}_{sc.window_len}_{key}.csv"
    if out.exists():
        return out
    df = pd.read_csv(src)
    df = df[(df.num_prefill_tokens + df.num_decode_tokens) <= KV_CAPACITY[sc.model]].reset_index(drop=True)
    if sc.window_start >= 0:
        df = df.iloc[sc.window_start: sc.window_start + sc.window_len].copy()
        if len(df) < sc.window_len:
            raise ValueError(f"window {sc.tag} runs past the end of the trace")
        df["arrived_at"] = df["arrived_at"] - df["arrived_at"].min()
    _atomic_write(out, lambda t: df.to_csv(t, index=False))
    return out


_SHA: Dict[Path, str] = {}


def file_sha(p: Path) -> str:
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def code_version() -> str:
    """Commit of the simulator + harness; a dirty vidur/ tree is part of the key."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "vidur", "scripts/tail_metrics.py", "evolve/evaluate.py"],
                               cwd=REPO, capture_output=True, text=True).stdout.strip()
        return head[:12] + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def sim_command(scheduler: str, policy: Optional[str], sc: Scenario, trace_csv: Path, out_dir: Path) -> List[str]:
    gtype = "evolved" if policy else scheduler
    cmd = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_device", "a100",
        "--replica_config_model_name", MODEL_NAME[sc.model],
        "--replica_config_tensor_parallel_size", str(TP[sc.model]),
        "--cluster_config_num_replicas", str(sc.replicas),
        "--replica_scheduler_config_type", "vllm_v1",
        "--global_scheduler_config_type", gtype,
        "--request_generator_config_type", "trace",
        "--trace_request_generator_config_trace_file", str(trace_csv),
        "--trace_request_generator_config_time_scale_factor", str(sc.scale),
        "--trace_request_generator_config_max_tokens", str(KV_CAPACITY[sc.model]),
        "--trace_request_generator_config_seed", str(sc.seed),
        "--seed", str(sc.seed),
        f"--{SEED_FLAG[gtype]}", str(sc.seed),
        "--cache_config_enable_prefix_caching",
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_batch_metrics",
        "--no-metrics_config_enable_chrome_trace",
        "--metrics_config_output_dir", str(out_dir),
    ]
    if sc.jitter:
        cmd += ["--trace_request_generator_config_arrival_jitter", str(sc.jitter)]
    if policy:
        cmd += ["--evolved_global_scheduler_config_policy_path", str(policy)]
    return cmd


def run_one(scheduler: str, sc: Scenario, policy: Optional[str] = None, timeout: int = 3600,
            launcher: Optional[List[str]] = None) -> dict:
    """Simulate one scenario; return the tail summary or a failure record. Never raises."""
    trace_csv = window_csv(sc)
    n_expected = len(pd.read_csv(trace_csv, usecols=["arrived_at"]))
    with tempfile.TemporaryDirectory(prefix="evolve_") as tmp:
        out_dir = Path(tmp) / "out"
        cmd = (launcher or []) + sim_command(scheduler, policy, sc, trace_csv, out_dir)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": f"timeout after {timeout}s", "wall_s": time.time() - t0}
        wall = time.time() - t0
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            return {"ok": False, "reason": f"exit {proc.returncode}", "stderr": tail, "wall_s": wall}
        files = list(out_dir.rglob("request_metrics.csv"))
        if not files:
            return {"ok": False, "reason": "no request_metrics.csv", "wall_s": wall}
        df = pd.read_csv(files[0])
    completed = len(df)
    if completed != n_expected:  # completion gate: a policy may never strand requests
        return {"ok": False, "reason": f"completion gate: {completed}/{n_expected} requests completed",
                "wall_s": wall}
    if sc.window_start >= 0 and sc.warmup_frac > 0:
        span = df["request_arrived_at"].max() - df["request_arrived_at"].min()
        cut = df["request_arrived_at"].min() + sc.warmup_frac * span
        df = df[df["request_arrived_at"] >= cut]
    s = summarize_rows(df.to_dict("records"))
    s.update({"ok": True, "completed": completed, "scored": len(df), "wall_s": wall})
    return s


def baseline_key(scheduler: str, sc: Scenario) -> str:
    return f"{code_version()}|{file_sha(TRACES[sc.trace])[:12]}|{sc.tag}|{scheduler}"


def load_baseline_cache() -> dict:
    p = CACHE_DIR / "baselines.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_baseline_cache(cache: dict) -> None:
    _atomic_write(CACHE_DIR / "baselines.json",
                  lambda t: Path(t).write_text(json.dumps(cache, indent=1, sort_keys=True)))


def run_many(jobs, workers: int, launcher=None, timeout=3600) -> list:
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, sched, sc, pol, timeout, launcher) for sched, sc, pol in jobs]
        return [f.result() for f in futs]


def ensure_baselines(scenarios, baselines=DEFAULT_BASELINES, workers=4, launcher=None, timeout=3600) -> dict:
    cache = load_baseline_cache()
    todo = [(b, sc, None) for sc in scenarios for b in baselines if baseline_key(b, sc) not in cache]
    for (b, sc, _), res in zip(todo, run_many(todo, workers, launcher, timeout)):
        if not res.get("ok"):
            raise RuntimeError(f"baseline {b} failed on {sc.tag}: {res}")
        cache[baseline_key(b, sc)] = res
    if todo:
        save_baseline_cache(cache)
    return {(b, sc): cache[baseline_key(b, sc)] for sc in scenarios for b in baselines}


METRICS = ("e2e", "ttft", "tpot")


def score(candidate: Dict[Scenario, dict], base: dict, baselines=DEFAULT_BASELINES) -> dict:
    rows, per_s = [], []
    penalty = 0.0
    for sc, res in candidate.items():
        rs = []
        for m in METRICS:
            best = min(base[(b, sc)][f"p99_{m}"] for b in baselines)
            r = math.log(res[f"p99_{m}"] / best)
            rs.append(r)
            rows.append({"scenario": sc.tag, "metric": m, "p99": res[f"p99_{m}"], "best_heuristic_p99": best, "log_ratio": r})
        per_s.append(sum(rs) / len(rs))
        best_mean = min(base[(b, sc)]["mean_e2e"] for b in baselines)
        if res["mean_e2e"] > 1.05 * best_mean:
            penalty += math.log(res["mean_e2e"] / best_mean)
    all_r = [r["log_ratio"] for r in rows]
    fitness = -(sum(all_r) / len(all_r) + 0.5 * max(per_s)) - penalty
    return {"combined_score": fitness, "mean_log_ratio": sum(all_r) / len(all_r),
            "worst_scenario_log_ratio": max(per_s), "mean_e2e_penalty": penalty, "details": rows}


def evaluate_policy(policy: str, scenarios=None, workers=4, launcher=None, timeout=3600,
                    baselines=DEFAULT_BASELINES) -> dict:
    scenarios = scenarios or STAGE1
    base = ensure_baselines(scenarios, baselines, workers, launcher, timeout)
    results = run_many([("evolved", sc, policy) for sc in scenarios], workers, launcher, timeout)
    failures = {sc.tag: r for sc, r in zip(scenarios, results) if not r.get("ok")}
    if failures:
        return {"combined_score": -1e9, "valid": False, "failures": failures}
    out = score(dict(zip(scenarios, results)), base, baselines)
    out.update({"valid": True, "code_version": code_version(),
                "wall_s_total": sum(r["wall_s"] for r in results),
                "scenarios": [asdict(sc) for sc in scenarios]})
    return out


def feedback_text(result: dict) -> str:
    """Compact per-scenario table for the proposer's prompt."""
    if not result.get("valid"):
        return "INVALID: " + json.dumps({k: v.get("reason") for k, v in result.get("failures", {}).items()})
    lines = [f"fitness {result['combined_score']:.4f} (0 = best heuristic on every tail; higher is better)"]
    for d in result["details"]:
        lines.append(f"{d['scenario']:32s} {d['metric']:5s} P99 {d['p99']:9.3f}  best heuristic {d['best_heuristic_p99']:9.3f}  ratio {math.exp(d['log_ratio']):.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--builtin", help="evaluate a built-in router instead, e.g. lor")
    ap.add_argument("--stage", choices=sorted(STAGES), default="1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--srun", action="store_true", help="launch each simulation with srun --exact (inside an allocation)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    launcher = ["srun", "--exact", "-n1", "-c1", "--mem=6G"] if a.srun else None
    scenarios = STAGES[a.stage]
    if a.builtin:  # sanity check: score a heuristic against the pool it belongs to
        results = run_many([(a.builtin, sc, None) for sc in scenarios], a.workers, launcher, a.timeout)
        base = ensure_baselines(scenarios, workers=a.workers, launcher=launcher, timeout=a.timeout)
        res = score(dict(zip(scenarios, results)), base)
        res["valid"] = True
    else:
        res = evaluate_policy(a.policy, scenarios, a.workers, launcher, a.timeout)
    print(json.dumps(res, indent=1) if a.json else feedback_text(res))


if __name__ == "__main__":
    main()
