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
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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
    "learned": "learned_global_scheduler_config_seed",
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
    transform: str = ""  # stress workload applied to the window: see TRANSFORMS (EXP12 v3)
    jitter_s: float = -1.0  # >= 0 overrides the arrival jitter (protocol v2: Synth arrivals jittered too)

    @property
    def seed(self) -> int:
        return 100 * self.perturbation + 1

    @property
    def jitter(self) -> float:
        if self.jitter_s >= 0:
            return self.jitter_s if self.perturbation > 0 else 0.0
        return 3.0 if (self.perturbation > 0 and self.trace in BUCKETED) else 0.0

    @property
    def tag(self) -> str:
        w = "full" if self.window_start < 0 else f"w{self.window_start}+{self.window_len}"
        x = f"-{self.transform}" if self.transform else ""
        j = f"-j{self.jitter_s:g}" if self.jitter_s >= 0 else ""
        return f"{self.model}-{self.replicas}R-{self.trace}-s{self.scale}-{w}-p{self.perturbation}{x}{j}"


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
# Protocol v2 (2026-10-06): search on two arrival draws per scenario so selection is less exposed to one lucky
# draw; held-out on fresh draws (p2, p3, p5), Synth with real arrival jitter (its timestamps are not bucketed, so
# p1-p3 used to be identical arrivals), and 16R on fresh draws (16R p1 replayed the search draw). The bar adds our
# Mooncake and Dynamo re-implementations (repo 6647bd6) to the five heuristics; AIBrix/SGLang are dominated.
SYNTH_JITTER_S = 1.0  # U[0, 1 s) per request; Synth mean inter-arrival is ~0.26 s, so arrivals reorder locally
SEARCH_V2 = [Scenario(t, 8, sc, -1, 0, p) for t in ("Tool", "Conv") for sc in (1.0, 0.75) for p in (1, 4)]
HELDOUT_V2 = (
    [Scenario(t, 8, sc, -1, 0, p) for t in ("Tool", "Conv") for sc in (1.0, 0.75) for p in (2, 3, 5)]
    + [Scenario("Synth", 8, sc, -1, 0, p, jitter_s=SYNTH_JITTER_S) for sc in (1.0, 0.75) for p in (1, 2, 3)]
    + [Scenario(t, 16, 0.5, -1, 0, p) for t in ("Tool", "Conv") for p in (2, 3, 5)]
    + [Scenario("Synth", 16, 0.5, -1, 0, p, jitter_s=SYNTH_JITTER_S) for p in (1, 2, 3)]
)
BAR_V2 = DEFAULT_BASELINES + tuple(
    f"policy:{REPO / 'evolve' / 'policies' / 'systems' / n}.py" for n in ("mooncake_min_ttft", "dynamo_kv_cost"))
# Final test set (2026-10-07): the held-out set has been used to choose among policies, so final claims are
# made on fresh draws evaluated once, for frozen finalists only. Same shape as HELDOUT_V2: Tool/Conv p6-p8,
# Synth p4-p6, 16R likewise.
TEST_V2 = (
    [Scenario(t, 8, sc, -1, 0, p) for t in ("Tool", "Conv") for sc in (1.0, 0.75) for p in (6, 7, 8)]
    + [Scenario("Synth", 8, sc, -1, 0, p, jitter_s=SYNTH_JITTER_S) for sc in (1.0, 0.75) for p in (4, 5, 6)]
    + [Scenario(t, 16, 0.5, -1, 0, p) for t in ("Tool", "Conv") for p in (6, 7, 8)]
    + [Scenario("Synth", 16, 0.5, -1, 0, p, jitter_s=SYNTH_JITTER_S) for p in (4, 5, 6)]
)
STAGES = {"1": STAGE1, "heldout": HELDOUT, "search2": SEARCH_V2, "heldout2": HELDOUT_V2, "test2": TEST_V2}

_WINDOW_LOCK = threading.Lock()


def window_csv(sc: Scenario) -> Path:
    """Deterministic, cached window of the capacity-filtered trace, rebased to t=0."""
    with _WINDOW_LOCK:  # parallel workers may request the same window
        if sc.transform:
            return _transformed_csv(sc)
        return _window_csv(sc)


# Stress workloads (EXP12 v3): where fixed-rule routers may break. Each maps a base window to a
# new trace, deterministically. Rates keep the window's mean load; the time-scale factor of the
# scenario then applies on top.
#   ramp      the arrival rate swings sinusoidally by +-35 % around the base rate (period ~600 s)
#   burst     every ~150 s, 30 % of the requests arrive in a 15 s burst at 3x the base rate; the
#             rest arrive at ~0.78x, so the mean rate (and the trace span) is unchanged
#   mix-<T>   trace <T> (capacity-filtered, arrivals stretched 2x, tiled over the window) overlaid
#             on the window: two workloads at once, with disjoint session ids and prefix-hash ids
#             per tile (no false cache sharing); total load rises accordingly
TRANSFORMS = ("ramp", "burst", "mix-Synth", "mix-Tool", "mix-Conv")


def _warp(arrivals: np.ndarray, rate) -> np.ndarray:
    """New arrival times when the instantaneous arrival rate is multiplied by rate(t)."""
    order = np.argsort(arrivals, kind="stable")
    a = arrivals[order]
    gaps = np.diff(a, prepend=a[0])
    new = np.cumsum(gaps / rate(a))
    new = new - new[0]
    out = np.empty_like(new)
    out[order] = new
    return out


def _transformed_csv(sc: Scenario) -> Path:
    base_sc = replace(sc, transform="")
    base = _window_csv(base_sc)
    key = hashlib.sha256(f"{file_sha(base)}|{sc.transform}|v2".encode()).hexdigest()[:16]
    out = CACHE_DIR / "windows" / f"{sc.trace}_{sc.window_start}_{sc.window_len}_{sc.transform}_{key}.csv"
    if out.exists():
        return out
    df = pd.read_csv(base)
    t = df["arrived_at"].to_numpy(dtype=float)
    if sc.transform == "ramp":
        # mean of 1/(1 + a sin) over a period is 1/sqrt(1 - a^2): rescale so the span is unchanged
        amp = 0.35
        k = 1.0 / np.sqrt(1.0 - amp * amp)
        df["arrived_at"] = _warp(t, lambda x: k * (1.0 + amp * np.sin(2 * np.pi * x / 600.0)))
    elif sc.transform == "burst":
        # in original time, 45 s of every 150 s (30 % of requests) compressed 3x -> a 15 s burst;
        # the other 105 s at 0.7/0.9 = 0.778x -> 135 s. New period 150 s: span unchanged.
        df["arrived_at"] = _warp(t, lambda x: np.where((x % 150.0) < 45.0, 3.0, 0.7 / 0.9))
    elif sc.transform.startswith("mix-"):
        other = sc.transform[4:]
        o0 = pd.read_csv(_window_csv(Scenario(other, sc.replicas, sc.scale, -1, 0, 0, sc.model)))
        o0["arrived_at"] = (o0["arrived_at"] - o0["arrived_at"].min()) * 2.0
        period = o0["arrived_at"].max() + 1.0
        span = t.max() - t.min()
        tiles = []
        for k in range(int(np.ceil(span / period))):
            o = o0.copy()
            o["arrived_at"] = o["arrived_at"] + t.min() + k * period
            o = o[o["arrived_at"] <= t.max()]
            if "session_id" in o.columns:
                o["session_id"] = o["session_id"] + (k + 1) * 10**9
            if "block_hash_ids" in o.columns:
                off = (k + 1) * 10**12
                o["block_hash_ids"] = o["block_hash_ids"].map(
                    lambda s, off=off: "[" + ", ".join(str(int(v) + off) for v in s.strip("[]").split(",")
                                                       if v.strip()) + "]")
            tiles.append(o)
        df = pd.concat([df] + tiles, ignore_index=True)
    else:
        raise ValueError(f"unknown transform {sc.transform!r}")
    df = df.sort_values("arrived_at", kind="stable").reset_index(drop=True)
    _atomic_write(out, lambda p: df.to_csv(p, index=False))
    return out


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


# Files that cannot change a heuristic or evolved-policy run: the RL scheduler stack.
# Excluding them lets RL work proceed without invalidating the cached baselines.
_VERSION_EXCLUDE = {
    "vidur/scheduler/global_scheduler/rl_cache_aware_global_scheduler.py",
    "vidur/scheduler/global_scheduler/rl_agent.py",
    "vidur/scheduler/global_scheduler/replay_buffer.py",
    "vidur/scheduler/global_scheduler/reward_calculator.py",
    "vidur/scheduler/global_scheduler/state_tracker.py",
    "vidur/scheduler/global_scheduler/learned_global_scheduler.py",
}
_VERSION: Dict[str, str] = {}


def code_version() -> str:
    """Content hash of the simulator + harness files that determine a run's outcome.

    Hashes the files on disk (so uncommitted edits count), not the commit: unrelated
    commits (docs, finalists, RL code) must not invalidate the baseline cache.
    """
    if "v" not in _VERSION:
        try:
            # Walk the tree rather than ask git: cluster copies need not be git checkouts.
            files = sorted(
                str((Path(d) / f).relative_to(REPO)).replace(os.sep, "/")
                for d, _, names in os.walk(REPO / "vidur") if "__pycache__" not in d
                for f in names if f.endswith(".py")
            )
            files = [f for f in files if f not in _VERSION_EXCLUDE]
            files += ["scripts/tail_metrics.py"]
            h = hashlib.sha256()
            for f in files:
                h.update(f.encode())
                h.update(file_sha(REPO / f).encode())
            # The parts of this harness that shape a run (not the scenario lists or scoring).
            for fn in (sim_command, run_one, _window_csv):
                h.update(inspect.getsource(fn).encode())
            _VERSION["v"] = "c" + h.hexdigest()[:11]
        except Exception:
            _VERSION["v"] = "unknown"
    return _VERSION["v"]


def sim_command(scheduler: str, policy: Optional[str], sc: Scenario, trace_csv: Path, out_dir: Path,
                extra: Optional[List[str]] = None) -> List[str]:
    gtype = scheduler if scheduler in ("evolved", "learned") else ("evolved" if policy else scheduler)
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
        "--metrics_config_store_token_gaps",
    ]
    if sc.jitter:
        cmd += ["--trace_request_generator_config_arrival_jitter", str(sc.jitter)]
    if policy:
        cmd += [f"--{gtype}_global_scheduler_config_policy_path", str(policy)]
    return cmd + list(extra or [])


def run_one(scheduler: str, sc: Scenario, policy: Optional[str] = None, timeout: int = 3600,
            launcher: Optional[List[str]] = None, extra: Optional[List[str]] = None) -> dict:
    """Simulate one scenario; return the tail summary or a failure record. Never raises."""
    trace_csv = window_csv(sc)
    n_expected = len(pd.read_csv(trace_csv, usecols=["arrived_at"]))
    with tempfile.TemporaryDirectory(prefix="evolve_") as tmp:
        out_dir = Path(tmp) / "out"
        cmd = (launcher or []) + sim_command(scheduler, policy, sc, trace_csv, out_dir, extra)
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
        gaps_file = files[0].parent / "token_gaps.npz"
        gaps = np.load(gaps_file) if gaps_file.exists() else None
        gap, gap_req = (gaps["gap"], gaps["request_id"]) if gaps is not None else (None, None)
    completed = len(df)
    if completed != n_expected:  # completion gate: a policy may never strand requests
        return {"ok": False, "reason": f"completion gate: {completed}/{n_expected} requests completed",
                "wall_s": wall}
    if sc.window_start >= 0 and sc.warmup_frac > 0:
        span = df["request_arrived_at"].max() - df["request_arrived_at"].min()
        cut = df["request_arrived_at"].min() + sc.warmup_frac * span
        df = df[df["request_arrived_at"] >= cut]
    s = summarize_rows(df.to_dict("records"))
    if gap is not None and len(gap):  # exact time-between-tokens over the scored requests' decode tokens
        keep = np.isin(gap_req, df["Request Id"].to_numpy()) if "Request Id" in df else np.ones(len(gap), bool)
        g = gap[keep]
        s.update({"p50_tbt": float(np.percentile(g, 50)), "p95_tbt": float(np.percentile(g, 95)),
                  "p99_tbt": float(np.percentile(g, 99)), "p999_tbt": float(np.percentile(g, 99.9)),
                  "mean_tbt": float(g.mean()), "n_tbt": int(len(g))})
    s.update({"ok": True, "completed": completed, "scored": len(df), "wall_s": wall})
    return s


def baseline_key(scheduler: str, sc: Scenario) -> str:
    if scheduler.startswith("policy:"):  # a reference policy: keyed by its file contents
        scheduler = "policy:" + file_sha(Path(scheduler[len("policy:"):]).resolve())[:12]
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


_BASELINE_LOCK = threading.Lock()


def ensure_baselines(scenarios, baselines=DEFAULT_BASELINES, workers=4, launcher=None, timeout=3600) -> dict:
    # One thread computes missing baselines; concurrent candidates (the driver evaluates seeds
    # and proposals in parallel) wait for it instead of re-running them and racing on the cache.
    with _BASELINE_LOCK:
        return _ensure_baselines(scenarios, baselines, workers, launcher, timeout)


def _ensure_baselines(scenarios, baselines, workers, launcher, timeout) -> dict:
    cache = load_baseline_cache()
    todo = [(b, sc, None) for sc in scenarios for b in baselines if baseline_key(b, sc) not in cache]
    # "policy:<path>" references are evolved-router runs of that policy file
    jobs = [("evolved", sc, b[len("policy:"):]) if b.startswith("policy:") else (b, sc, None) for b, sc, _ in todo]
    for (b, sc, _), res in zip(todo, run_many(jobs, workers, launcher, timeout)):
        if not res.get("ok"):
            raise RuntimeError(f"baseline {b} failed on {sc.tag}: {res}")
        cache[baseline_key(b, sc)] = res
    if todo:
        save_baseline_cache(cache)
    return {(b, sc): cache[baseline_key(b, sc)] for sc in scenarios for b in baselines}


METRICS = ("e2e", "ttft", "tpot")


# Objectives: weights on the per-scenario log P99 ratios (they sum to 1). "balanced" is the
# EXP11 fitness (equal weights) and reproduces it exactly. "completion" targets long-running
# agents, where time to completion (TTFT + generation time, i.e. E2E and TPOT) matters more
# than the first token. "interactive" favours the first token.
OBJECTIVES = {
    "balanced": {"e2e": 1 / 3, "ttft": 1 / 3, "tpot": 1 / 3},
    "completion": {"e2e": 0.5, "ttft": 0.1, "tpot": 0.4},
    "interactive": {"e2e": 0.3, "ttft": 0.6, "tpot": 0.1},
    # 2026-10-06: adds exact P99 time-between-tokens (token-weighted; stalls a request's per-request TPOT averages
    # away). E2E here is arrival -> last token, i.e. TTLT. Needs runs with token-gap logging (code >= ca54802e0b09).
    "tail4": {"e2e": 0.25, "ttft": 0.25, "tpot": 0.25, "tbt": 0.25},
}


def score(candidate: Dict[Scenario, dict], base: dict, baselines=DEFAULT_BASELINES,
          objective: str = "balanced") -> dict:
    """Fitness = -(weighted mean log ratio + 0.5 * worst scenario) - mean-E2E penalty.

    The reference for each scenario and metric is the best of ``baselines`` (heuristics, and
    optionally reference policies "policy:<path>", e.g. the evolved router as a bar to beat).
    """
    w = OBJECTIVES[objective]
    rows, per_s = [], []
    penalty = 0.0
    for sc, res in candidate.items():
        s = 0.0
        for m in (m for m in ("e2e", "ttft", "tpot", "tbt") if w.get(m)):
            best = min(base[(b, sc)][f"p99_{m}"] for b in baselines)
            r = math.log(res[f"p99_{m}"] / best)
            s += w[m] * r
            rows.append({"scenario": sc.tag, "metric": m, "p99": res[f"p99_{m}"], "best_heuristic_p99": best,
                         "log_ratio": r})
        per_s.append(s)
        best_mean = min(base[(b, sc)]["mean_e2e"] for b in baselines)
        if res["mean_e2e"] > 1.05 * best_mean:
            penalty += math.log(res["mean_e2e"] / best_mean)
    mean_w = sum(per_s) / len(per_s)
    fitness = -(mean_w + 0.5 * max(per_s)) - penalty
    return {"combined_score": fitness, "mean_log_ratio": mean_w, "objective": objective,
            "worst_scenario_log_ratio": max(per_s), "mean_e2e_penalty": penalty, "details": rows}


def evaluate_policy(policy: str, scenarios=None, workers=4, launcher=None, timeout=3600,
                    baselines=DEFAULT_BASELINES, objective: str = "balanced") -> dict:
    scenarios = scenarios or STAGE1
    base = ensure_baselines(scenarios, baselines, workers, launcher, timeout)
    results = run_many([("evolved", sc, policy) for sc in scenarios], workers, launcher, timeout)
    failures = {sc.tag: r for sc, r in zip(scenarios, results) if not r.get("ok")}
    if failures:
        return {"combined_score": -1e9, "valid": False, "failures": failures}
    out = score(dict(zip(scenarios, results)), base, baselines, objective)
    out.update({"valid": True, "code_version": code_version(),
                "wall_s_total": sum(r["wall_s"] for r in results),
                "scenarios": [asdict(sc) for sc in scenarios]})
    return out


def feedback_text(result: dict) -> str:
    """Compact per-scenario table for the proposer's prompt."""
    if not result.get("valid"):
        return "INVALID: " + json.dumps({k: v.get("reason") for k, v in result.get("failures", {}).items()})
    lines = [f"fitness {result['combined_score']:.4f} (0 = matches the bar on every tail; higher is better)"]
    for d in result["details"]:
        lines.append(f"{d['scenario']:32s} {d['metric']:5s} P99 {d['p99']:9.3f}  bar {d['best_heuristic_p99']:9.3f}  ratio {math.exp(d['log_ratio']):.3f}")
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
