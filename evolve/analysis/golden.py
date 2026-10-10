"""Golden set: byte-level regression check for the simulator (plan §2, readiness report §7).

A fixed list of short scenarios x policies is simulated through the harness's own ``run_one`` (same command, same
window files, same scoring); the per-request outputs are kept and hashed. A manifest records the hashes, the summary
metrics, the code version, the prediction bundle each model reads and the library versions.

    python -m evolve.analysis.golden build [--workers 4] [--out evolve/golden/v3_<host>.json]
    python -m evolve.analysis.golden check --manifest evolve/golden/v3_<host>.json [--workers 4]

``check`` re-runs every cell and reports byte-identical cells, or for each differing cell which file changed and how
far the summary metrics moved. Run it before and after every simulator change, and on the Mac and on Alpine before
pooling results from the two machines (compare their manifests with ``check --manifest A --against B``).

Hashes: ``request_metrics.csv`` as bytes; ``token_gaps.npz`` as its arrays (name, dtype, shape, bytes; sorted by name),
not as the zip container, which embeds timestamps.
"""
import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import evolve.evaluate as ev

REPO = ev.REPO
OUT_DIR = REPO / "evolve" / "golden"
SYSTEMS = REPO / "evolve" / "policies" / "systems"
SCHEMA = 1

# Short windows (2,000-3,000 requests, 20 % warm-up) so the set runs in minutes; a regression check, not an experiment.
# Covers the headline model (Meta-Llama-3-8B), Llama-2-7B and Llama-2-70B TP4; Tool, Conv and Synth; 8 and 16
# replicas; the knee and moderate loads; exact replay (p0) and perturbed arrivals with jitter.
SCENARIOS = [
    ev.Scenario("Tool", 8, 0.75, 4000, 2000, 1, model="l3_8b"),                                  # knee
    ev.Scenario("Conv", 8, 1.0, 2000, 2000, 0, model="l3_8b"),                                   # moderate, exact replay
    ev.Scenario("Synth", 8, 1.0, 0, 2000, 2, model="l3_8b", jitter_s=ev.SYNTH_JITTER_S),         # Synth, jittered
    ev.Scenario("Tool", 16, 0.5, 8000, 3000, 2, model="l3_8b"),                                  # 16 replicas
    ev.Scenario("Conv", 8, 0.75, 0, 2000, 1, model="7b"),                                        # Llama-2-7B
    ev.Scenario("Tool", 8, 2.0, 2000, 2000, 1, model="70b"),                                     # Llama-2-70B TP4 knee
]
POLICIES = {
    "lor": ("lor", None),
    "sticky_lor": ("sticky_lor", None),
    "lop_uncached": ("lop_uncached", None),
    "mooncake": ("evolved", SYSTEMS / "mooncake_min_ttft.py"),
    "l3_48a0": ("evolved", REPO / "evolve" / "finalists" / "l3_48a067a7df63.py"),
}
SUMMARY_KEYS = ("completed", "scored", "p50_e2e", "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e",
                "prefill_hit_ratio", "queue_delay_p99")


# --- hashing -------------------------------------------------------------------------------------------------------

def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_npz(path) -> str:
    """Content hash of an .npz: every array's name, dtype, shape and bytes, in sorted name order."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for name in sorted(z.files):
            a = np.ascontiguousarray(z[name])
            h.update(name.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


# --- environment ---------------------------------------------------------------------------------------------------

def bundle_keys(models) -> dict:
    """Prediction-bundle key per model, and whether the bundle file exists (a missing bundle means the simulator
    would train its own forests, which differ across machines). Builds the predictor config only; trains nothing."""
    from vidur.config import SimulationConfig
    from vidur.execution_time_predictor import sklearn_execution_time_predictor as sk
    from vidur.execution_time_predictor.execution_time_predictor_registry import ExecutionTimePredictorRegistry

    class _Key(Exception):
        pass

    def grab(self):
        raise _Key(self._bundle_key(), self._config.cache_dir)

    out, saved, argv = {}, sk.SklearnExecutionTimePredictor._predict_from_models, sys.argv
    sk.SklearnExecutionTimePredictor._predict_from_models = grab
    try:
        with tempfile.TemporaryDirectory(prefix="golden_bk_") as tmp:
            for m in sorted(set(models)):
                sc = ev.Scenario("Tool", 8, 1.0, 0, 10, 1, model=m)
                sys.argv = ["vidur"] + ev.sim_command("lor", None, sc, Path(tmp) / "t.csv", Path(tmp) / "o")[3:]
                cfg = SimulationConfig.create_from_cli_args()
                try:
                    ExecutionTimePredictorRegistry.get(
                        cfg.execution_time_predictor_config.get_type(),
                        predictor_config=cfg.execution_time_predictor_config,
                        replica_config=cfg.cluster_config.replica_config,
                        cache_config=cfg.cluster_config.cache_config)
                except _Key as k:
                    key, cache_dir = k.args
                    p = (REPO / cache_dir / f"prediction_bundle_{key}.pkl")
                    out[m] = {"bundle_key": key, "present": p.exists()}
    finally:
        sk.SklearnExecutionTimePredictor._predict_from_models = saved
        sys.argv = argv
    return out


def environment(models) -> dict:
    import pandas
    import sklearn
    return {
        "code_version": ev.code_version(),
        "host": socket.gethostname().split(".")[0],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "bundles": bundle_keys(models),
        "traces": {t: ev.file_sha(p)[:16] for t, p in ev.TRACES.items()},
        "policies": {n: ev.file_sha(p)[:16] for n, (_, p) in POLICIES.items() if p},
    }


# --- running -------------------------------------------------------------------------------------------------------

_LOCAL = threading.local()
_REAL_TMPDIR = tempfile.TemporaryDirectory


@contextlib.contextmanager
def _keep_dir(*args, **kwargs):
    """Stand-in for ``tempfile.TemporaryDirectory`` inside ``run_one``: hands out this thread's kept directory."""
    d = getattr(_LOCAL, "keep", None)
    if d is None:  # not a golden run: behave normally
        with _REAL_TMPDIR(*args, **kwargs) as t:
            yield t
        return
    d.mkdir(parents=True, exist_ok=True)
    yield str(d)


class _TempfileShim:
    """Replaces the ``tempfile`` module seen by evolve.evaluate, overriding only TemporaryDirectory."""

    def __getattr__(self, name):
        return _keep_dir if name == "TemporaryDirectory" else getattr(tempfile, name)


def run_cell(sc, name, keep_root: Path, timeout: int) -> dict:
    sched, pol = POLICIES[name]
    keep = keep_root / f"{sc.tag}__{name}"
    shutil.rmtree(keep, ignore_errors=True)
    _LOCAL.keep = keep
    try:
        res = ev.run_one(sched, sc, str(pol) if pol else None, timeout=timeout)
    finally:
        _LOCAL.keep = None
    cell = {"scenario": sc.tag, "policy": name, "ok": bool(res.get("ok")), "wall_s": round(res.get("wall_s", 0), 1)}
    if not res.get("ok"):
        cell["reason"] = res.get("reason")
        return cell
    rm = list(keep.rglob("request_metrics.csv"))
    tg = list(keep.rglob("token_gaps.npz"))
    cell["request_metrics_sha256"] = sha_file(rm[0]) if rm else None
    cell["token_gaps_sha256"] = sha_npz(tg[0]) if tg else None
    cell["summary"] = {k: res[k] for k in SUMMARY_KEYS if k in res}
    return cell


def run_all(workers: int, keep_root: Path, timeout: int, cells=None) -> list:
    cells = cells or [(sc, n) for sc in SCENARIOS for n in POLICIES]
    for sc, _ in cells:  # build window files serially (they are shared and cached)
        ev.window_csv(sc)
    ev.tempfile = _TempfileShim()
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda c: run_cell(c[0], c[1], keep_root, timeout), cells))
    finally:
        ev.tempfile = tempfile


# --- comparison ----------------------------------------------------------------------------------------------------

def compare(ref: list, new: list) -> list:
    """Differences between two lists of cells, keyed by (scenario, policy)."""
    idx = {(c["scenario"], c["policy"]): c for c in new}
    diffs = []
    for r in ref:
        n = idx.get((r["scenario"], r["policy"]))
        key = f'{r["scenario"]} / {r["policy"]}'
        if n is None:
            diffs.append(f"{key}: missing")
            continue
        if r.get("ok") != n.get("ok"):
            diffs.append(f"{key}: ok {r.get('ok')} -> {n.get('ok')} ({n.get('reason', '')})")
            continue
        changed = [f for f in ("request_metrics_sha256", "token_gaps_sha256") if r.get(f) != n.get(f)]
        if changed:
            moves = []
            for k, v in r.get("summary", {}).items():
                w = n.get("summary", {}).get(k)
                if isinstance(v, (int, float)) and isinstance(w, (int, float)) and v != w:
                    moves.append(f"{k} {v:.6g}->{w:.6g} ({(w / v - 1) * 100:+.3f} %)" if v else f"{k} {v}->{w}")
            diffs.append(f"{key}: {', '.join(changed)} differ; " + ("; ".join(moves) or "summary metrics equal"))
    return diffs


def env_diffs(a: dict, b: dict) -> list:
    out = []
    for k in ("code_version", "python", "numpy", "pandas", "sklearn", "traces", "policies", "bundles"):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    return out


# --- CLI -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out")
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--against", help="compare two manifests without running anything")
    for p in (b, c):
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--keep", help="directory for the kept outputs (default: a temporary directory)")
        p.add_argument("--allow-train", action="store_true",
                       help="run even if a model's prediction bundle is missing (results become machine-specific)")
    a = ap.parse_args(argv)

    if a.cmd == "check" and a.against:
        ref, new = json.loads(Path(a.manifest).read_text()), json.loads(Path(a.against).read_text())
        for line in env_diffs(ref["environment"], new["environment"]):
            print("env  ", line)
        diffs = compare(ref["cells"], new["cells"])
        print("\n".join(diffs) if diffs else f"all {len(ref['cells'])} cells byte-identical")
        return 1 if diffs else 0

    env = environment(sc.model for sc in SCENARIOS)
    missing = [m for m, v in env["bundles"].items() if not v["present"]]
    if missing and not a.allow_train:
        print(f"prediction bundle missing for {missing}: copy it from the reference machine (or pass --allow-train)")
        return 2
    keep_ctx = contextlib.nullcontext(a.keep) if a.keep else _REAL_TMPDIR(prefix="golden_")
    t0 = time.time()
    with keep_ctx as keep:
        cells = run_all(a.workers, Path(keep), a.timeout)
    wall = time.time() - t0

    if a.cmd == "build":
        out = Path(a.out) if a.out else OUT_DIR / f"v3_{env['host']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(wall),
                    "environment": env, "cells": cells}
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        bad = [f'{c["scenario"]} / {c["policy"]}: {c.get("reason")}' for c in cells if not c["ok"]]
        print(f"wrote {out}: {len(cells)} cells in {wall:.0f} s" + (f"; FAILED: {bad}" if bad else ""))
        return 1 if bad else 0

    ref = json.loads(Path(a.manifest).read_text())
    for line in env_diffs(ref["environment"], env):
        print("env  ", line)
    diffs = compare(ref["cells"], cells)
    print("\n".join(diffs) if diffs else f"all {len(cells)} cells byte-identical ({wall:.0f} s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
