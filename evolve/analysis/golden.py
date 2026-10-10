"""Golden set: byte-level regression check for the simulator (plan §2, readiness report §7).

A fixed list of short scenarios x policies is simulated through the harness's own ``run_one`` (same command, same
window files, same scoring); the per-request outputs are kept and hashed. A manifest records the hashes, the summary
metrics, the code version, the prediction bundle each model reads and the library versions.

    python -m evolve.analysis.golden build [--set default|extended] [--workers 4] [--out PATH] [--force]
    python -m evolve.analysis.golden check --manifest evolve/golden/v4_<host>.json [--workers 4]

``build`` writes ``evolve/golden/<v3|v4>[ext]_<host>.json`` by default (v3 when EVOLVE_V3_DATA=1) and refuses to
overwrite an existing manifest without ``--force``; a run with a failed cell or a missing hash is written to
``<name>.failed.json`` instead. ``check`` re-runs the cells of the manifest's set and reports byte-identical cells, or
for each differing cell which file changed and the size of the difference of every summary metric. Run it before and
after every simulator change, and on the Mac and on Alpine before pooling results from the two machines (compare their
manifests with ``check --manifest A --against B``). Both refuse to run with EVOLVE_KEEP_DIR set (``run_one`` would
write the outputs where the golden tool does not hash them).

Hashes: ``request_metrics.csv`` as bytes; ``token_gaps.npz`` as its arrays (name, dtype, shape, bytes; sorted by name),
not as the zip container, which embeds timestamps. The CSV is written at %.6f, so ``token_gaps`` is the only
full-precision witness of sub-microsecond timing changes.

Coverage. The default set (6 scenarios x 5 policies) covers the default code paths: no preemption or restart occurs in
it (verified on 3 cells), no router delay, exact replica views, and no policy defines ``observe``. The opt-in
``extended`` set adds (a) a Llama-2-7B 8-replica Tool window past the knee where preemption restarts occur (``build``
fails unless every cell of it has restarted requests), under built-in and policy-file routers, (b) a policy with
``observe()`` (``evolve/golden/policies/feedback.py``) on the default knee and on the preemption window, and (c) cells
with ``route_delay_s > 0`` and ``view_refresh_s > 0``. The (c) cells are PENDING: their semantics were being fixed on a
separate branch on 2026-10-10, so their reference must be built from the fixed code (``build --set extended
--include-pending``) and is not in ``v4ext_<host>.json``. Not covered at all: the H100 device, workload transforms, the
learned/RL scheduler (also outside ``code_version``) and, under EVOLVE_V3_DATA, the 128K context cap.
"""
import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import math
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
# Opt-in extended set. PREEMPT: Llama-2-7B (122,752 KV tokens per replica) past the knee; under LOR 183 of 2,000
# requests are preempted and restarted, 300 restarts, P99 E2E 45 s (probe 2026-10-10, v4 data). Meta-Llama-3-8B
# showed no restart even at 4 replicas and scale 0.5 (P99 E2E 526 s): its KV capacity is 4x larger.
PREEMPT = ev.Scenario("Tool", 8, 0.75, 2000, 2000, 1, model="7b")
KNEE = SCENARIOS[0]
FEEDBACK = REPO / "evolve" / "golden" / "policies" / "feedback.py"
# name -> (scheduler, policy file, extra simulator flags); names must not collide with POLICIES.
EXT_POLICIES = {
    "feedback": ("evolved", FEEDBACK, ()),
    "lor_rd5ms": ("lor", None, ("--cluster_config_route_delay_s", "0.005")),
    "mooncake_rd5ms": ("evolved", SYSTEMS / "mooncake_min_ttft.py", ("--cluster_config_route_delay_s", "0.005")),
    "feedback_vr1s": ("evolved", FEEDBACK, ("--evolved_global_scheduler_config_view_refresh_s", "1.0")),
}
# (scenario, policy, pending). Pending cells exercise code whose semantics were still being fixed when the set was
# defined; their reference is built only with --include-pending, from the fixed code.
EXTENDED = [
    (PREEMPT, "lor", False),
    (PREEMPT, "lop_uncached", False),
    (PREEMPT, "mooncake", False),
    (PREEMPT, "feedback", False),
    (KNEE, "feedback", False),
    (KNEE, "lor_rd5ms", True),
    (KNEE, "mooncake_rd5ms", True),
    (KNEE, "feedback_vr1s", True),
]
EXPECT_RESTARTS = {PREEMPT.tag}  # build fails unless every cell of these scenarios has restarted requests
SETS = ("default", "extended")
SUMMARY_KEYS = ("completed", "scored", "p50_e2e", "p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e",
                "prefill_hit_ratio", "queue_delay_p99", "mean_tbt", "p999_tbt", "n_tbt")
HASH_FIELDS = ("request_metrics_sha256", "token_gaps_sha256")


def policy_spec(name: str) -> tuple:
    """(scheduler, policy file or None, extra simulator flags) for a policy name of either set."""
    if name in POLICIES:
        return POLICIES[name] + ((),)
    return EXT_POLICIES[name]


def cells_for(set_name: str = "default", include_pending: bool = False) -> list:
    if set_name == "default":
        return [(sc, n) for sc in SCENARIOS for n in POLICIES]
    if set_name == "extended":
        return [(sc, n) for sc, n, pending in EXTENDED if include_pending or not pending]
    raise ValueError(f"unknown golden set {set_name!r}")


def default_out(set_name: str, host: str) -> Path:
    """evolve/golden/<v3|v4>[ext]_<host>.json; v3 when the harness reads the v3 traces (EVOLVE_V3_DATA=1)."""
    data = "v3" if ev.V3_DATA else "v4"
    return OUT_DIR / f"{data}{'' if set_name == 'default' else 'ext'}_{host}.json"


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


def environment(cells) -> dict:
    import pandas
    import sklearn
    cells = list(cells)
    pols = {n: policy_spec(n)[1] for _, n in cells}
    return {
        "code_version": ev.code_version(),
        "host": socket.gethostname().split(".")[0],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
        "v3_data": ev.V3_DATA,
        "bundles": bundle_keys(sc.model for sc, _ in cells),
        "traces": {t: ev.file_sha(p)[:16] for t, p in ev.TRACES.items()},
        "policies": {n: ev.file_sha(p)[:16] for n, p in sorted(pols.items()) if p},
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


def is_hash(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(ch in "0123456789abcdef" for ch in x)


def restart_counts(path) -> dict:
    """Requests with at least one preemption restart, and all restarts, from a request_metrics.csv."""
    import pandas as pd
    if "request_num_restarts" not in pd.read_csv(path, nrows=0).columns:
        return {}
    r = pd.read_csv(path, usecols=["request_num_restarts"])["request_num_restarts"]
    return {"restarted_requests": int((r > 0).sum()), "total_restarts": int(r.sum())}


def run_cell(sc, name, keep_root: Path, timeout: int) -> dict:
    sched, pol, extra = policy_spec(name)
    keep = keep_root / f"{sc.tag}__{name}"
    cell = {"scenario": sc.tag, "policy": name, "ok": False, "wall_s": 0.0}
    if extra:
        cell["flags"] = list(extra)
    try:  # a stale output left next to the new one could be hashed instead: cleanup must succeed
        if keep.exists():
            shutil.rmtree(keep)
    except OSError as e:
        cell["reason"] = f"cleanup of {keep} failed: {e}"
        return cell
    if keep.exists():
        cell["reason"] = f"cleanup of {keep} failed: still exists"
        return cell
    _LOCAL.keep = keep
    try:
        res = ev.run_one(sched, sc, str(pol) if pol else None, timeout=timeout, extra=list(extra) or None)
    finally:
        _LOCAL.keep = None
    cell["wall_s"] = round(res.get("wall_s", 0), 1)
    if not res.get("ok"):
        cell["reason"] = res.get("reason")
        return cell
    rm = sorted(keep.rglob("request_metrics.csv"))
    tg = sorted(keep.rglob("token_gaps.npz"))
    if len(rm) != 1 or len(tg) != 1 or rm[0].parent != tg[0].parent:
        cell["reason"] = (f"expected one request_metrics.csv and one token_gaps.npz side by side in {keep}, found "
                          f"{len(rm)} and {len(tg)}")
        return cell
    cell.update(ok=True, request_metrics_sha256=sha_file(rm[0]), token_gaps_sha256=sha_npz(tg[0]),
                summary={k: res[k] for k in SUMMARY_KEYS if k in res}, **restart_counts(rm[0]))
    return cell


def run_all(workers: int, keep_root: Path, timeout: int, cells=None) -> list:
    if os.environ.get("EVOLVE_KEEP_DIR"):
        raise RuntimeError("EVOLVE_KEEP_DIR is set: run_one would write the outputs outside the golden keep dir")
    cells = cells or cells_for("default")
    for sc, _ in cells:  # build window files serially (they are shared and cached)
        ev.window_csv(sc)
    ev.tempfile = _TempfileShim()
    try:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda c: run_cell(c[0], c[1], keep_root, timeout), cells))
    finally:
        ev.tempfile = tempfile


def validate(cells) -> list:
    """Problems that make a run unfit as a reference: failed cells, missing hashes, an expected restart absent."""
    bad = []
    for c in cells:
        key = f'{c["scenario"]} / {c["policy"]}'
        if not c.get("ok"):
            bad.append(f"{key}: failed ({c.get('reason')})")
        elif not all(is_hash(c.get(f)) for f in HASH_FIELDS):
            bad.append(f"{key}: missing hash")
        elif c["scenario"] in EXPECT_RESTARTS and not c.get("restarted_requests"):
            bad.append(f"{key}: no preemption restart occurred (restarted_requests={c.get('restarted_requests')})")
    return bad


# --- comparison ----------------------------------------------------------------------------------------------------

def _delta(k, v, w) -> str:
    num = (int, float)
    if isinstance(v, num) and isinstance(w, num) and not isinstance(v, bool) and not isinstance(w, bool):
        if v == w or (math.isnan(v) and math.isnan(w)):
            return f"{k} Δ0"
        pct = f", {(w / v - 1) * 100:+.3f} %" if v and math.isfinite(v) and math.isfinite(w) else ""
        return f"{k} {v:.6g}->{w:.6g} (Δ{w - v:+.3g}{pct})"
    return f"{k} {v}->{w}"


def compare_report(ref: list, new: list) -> tuple:
    """(differences, cells compared, cells byte-identical) between two lists of cells keyed by (scenario, policy).

    A cell is identical only if it succeeded in both runs and both of its hashes are present, valid and equal: a
    missing or None hash is a mismatch, a cell failed in both runs is reported, and so is a cell present on one side
    only."""
    ri = {(c["scenario"], c["policy"]): c for c in ref}
    ni = {(c["scenario"], c["policy"]): c for c in new}
    diffs, identical = [], 0
    for k, r in ri.items():
        key = f"{k[0]} / {k[1]}"
        n = ni.get(k)
        if n is None:
            diffs.append(f"{key}: missing from the new run")
            continue
        if not r.get("ok") and not n.get("ok"):
            diffs.append(f"{key}: failed in both runs (reference: {r.get('reason')}; new: {n.get('reason')})")
            continue
        if not r.get("ok") or not n.get("ok"):
            diffs.append(f"{key}: ok {r.get('ok')} -> {n.get('ok')} ({n.get('reason') or r.get('reason') or ''})")
            continue
        problems = [f"{f} missing" for f in HASH_FIELDS if not (is_hash(r.get(f)) and is_hash(n.get(f)))]
        problems += [f"{f} differ" for f in HASH_FIELDS if is_hash(r.get(f)) and is_hash(n.get(f)) and r[f] != n[f]]
        if not problems:
            identical += 1
            continue
        rs, ns = r.get("summary", {}), n.get("summary", {})
        names = [x for x in SUMMARY_KEYS if x in rs or x in ns] + sorted((set(rs) | set(ns)) - set(SUMMARY_KEYS))
        moves = [_delta(x, rs.get(x), ns.get(x)) for x in names]
        diffs.append(f"{key}: {', '.join(problems)}; " + ("; ".join(moves) or "no summary metrics"))
    for k in ni.keys() - ri.keys():
        diffs.append(f"{k[0]} / {k[1]}: not in the reference")
    return diffs, len(ri.keys() & ni.keys()), identical


def compare(ref: list, new: list) -> list:
    """Differences between two lists of cells (see compare_report)."""
    return compare_report(ref, new)[0]


def report(ref: list, new: list, suffix: str = "") -> int:
    diffs, compared, identical = compare_report(ref, new)
    if diffs:
        print("\n".join(diffs))
        print(f"{len(diffs)} difference(s); {identical} of {compared} compared cells byte-identical"
              f" (reference {len(ref)} cells, new {len(new)}){suffix}")
        return 1
    if not compared:
        print(f"no cells compared{suffix}")
        return 1
    print(f"all {compared} compared cells byte-identical{suffix}")
    return 0


def env_diffs(a: dict, b: dict) -> list:
    out = []
    for k in ("code_version", "v3_data", "python", "numpy", "pandas", "sklearn", "traces", "policies", "bundles"):
        if k == "v3_data" and (k not in a or k not in b):  # manifests before 2026-10-10 do not record it
            continue
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    return out


# --- CLI -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", help="default: evolve/golden/<v3|v4>[ext]_<host>.json")
    b.add_argument("--set", choices=SETS, default="default")
    b.add_argument("--include-pending", action="store_true",
                   help="include the extended set's pending cells (route_delay_s, view_refresh_s); only on fixed code")
    b.add_argument("--force", action="store_true", help="overwrite an existing manifest")
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
        return report(ref["cells"], new["cells"])

    if os.environ.get("EVOLVE_KEEP_DIR"):
        print("EVOLVE_KEEP_DIR is set: run_one would write the outputs where they are not hashed; unset it")
        return 2
    if a.cmd == "build":
        set_name, pending, ref = a.set, a.include_pending, None
        out = Path(a.out) if a.out else default_out(set_name, socket.gethostname().split(".")[0])
        if out.exists() and not a.force:
            print(f"{out} exists: pass --force to overwrite it, or --out another path")
            return 2
    else:
        ref = json.loads(Path(a.manifest).read_text())
        set_name, pending = ref.get("set", "default"), ref.get("include_pending", False)
    spec = cells_for(set_name, pending)
    env = environment(spec)
    missing = [m for m, v in env["bundles"].items() if not v["present"]]
    if missing and not a.allow_train:
        print(f"prediction bundle missing for {missing}: copy it from the reference machine (or pass --allow-train)")
        return 2
    keep_ctx = contextlib.nullcontext(a.keep) if a.keep else _REAL_TMPDIR(prefix="golden_")
    t0 = time.time()
    with keep_ctx as keep:
        cells = run_all(a.workers, Path(keep), a.timeout, spec)
    wall = time.time() - t0

    if a.cmd == "build":
        manifest = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(wall),
                    "set": set_name, "include_pending": pending, "environment": env, "cells": cells}
        bad = validate(cells)
        if bad:
            out = out.with_name(out.stem + ".failed.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out}: {len(cells)} cells in {wall:.0f} s" + ("; NOT A REFERENCE:\n  " + "\n  ".join(bad)
                                                                    if bad else ""))
        return 1 if bad else 0

    for line in env_diffs(ref["environment"], env):
        print("env  ", line)
    return report(ref["cells"], cells, f" ({wall:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
