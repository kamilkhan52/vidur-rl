"""Cross-simulator grid: the same routers on the same windows and arrival draws in LLMServingSim 2.0 and Vidur.

Cells: trace (Tool, Conv) x 6 disjoint 2000-request windows x load (knee, moderate) x draw (p0, p9-p13), and
9 policies per cell. Loads are matched relative to each simulator's own knee (LSS_LOADS from the RR sweep of
2026-10-15; Vidur's A100 8-replica knee is 0.75); ``same`` also runs Vidur at LLMServingSim's time scales.

    # LLMServingSim side (Alpine only; one Slurm array task per cell, 9 runs in parallel)
    python -m evolve.lss2.grid lss-cell --index $SLURM_ARRAY_TASK_ID --out $O/grid --window-dir $W --lss-root $LSS_ROOT
    # Vidur side (resumable; results appended to <out>/vidur_<set>.jsonl)
    nice -n 10 python -m evolve.lss2.grid vidur --set primary --workers 8 --out DIR
    # analysis
    python -m evolve.lss2.grid analyze --lss DIR_WITH_LSS_SUMMARIES --vidur DIR --out DIR
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TRACES = ("Tool", "Conv")
STARTS = (0, 2000, 4000, 6000, 8000, 10000)
LENGTH = 2000
DRAWS = (0, 9, 10, 11, 12, 13)
LOADS = ("knee", "moderate")
# LLMServingSim (8 x RTXPRO6000, Llama-3.1-8B): utilisation 1 at Tool ~1.45, Conv ~1.2 (RR sweep, notes 2026-10-15)
LSS_LOADS = {"Tool": {"knee": 1.5, "moderate": 2.0}, "Conv": {"knee": 1.25, "moderate": 2.0}}
VIDUR_KNEE = 0.75  # Vidur A100 l3_8b 8 replicas (evolve.evaluate protocol v3 calibration)
VIDUR_LOADS = {
    # primary: same position relative to the simulator's own knee (moderate = knee x LSS moderate/knee ratio)
    "primary": {t: {"knee": VIDUR_KNEE, "moderate": round(VIDUR_KNEE * LSS_LOADS[t]["moderate"] / LSS_LOADS[t]["knee"], 4)}
                for t in TRACES},
    # secondary: LLMServingSim's own time scales (Vidur is lightly loaded there)
    "same": LSS_LOADS,
}
# l3_8b windows (max_request_tokens 131072): evolve.evaluate cache keys, checked against ev.window_csv on the Mac
WINDOW_KEYS = {
    ("Tool", 0): "26e6b15c8dbe3740", ("Tool", 2000): "9add6c600362dfe0", ("Tool", 4000): "09e496b85274e3db",
    ("Tool", 6000): "36db0a9cd86828bf", ("Tool", 8000): "5d92be8fabb5dcd7", ("Tool", 10000): "7f2b6f820140500c",
    ("Conv", 0): "44f3f787181d5b53", ("Conv", 2000): "4f60ff551d0ad23d", ("Conv", 4000): "2bc2ec04a25ced9a",
    ("Conv", 6000): "f60583dc99d9ed24", ("Conv", 8000): "14c250fd7639703c", ("Conv", 10000): "07f40ba406afd820",
}
# name -> (LLMServingSim --policy, Vidur scheduler, Vidur policy file or None); files relative to the repo
POLICIES = {
    "rr": ("rr", "round_robin", None),
    "lor": ("lor", "lor", None),
    "sticky_lor": ("sticky_lor", "sticky_lor", None),
    "lop_uncached": ("lop_uncached", "lop_uncached", None),
    "4873": ("evolve/finalists/v4/v4_487363424749.py", "evolved", "evolve/finalists/v4/v4_487363424749.py"),
    "es_lin9_20": ("evolve/finalists/rl_es/es_lin9_20.py", "evolved", "evolve/finalists/rl_es/es_lin9_20.py"),
    "es_mlp9_10": ("evolve/finalists/rl_es/es_mlp9_10.py", "evolved", "evolve/finalists/rl_es/es_mlp9_10.py"),
    "mooncake_tuned": ("evolve/policies/tuned/mooncake_min_ttft_tuned_best.py", "evolved",
                       "evolve/policies/tuned/mooncake_min_ttft_tuned_best.py"),
    "dynamo_tuned": ("evolve/policies/tuned/dynamo_kv_cost_tuned_best.py", "evolved",
                     "evolve/policies/tuned/dynamo_kv_cost_tuned_best.py"),
}
REF = "4873"
MEANS = ("mean_e2e", "mean_ttft", "mean_tpot", "mean_tbt")
P99S = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt")
P50S = ("p50_e2e", "p50_ttft", "p50_tpot", "p50_tbt")
METRICS = MEANS + P99S + P50S


def cells():
    return [(t, s, load, p) for t in TRACES for s in STARTS for load in LOADS for p in DRAWS]


def cell_tag(t, s, load, p) -> str:
    return f"{t}_w{s}_{load}_p{p}"


def window_name(t, s) -> str:
    return f"{t}_{s}_{LENGTH}_{WINDOW_KEYS[(t, s)]}.csv"


# ------------------------------------------------------------------ LLMServingSim side (Alpine)

def lss_cell(a) -> None:
    t, s, load, p = cells()[a.index]
    tag = cell_tag(t, s, load, p)
    window = str(Path(a.window_dir) / window_name(t, s))
    if not Path(window).exists():
        raise SystemExit(f"missing window {window}")
    procs = []
    for name, (spec, _, _) in POLICIES.items():
        out = Path(a.out) / tag / name
        if (out / "summary.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        pol = spec if spec in ("rr", "lor", "sticky_lor", "lop_uncached") else str(REPO / spec)
        cmd = [sys.executable, "-m", "evolve.lss2.run", "--policy", pol, "--window", window, "--replicas", "8",
               "--scale", str(LSS_LOADS[t][load]), "--perturbation", str(p), "--out", str(out),
               "--lss-root", a.lss_root]
        log = open(out / "run.txt", "w")
        procs.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=a.lss_root), log))
    for name, pr, log in procs:
        rc = pr.wait()
        log.close()
        print(f"{tag} {name} exit {rc}", flush=True)


def lss_missing(a) -> None:
    """Write the (cell index, policy) pairs without a summary.json, one per line (for a per-run Slurm array)."""
    lines = []
    for i, x in enumerate(cells()):
        for name in POLICIES:
            if not (Path(a.out) / cell_tag(*x) / name / "summary.json").exists():
                lines.append(f"{i} {name}")
    Path(a.list).write_text("".join(l + "\n" for l in lines))
    print(f"{len(lines)} runs missing -> {a.list}")


def lss_run(a) -> None:
    """One (cell, policy) run: line ``index`` of ``--list`` (0-based)."""
    i, name = Path(a.list).read_text().splitlines()[a.index].split()
    t, s, load, p = cells()[int(i)]
    out = Path(a.out) / cell_tag(t, s, load, p) / name
    if (out / "summary.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    spec = POLICIES[name][0]
    pol = spec if spec in ("rr", "lor", "sticky_lor", "lop_uncached") else str(REPO / spec)
    cmd = [sys.executable, "-m", "evolve.lss2.run", "--policy", pol, "--window",
           str(Path(a.window_dir) / window_name(t, s)), "--replicas", "8", "--scale", str(LSS_LOADS[t][load]),
           "--perturbation", str(p), "--out", str(out), "--lss-root", a.lss_root]
    with open(out / "run.txt", "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=a.lss_root)
    print(f"{cell_tag(t, s, load, p)} {name} exit {rc}", flush=True)


# ------------------------------------------------------------------ Vidur side (Mac or Alpine)

def vidur_jobs(vset: str):
    from evolve import evaluate as ev
    jobs = []
    for t, s, load, p in cells():
        sc = ev.Scenario(t, 8, VIDUR_LOADS[vset][t][load], s, LENGTH, p, model="l3_8b")
        for name, (_, sched, pfile) in POLICIES.items():
            jobs.append((f"{vset}|{cell_tag(t, s, load, p)}|{name}", sc, sched, pfile))
    return jobs


def vidur(a) -> None:
    from evolve import evaluate as ev
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"vidur_{a.set}.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["key"])
    jobs = [j for j in vidur_jobs(a.set) if j[0] not in done]
    if a.only:
        jobs = [j for j in jobs if any(x in j[0] for x in a.only.split(","))]
    if a.limit:
        jobs = jobs[: a.limit]
    # window check: the Mac's windows are the l3_8b windows the Alpine side uses
    for (t, s) in WINDOW_KEYS:
        w = ev.window_csv(ev.Scenario(t, 8, 0.75, s, LENGTH, 0, model="l3_8b"))
        if w.name != window_name(t, s):
            raise SystemExit(f"window key mismatch: {w.name} != {window_name(t, s)}")
    print(f"code_version {ev.code_version()}; {len(done)} done, {len(jobs)} to run with {a.workers} workers", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    n = [0]

    def one(job):
        key, sc, sched, pfile = job
        r = ev.run_one(sched, sc, policy=str(REPO / pfile) if pfile else None, timeout=a.timeout)
        r.update({"key": key, "set": a.set, "scale": sc.scale, "tag": sc.tag, "code_version": ev.code_version()})
        with lock:
            with open(res_path, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            n[0] += 1
            if n[0] % 20 == 0 or not r.get("ok"):
                print(f"{n[0]}/{len(jobs)} {time.time() - t0:.0f}s {key} ok={r.get('ok')} {r.get('reason', '')}",
                      flush=True)

    with cf.ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, jobs))
    print(f"finished {n[0]} runs in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ analysis

def _load_lss(root: Path) -> dict:
    out = {}
    for p in root.rglob("summary.json"):
        tag, name = p.parent.parent.name, p.parent.name
        s = json.loads(p.read_text())
        if s.get("ok"):
            out[(tag, name)] = s
    return out


def _load_vidur(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("ok"):
            _, tag, name = r["key"].split("|")
            out[(tag, name)] = r
    return out


def _parse(tag):
    t, w, load, p = tag.split("_")
    return t, int(w[1:]), load, int(p[1:])


def kendall_tau(x, y) -> float:
    n, c, d = len(x), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            if a * b > 0:
                c += 1
            elif a * b < 0:
                d += 1
    return (c - d) / (n * (n - 1) / 2) if n > 1 else float("nan")


def ratios(results: dict) -> dict:
    """(tag, policy) -> {metric: ratio to 4873}."""
    out = {}
    for (tag, name), s in results.items():
        ref = results.get((tag, REF))
        if ref is None:
            continue
        out[(tag, name)] = {m: s[m] / ref[m] for m in METRICS if s.get(m) and ref.get(m)}
    return out


def cluster_boot(rows, n_boot=2000, seed=0):
    """rows: list of (window, log ratio). Geometric mean and a 95 % interval resampling windows."""
    import random
    by = defaultdict(list)
    for w, v in rows:
        by[w].append(v)
    ws = sorted(by)
    allv = [v for w in ws for v in by[w]]
    g = math.exp(sum(allv) / len(allv))
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        pick = [v for w in (rng.choice(ws) for _ in ws) for v in by[w]]
        bs.append(math.exp(sum(pick) / len(pick)))
    bs.sort()
    return g, bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1]


def gen8(r: dict) -> float:
    vals = [r[m] for m in MEANS + P99S if m in r]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if len(vals) == 8 else float("nan")


def analyze(a) -> None:
    lss = _load_lss(Path(a.lss))
    sims = {"lss": ratios(lss)}
    for vset in ("primary", "same"):
        v = _load_vidur(Path(a.vidur) / f"vidur_{vset}.jsonl")
        if v:
            sims[f"vidur_{vset}"] = ratios(v)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # cell-by-cell CSV
    import csv
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim", "trace", "window", "load", "draw", "policy", "gen8"] + list(METRICS))
        for sim, rs in sims.items():
            for (tag, name), r in sorted(rs.items()):
                t, s, load, p = _parse(tag)
                w.writerow([sim, t, s, load, p, name, f"{gen8(r):.4f}"] + [f"{r.get(m, float('nan')):.4f}" for m in METRICS])
    summary = {"n_cells": {k: len({tag for tag, _ in v}) for k, v in sims.items()}}
    # per simulator, trace, load: geometric mean ratio to 4873 with window-clustered bootstrap, wins
    agg = {}
    for sim, rs in sims.items():
        for t in TRACES + ("all",):
            for load in LOADS + ("all",):
                for name in POLICIES:
                    if name == REF:
                        continue
                    for m in METRICS + ("gen8",):
                        rows = []
                        for (tag, n2), r in rs.items():
                            tt, s, ll, p = _parse(tag)
                            if n2 != name or (t != "all" and tt != t) or (load != "all" and ll != load):
                                continue
                            v = gen8(r) if m == "gen8" else r.get(m)
                            if v and v > 0 and not math.isnan(v):
                                rows.append(((tt, s), math.log(v)))
                        if rows:
                            g, lo, hi = cluster_boot(rows)
                            agg[f"{sim}|{t}|{load}|{name}|{m}"] = {
                                "geo": g, "lo": lo, "hi": hi, "n": len(rows),
                                "wins": sum(1 for _, v in rows if v < 0)}
    summary["agg"] = agg
    # cross-simulator agreement per cell (needs both sides): Kendall tau of the policy ordering, sign agreement
    agree = {}
    for vset in ("primary", "same"):
        vs = sims.get(f"vidur_{vset}")
        if not vs:
            continue
        lt = defaultdict(dict)
        vt = defaultdict(dict)
        for (tag, name), r in sims["lss"].items():
            lt[tag][name] = r
        for (tag, name), r in vs.items():
            vt[tag][name] = r
        for m in METRICS + ("gen8",):
            taus, signs, nsign = [], 0, 0
            per_trace_load = defaultdict(list)
            for tag in sorted(set(lt) & set(vt)):
                names = [n for n in POLICIES if n in lt[tag] and n in vt[tag]]
                if len(names) < len(POLICIES):
                    continue
                x = [gen8(lt[tag][n]) if m == "gen8" else lt[tag][n].get(m, float("nan")) for n in names]
                y = [gen8(vt[tag][n]) if m == "gen8" else vt[tag][n].get(m, float("nan")) for n in names]
                if any(math.isnan(v) for v in x + y):
                    continue
                tau = kendall_tau(x, y)
                taus.append(tau)
                tt, s, ll, p = _parse(tag)
                per_trace_load[f"{tt}|{ll}"].append(tau)
                for n, xv, yv in zip(names, x, y):
                    if n == REF:
                        continue
                    nsign += 1
                    signs += (xv < 1) == (yv < 1)
            if taus:
                agree[f"{vset}|{m}"] = {
                    "mean_tau": sum(taus) / len(taus), "median_tau": sorted(taus)[len(taus) // 2], "n_cells": len(taus),
                    "frac_tau_pos": sum(t > 0 for t in taus) / len(taus), "sign_agreement": signs / nsign,
                    "by_trace_load": {k: sum(v) / len(v) for k, v in per_trace_load.items()}}
    summary["agreement"] = agree
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # short text report
    lines = []
    for sim in sims:
        lines.append(f"\n== {sim}: geometric mean ratio to 4873 [95% window-cluster bootstrap] wins/n, all cells")
        for name in POLICIES:
            if name == REF:
                continue
            parts = []
            for m in ("gen8",) + MEANS + P99S + P50S:
                x = agg.get(f"{sim}|all|all|{name}|{m}")
                if x:
                    parts.append(f"{m}={x['geo']:.3f}[{x['lo']:.2f},{x['hi']:.2f}]{x['wins']}/{x['n']}")
            lines.append(f"{name:15s} " + " ".join(parts))
    lines.append("\n== agreement (Kendall tau of the 9-policy ordering per cell; sign = same side of 4873)")
    for k, x in agree.items():
        lines.append(f"{k:24s} tau mean {x['mean_tau']:.3f} median {x['median_tau']:.3f} pos {x['frac_tau_pos']:.2f} "
                     f"sign {x['sign_agreement']:.3f} n {x['n_cells']} " +
                     " ".join(f"{kk}={vv:.2f}" for kk, vv in sorted(x["by_trace_load"].items())))
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cells")
    c = sub.add_parser("lss-cell")
    c.add_argument("--index", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--window-dir", required=True)
    c.add_argument("--lss-root", required=True)
    c = sub.add_parser("lss-missing")
    c.add_argument("--out", required=True)
    c.add_argument("--list", required=True)
    c = sub.add_parser("lss-run")
    c.add_argument("--list", required=True)
    c.add_argument("--index", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--window-dir", required=True)
    c.add_argument("--lss-root", required=True)
    c = sub.add_parser("vidur")
    c.add_argument("--set", choices=tuple(VIDUR_LOADS), default="primary")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--out", required=True)
    c.add_argument("--only", default="", help="comma-separated substrings of job keys")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--timeout", type=int, default=3600)
    c = sub.add_parser("analyze")
    c.add_argument("--lss", required=True)
    c.add_argument("--vidur", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "cells":
        for i, x in enumerate(cells()):
            print(i, cell_tag(*x), LSS_LOADS[x[0]][x[2]], {k: v[x[0]][x[2]] for k, v in VIDUR_LOADS.items()})
    elif a.cmd == "lss-cell":
        lss_cell(a)
    elif a.cmd == "lss-missing":
        lss_missing(a)
    elif a.cmd == "lss-run":
        lss_run(a)
    elif a.cmd == "vidur":
        vidur(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
