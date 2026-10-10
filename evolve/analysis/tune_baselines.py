"""Equal-budget tuning control (readiness plan §5 item 4): an evolution strategy over the cost constants of our
Mooncake and Dynamo re-implementations, scored like an evolve candidate (same stage and objective) against BAR_V4, the
bar before tuning exists. The evolved search uses bar v4t (BAR_V4 + these tuned rules), so the two fitness values are
compared only through bar_eval on one bar, never directly.

If a tuned production rule matches the evolved router, the LLM search added nothing. The budget is counted in
candidate evaluations, and must equal one evolve run's candidate count (e.g. 125 on search3b).

Search space: each parameter is searched in [0, 1] and mapped to its range (log scale where marked). The ES is
a diagonal (mu/mu_w, lambda) strategy with a cross-entropy step-size update and reflected bounds, which keeps it dependency-free.
The defaults (QUEUE_W = 1, ..., the untuned constants) are always evaluated first.

  python -m evolve.analysis.tune_baselines --family dynamo --stage search3b --budget 125 --run-dir evolve/runs/tune_dynamo
  python -m evolve.analysis.tune_baselines --family mooncake --stage search3b --budget 125 --run-dir evolve/runs/tune_mooncake

After both runs, freeze the winners into the bar (this creates bar "v4t" in evolve.evaluate):

  python -m evolve.analysis.tune_baselines --family dynamo --run-dir evolve/runs/tune_dynamo --install
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import re
import threading
from pathlib import Path

import numpy as np

from evolve import evaluate as ev

TUNABLE = ev.REPO / "evolve" / "policies" / "tunable"
FAMILIES = {
    # name: (template, {param: (lo, hi, log, default)})
    "dynamo": (TUNABLE / "dynamo_kv_cost_tuned.py",
               {"PREFILL_SCALE": (0.05, 20.0, True, 1.0), "TEMPERATURE": (0.0, 1.0, False, 0.0)}),
    "mooncake": (TUNABLE / "mooncake_min_ttft_tuned.py",
                 {"QUEUE_W": (0.1, 10.0, True, 1.0), "LEN_QUAD": (0.0, 0.5, False, 0.0),
                  "KV_W": (0.0, 1.0, False, 0.0)}),
}


def to_value(u: float, lo: float, hi: float, log: bool) -> float:
    return lo * (hi / lo) ** u if log else lo + u * (hi - lo)


def to_unit(v: float, lo: float, hi: float, log: bool) -> float:
    return math.log(v / lo) / math.log(hi / lo) if log else (v - lo) / (hi - lo)


def render(template: str, params: dict) -> str:
    src = template
    for k, v in params.items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"{k} not found in template")
    return src


def reflect(u: np.ndarray) -> np.ndarray:
    u = np.mod(u, 2.0)
    return np.where(u > 1.0, 2.0 - u, u)


def lam_of(dim: int) -> int:
    return max(4, 4 + int(3 * math.log(dim)))


def ask_tell(budget: int, dim: int, evaluate, seed: int = 0, lam: int = 0):
    """Diagonal (mu/mu_w, lambda)-ES in [0, 1]^dim. evaluate(list of unit vectors) -> list of fitness (higher is
    better). Returns the history of (u, fitness)."""
    rng = np.random.default_rng(seed)
    lam = lam or lam_of(dim)
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mean, sigma = np.full(dim, 0.5), np.full(dim, 0.3)
    hist = []
    while len(hist) < budget:
        n = min(lam, budget - len(hist))
        pop = [reflect(mean + sigma * rng.standard_normal(dim)) for _ in range(n)]
        hist += list(zip(pop, evaluate(pop)))
        if n < lam:
            break
        gen = sorted(hist[-n:], key=lambda x: -x[1])[:mu]
        elite = np.array([u for u, _ in gen])
        new_mean = (w[:, None] * elite).sum(0)
        sigma = np.clip(np.sqrt((w[:, None] * (elite - mean) ** 2).sum(0)), 0.02, 0.5)  # cross-entropy update
        mean = new_mean
    return hist


def install(family: str, run: Path) -> Path:
    """Write run/best.json's constants into evolve/policies/tuned/<template>_best.py, with provenance in a header."""
    template_path, space = FAMILIES[family]
    best = json.loads((run / "best.json").read_text())
    n = len(load_history(run / "history.jsonl"))
    if best["code_version"] != ev.code_version():
        raise SystemExit(f"best.json was scored under code_version {best['code_version']}, not {ev.code_version()}")
    header = (f"# Frozen by evolve.analysis.tune_baselines --install from {run.name}/best.json: fitness {best['fitness']:+.4f} "
              f"after {n} evaluations,\n# code_version {best['code_version']}, params {best['params']}.\n")
    out = ev.TUNED / f"{template_path.stem}_best.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + render(template_path.read_text(), best["params"]))
    return out


def load_history(log: Path) -> dict:
    """Every recorded evaluation, by params key, failed ones included, so a resumed ES replays exactly what an
    uninterrupted run would have done. A job's time-limit kill takes the tuner down with its simulations, so it
    leaves no record; a failed record is a real failure (sim timeout, one simulation OOM-killed) that the
    uninterrupted run saw too, and retrying it could change every later generation (review D_r3).

    A truncated last line (killed mid-write) is removed and a missing final newline is added, so later appends stay
    parseable."""
    done = {}
    if not log.exists():
        return done
    lines = log.read_text().splitlines()
    good = []
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            continue
        good.append(line)
        done[json.dumps(r["params"], sort_keys=True)] = r
    text = "".join(l + "\n" for l in good)
    if log.read_text() != text:
        tmp = log.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(log)
    return done


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--stage", default="search3b", choices=sorted(ev.STAGES))
    ap.add_argument("--objective", default="tail4")
    ap.add_argument("--budget", type=int, default=125, help="candidate evaluations, defaults included")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1, help="ES candidates evaluated at once (each uses --workers)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--install", action="store_true", help="freeze run-dir/best.json into evolve/policies/tuned/")
    a = ap.parse_args()
    if a.install:
        print(f"wrote {install(a.family, Path(a.run_dir))}")
        return

    template_path, space = FAMILIES[a.family]
    template = template_path.read_text()
    names = list(space)
    run = Path(a.run_dir)
    (run / "candidates").mkdir(parents=True, exist_ok=True)
    log = run / "history.jsonl"
    bar = [b if not b.startswith("policy:") else f"{Path(b[7:]).name}:{ev.file_sha(Path(b[7:]))[:12]}"
           for b in ev.BAR_V4]
    meta = {"family": a.family, "stage": a.stage, "objective": a.objective, "seed": a.seed, "budget": a.budget,
            "code_version": ev.code_version(), "bar": bar, "template_sha": ev.file_sha(template_path)[:16],
            "tuner_sha": ev.file_sha(Path(__file__))[:16]}
    if (run / "meta.json").exists():
        old = json.loads((run / "meta.json").read_text())
        if old != meta:
            raise SystemExit(f"run dir was started with {old}, not {meta}")
    else:
        _write_atomic(run / "meta.json", json.dumps(meta, indent=2))
    done = load_history(log)
    loaded = set(done)
    scenarios = ev.STAGES[a.stage]

    lock = threading.Lock()
    gen = [-1]  # ES generation being evaluated (-1 = the defaults); set before each population is scored

    def score(params: dict) -> float:
        key = json.dumps(params, sort_keys=True)
        with lock:
            loaded.discard(key)
            if key in done:
                return done[key]["fitness"]
        # Named by content, so a resumed run never reuses another candidate's file name.
        path = run / "candidates" / f"{a.family}_{hashlib.sha256(key.encode()).hexdigest()[:10]}.py"
        path.write_text(render(template, params))
        res = ev.evaluate_policy(str(path), scenarios, workers=a.workers, baselines=ev.BAR_V4,
                                 objective=a.objective)
        rec = {"params": params, "fitness": res["combined_score"], "valid": res.get("valid", False),
               "file": str(path), "code_version": res.get("code_version") or meta["code_version"], "gen": gen[0]}
        if not rec["valid"]:
            rec["failures"] = {k: v.get("reason") for k, v in res.get("failures", {}).items()}
        with lock:
            done[key] = rec
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"{path.name} {params} fitness {rec['fitness']:+.4f}", flush=True)
        return rec["fitness"]

    def params_of(u):
        return {k: round(to_value(float(x), *space[k][:3]), 6) for k, x in zip(names, u)}

    defaults = {k: space[k][3] for k in names}
    score(defaults)
    ev.ensure_baselines(scenarios, ev.BAR_V4, a.workers)  # once, before candidates run in parallel
    with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        def evaluate(pop):
            gen[0] += 1
            return list(ex.map(lambda u: score(params_of(u)), pop))

        ask_tell(a.budget - 1, len(names), evaluate, seed=a.seed)
    if loaded:  # records the replayed trajectory never asked for: the history is not this run's
        raise SystemExit(f"{len(loaded)} recorded evaluations are not on the replayed ES trajectory")
    valid = [r for r in done.values() if r.get("valid")]
    if not valid:
        raise SystemExit("no valid evaluation")
    best = max(valid, key=lambda r: r["fitness"])
    _write_atomic(run / "best.json", json.dumps(best, indent=2))
    print(f"best {best['params']} fitness {best['fitness']:+.4f} ({len(done)} evaluations)")


if __name__ == "__main__":
    main()
