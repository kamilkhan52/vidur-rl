"""
Basic statistics for every reinforcement-learning (RL) run on disk (EXP12 LEARNED harness). No simulations.

Discovers every search archive whose rows have task == "rl" (evolve/runs/*/archive.jsonl, evolve/finalists/
*_archive/archive.jsonl, evolve/finalists/*_archive.jsonl; copies are de-duplicated by (run, id), preferring
evolve/runs), then reports per run and per candidate what the files allow: validity, fitness, cost, model and
code version, design CONFIG (algorithm, gamma, hidden sizes, parameters, features, teacher schedule), training
budget, per-episode learner curves from the 5-number summaries, Q drift, loss trend, late-training stability,
action collapse, the replay double-storage indicator (buffer_size_end - decisions), Spearman correlations of
fitness with teacher agreement (imitation) and with the training tails (train/search overlap), and the frozen
finalists' held-out ratios vs the bar, the 5 heuristics and the evolved rule (from evolve/.cache/baselines.json).

--offline adds analyses of the saved replay buffers and per-episode checkpoints (one candidate at a time):
imitation accuracy of the greedy net on buffer states, bandit explained variance per episode chunk, Q spread
across replicas vs residual RMSE, an approximate Bellman residual for DQN designs, and the fraction of states
whose greedy action changes between consecutive checkpoints.

    python -m evolve.analysis.rl_stats [--runs evolve/runs/rl_* ...] [--out evolve/analysis/out/rl_stats] [--offline]

Writes <out>.json (everything) and <out>.md (summary tables and the "not available" list).
"""

import argparse
import ast
import glob
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HEURISTICS = ("sticky_lor", "lor", "round_robin", "random", "lop_uncached")
RULE_FILE = "evolve/finalists/claude_r1_76be082eb0f5.py"  # teacher and v3 reference
METRICS = ("e2e", "ttft", "tpot")
TAG_RE = re.compile(r"^(?P<model>7b|70b|l3_8b|l3_70b)-(?P<R>\d+)R-(?P<trace>Tool|Conv|Synth)-s(?P<scale>[\d.]+)-"
                    r"(?P<win>full|w\d+\+\d+)-p(?P<p>\d+)(?:-(?P<transform>ramp|burst|mix-\w+?))?(?:-j(?P<jit>[\d.]+))?$")
DRIVER_RE = re.compile(r"^\[iter (\d+)\] candidates=(\d+) best=([-\d.]+) \((\w+)\) spent=([\d.]+) USD")
HELDOUT_TRACE = {"v2": "Synth", "loto_conv": "Conv", "loto_synth": "Synth", "loto_tool": "Tool"}
LEARNER_DEFAULTS = {  # learned_global_scheduler.DEFAULTS, used only if that module cannot be imported
    "algorithm": "bandit", "beta": 0.05, "hidden": [64, 64], "lr": 1e-3, "batch_size": 64, "buffer_size": 50000,
    "train_every": 4, "learning_starts": 500, "target_update": 500, "epsilon_start": 0.2, "epsilon_end": 0.02,
    "epsilon_decay_decisions": 20000, "teacher_mix": 0.0, "imitation_weight": 0.0, "teacher_mix_end": None,
    "imitation_weight_end": None, "teacher_decay_decisions": 30000, "imitation_margin": 0.5, "grad_clip": 10.0,
    "reward_clip": 100.0, "persist_buffer": True}

# Flag thresholds (rewards are O(1) in every design)
Q_DRIFT_ABS = 5.0
MONOTONE_EPISODES = 4
LOSS_RISE_RATIO = 1.5
LOSS_TAIL_K = 4
PLATEAU_REL_SLOPE = 0.05
COLLAPSE_SHARE = 0.3
LATE_K = 3

NOT_AVAILABLE = [
    ("Greedy learning curves (fitness of net_ep{k} vs decisions)",
     "Simulations: score training/<id>/net_ep{k}.pt greedily on the search set; rl_heldout.py needs a "
     "--stage loto_search choice (PROTOCOLS[p]['search']) and a loop over checkpoints (gap 5)."),
    ("Training-seed variance (runs before 2026-10-10)",
     "Implemented for new runs: rl_task.train(..., train_seed=s) / train_seeds(); train finalists and key "
     "ablations with >= 5 seeds (gap 3). Old runs have one seed (the episode-0 scenario seed)."),
    ("Gradient norm, TD error, TD vs DQfD loss split, batch Q spread, target drift, replay age (runs before "
     "2026-10-10)",
     "Implemented for new runs with CONFIG log=True (rl_task.train(log=True)): <ckpt>.updates.npz, one row per "
     "log_interval updates; summarised under candidate['logging'] (gap 7). Not recoverable for old runs."),
    ("Inference diagnostics (action share, Q gap, shadow-teacher agreement at deployment)",
     "Let inference write <out>.stats.json (checkpoint_out unused in inference) and accept teacher_path in "
     "inference for logging only (gap 8)."),
    ("Evaluation CIs, P99.9, TBT, per-scenario mean E2E for RL",
     "Re-score RL finalists with >= 3 arrival draws per held-out configuration (cluster_ci.py); keep the full "
     "run_one summary next to details in rl_task.evaluate_design / evaluate.score (gaps 4, 9)."),
    ("Exact per-episode replay chunks for v2/v3 runs, offline tests for rl_claude_r1",
     "rl_claude_r1 kept only the finalist's net_ep7.pt (no buffer, programs on Alpine); rl_loto_r1 had no "
     "buffer persistence. Needs re-runs with persist_buffer (or copying the Alpine run dir)."),
]
OFFLINE_NEEDS_WORK = [
    "Behaviour-cloning capacity curve (fit the design MLP to teacher actions at several widths) - heavier; "
    "separate flag (gap 10a).",
    "Capacity refits and feature-group ablations on the buffers (gaps 10d, 10e): need the feature-group map per "
    "design (only e413's groups are documented).",
    "Current-state candidate mask is not stored in the buffer; Q spread and greedy actions assume every replica "
    "allowed (exact for designs without candidate_mask).",
]


# ----------------------------------------------------------------------------- small helpers

def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def fnum(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}"


def parse_tag(tag):
    m = TAG_RE.match(tag or "")
    return m.groupdict() if m else None


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"rho": None, "n": len(pairs)}
    from scipy.stats import spearmanr

    r = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    rho = float(r.statistic if hasattr(r, "statistic") else r[0])
    return {"rho": rho if math.isfinite(rho) else None, "p": float(r.pvalue), "n": len(pairs)}


def slope(ys):
    pts = [(i, y) for i, y in enumerate(ys) if y is not None and math.isfinite(y)]
    if len(pts) < 2:
        return None
    return float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])


def rel_change(a, b):
    if a is None or b is None:
        return None
    return (b - a) / max(abs(a), 1e-9)


def longest_monotone(ys):
    """Longest run of consecutive episodes moving in one direction: (episodes, direction)."""
    best, cur, dirn, best_dir = 1, 1, 0, 0
    for i in range(1, len(ys)):
        if ys[i] is None or ys[i - 1] is None:
            cur, dirn = 1, 0
            continue
        s = (ys[i] > ys[i - 1]) - (ys[i] < ys[i - 1])
        if s != 0 and s == dirn:
            cur += 1
        else:
            cur, dirn = (2, s) if s != 0 else (1, 0)
        if cur > best:
            best, best_dir = cur, dirn
    return best if ys else 0, {1: "up", -1: "down", 0: None}[best_dir]


def iso(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else None


def geo(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def learner_defaults():
    try:
        from vidur.scheduler.global_scheduler.learned_global_scheduler import DEFAULTS

        return dict(DEFAULTS)
    except Exception:
        return dict(LEARNER_DEFAULTS)


# ----------------------------------------------------------------------------- discovery

def discover(repo, runs_args=None):
    """[(run_name, archive_path, is_primary)] for every archive with task == 'rl' rows."""
    if runs_args:
        paths = []
        for a in runs_args:
            for p in sorted(glob.glob(a)) or [a]:
                p = Path(p)
                paths.append(p / "archive.jsonl" if p.is_dir() else p)
    else:
        paths = [Path(p) for pat in ("evolve/runs/*/archive.jsonl", "evolve/finalists/*/archive.jsonl",
                                     "evolve/finalists/*archive*.jsonl") for p in sorted(glob.glob(str(repo / pat)))]
    out = []
    for p in paths:
        if not p.exists():
            continue
        try:
            is_rl = any(json.loads(line).get("task") == "rl" for line in p.open() if line.strip())
        except Exception:
            continue
        if not is_rl:
            continue
        name = p.parent.name if p.name == "archive.jsonl" else p.stem
        name = re.sub(r"_archive$", "", name)
        primary = p.parent.parent.name == "runs"
        out.append((name, p, primary))
    return out


def load_runs(sources):
    """Merge archive copies per run: primary rows first, then ids found only in copies."""
    by_run = defaultdict(list)
    for name, path, primary in sources:
        by_run[name].append((not primary, str(path), path, primary))
    runs = {}
    for name, srcs in by_run.items():
        srcs.sort()
        rows, seen, extra, used = [], set(), 0, []
        run_dir = None
        for k, (_, _, path, primary) in enumerate(srcs):
            if primary and run_dir is None:
                run_dir = path.parent
            try:
                lines = [json.loads(line) for line in path.open() if line.strip()]
            except Exception as e:
                used.append({"path": str(path), "error": str(e)})
                continue
            n_new = 0
            for r in lines:
                if r.get("task") not in ("rl", None):  # proposal failures carry no task
                    continue
                cid = r.get("id")
                if cid is not None and cid in seen:
                    continue
                if cid is None and k > 0:
                    continue  # proposal failures: only from the first (preferred) source
                rows.append(r)
                n_new += 1
                if cid is not None:
                    seen.add(cid)
            extra += n_new if k > 0 else 0
            used.append({"path": str(path), "rows": len(lines), "rows_added": n_new})
        runs[name] = {"rows": rows, "sources": used, "run_dir": run_dir, "rows_only_in_copies": extra}
    return runs


# ----------------------------------------------------------------------------- design source

def _module_constants(tree):
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(_substitute(node.value, consts))
            except Exception:
                pass
    return consts


def _substitute(node, consts):
    class Sub(ast.NodeTransformer):
        def visit_Name(self, n):
            return ast.copy_location(ast.Constant(consts[n.id]), n) if n.id in consts else n

    return Sub().visit(ast.parse(ast.unparse(node), mode="eval").body)


def design_info(path, defaults):
    """CONFIG (resolved with harness defaults) and contract flags of a design module."""
    info = {"path": None, "config": None, "config_source": None, "functions": None}
    if path is None or not Path(path).exists():
        return info
    info["path"] = str(Path(path).relative_to(REPO)) if str(path).startswith(str(REPO)) else str(path)
    try:
        src = Path(path).read_text()
        tree = ast.parse(src)
        info["n_lines"] = src.count("\n") + 1
        fns = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        info["functions"] = {f: f in fns for f in ("replica_features", "reward", "interval_reward", "candidate_mask")}
        cfg_node = next((n.value for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CONFIG"), None)
        cfg = None
        if cfg_node is not None:
            try:
                cfg = ast.literal_eval(cfg_node)
                info["config_source"] = "literal"
            except Exception:
                try:
                    cfg = ast.literal_eval(_substitute(cfg_node, _module_constants(tree)))
                    info["config_source"] = "literal+module constants"
                except Exception:
                    cfg = None
        if cfg is None:
            import importlib.util

            spec = importlib.util.spec_from_file_location(f"_rl_design_{abs(hash(str(path)))}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cfg = dict(getattr(mod, "CONFIG", {}) or {})
            info["config_source"] = "import"
        if any(isinstance(n, ast.Expr) and "CONFIG.update" in ast.unparse(n) for n in tree.body):
            info["config_source"] += " (CONFIG.update ignored)"
        full = dict(defaults)
        full.update(cfg)
        info["config"] = full
        info["config_explicit_keys"] = sorted(cfg)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def design_path(repo, run, run_dir, row):
    cid = row.get("id")
    cands = []
    if run_dir is not None and cid:
        cands.append(run_dir / "programs" / f"{cid}.py")
    if cid:
        cands.append(repo / "evolve" / "finalists" / f"{run}_{cid}.py")
    seed = get(row, "proposer", "seed")
    if seed:
        cands += [repo / "evolve" / "rl_designs" / f"{seed}.py", repo / "evolve" / "finalists" / f"{seed}.py"]
    for p in cands:
        if p.exists():
            return p, ("seed file today (may differ from the run's copy)" if p.parent.name == "rl_designs" else None)
    return None, None


# ----------------------------------------------------------------------------- checkpoints and episodes

def ckpt_dir(repo, run, run_dir, cid):
    if run_dir is not None and (run_dir / "training" / cid).is_dir():
        return run_dir / "training" / cid
    d = repo / "evolve" / "finalists" / f"{run}_{cid}_ckpt"
    return d if d.is_dir() else None


def ckpt_files(d):
    if d is None:
        return []
    out = []
    for p in d.glob("net_ep*.pt"):
        m = re.match(r"net_ep(\d+)\.pt$", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def ckpt_meta(path):
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        params = int(sum(v.numel() for v in ck["net"].values()))
        return {"path": str(path), "feature_dim": int(ck.get("feature_dim")), "hidden": list(ck.get("hidden", [])),
                "decisions": ck.get("decisions"), "steps": ck.get("steps"), "params": params}
    except Exception as e:
        return {"path": str(path), "error": f"{type(e).__name__}: {e}"}


def episodes_of(row, d):
    """Per-episode records: archive first, else the stats files beside the checkpoints."""
    eps = get(row, "result", "training", "episodes") or []
    if eps:
        return eps, "archive"
    out = []
    tags = {}
    man = Path(d) / "manifest.json" if d is not None else None
    if man is not None and man.exists():  # logged runs: episode tags from the manifest
        try:
            tags = {e["k"]: e["tag"] for e in json.loads(man.read_text()).get("episodes", [])}
        except Exception:
            tags = {}
    for k, p in ckpt_files(d):
        s = Path(str(p) + ".stats.json")
        if s.exists():
            try:
                out.append({"episode": tags.get(k), "k": k, "learner": json.loads(s.read_text())})
            except Exception:
                pass
    return out, ("stats_files" if out else None)


def episode_rows(eps, cap=None):
    rows, prev_dec, prev_steps = [], 0, 0
    for k, e in enumerate(eps):
        L = e.get("learner") or {}
        share = [s for s in (L.get("action_share") or []) if s is not None]
        tot = sum(share)
        if share and tot > 0:
            p = np.array(share) / tot
            nz = p[p > 0]
            H = float(-(nz * np.log(nz)).sum() / math.log(len(p))) if len(p) > 1 else None
            mx = float(p.max())
        else:
            H = mx = None
        dec, steps = L.get("decisions"), L.get("gradient_steps")
        bend, bload = L.get("buffer_size_end"), L.get("buffer_loaded_from_previous_episode")
        r = {"k": e.get("k", k), "tag": e.get("episode"), "p99_e2e": e.get("p99_e2e"), "p99_ttft": e.get("p99_ttft"),
             "p99_tpot": e.get("p99_tpot"), "mean_e2e": e.get("mean_e2e"), "wall_s": e.get("wall_s"),
             "decisions": dec, "decisions_ep": dec - prev_dec if dec is not None else None,
             "gradient_steps": steps, "steps_ep": steps - prev_steps if steps is not None else None,
             "train_seconds": L.get("train_seconds"), "policy_seconds": L.get("policy_seconds"),
             "epsilon_now": L.get("epsilon_now"), "explore_frac": L.get("explore_frac"),
             "teacher_acted_frac": L.get("teacher_acted_frac"), "teacher_mix_now": L.get("teacher_mix_now"),
             "imitation_weight_now": L.get("imitation_weight_now"), "teacher_agreement": L.get("teacher_agreement"),
             "action_entropy_norm": H, "action_max_share": mx,
             "buffer_size_end": bend, "buffer_loaded": bload,
             "unfinished_dropped": L.get("unfinished_decisions_dropped")}
        lg = L.get("logging")
        if lg:  # RL_LOGGING_SPEC block (runs with CONFIG log=True)
            wc = lg.get("who_counts") or {}
            nw = sum(wc.values()) or None
            r.update({"train_seed": L.get("train_seed"), "buffer_start": lg.get("buffer_start"),
                      "buffer_rows_added": lg.get("buffer_rows_added"), "buffer_capacity": lg.get("buffer_capacity"),
                      "who_epsilon_frac": wc.get("epsilon", 0) / nw if nw else None,
                      "who_teacher_frac": wc.get("teacher", 0) / nw if nw else None,
                      "decision_us_p50": get(lg, "decision_latency_us", "p50"),
                      "decision_us_p99": get(lg, "decision_latency_us", "p99"),
                      "features_us_p50": get(lg, "features_latency_us", "p50")})
        for key, nm in (("reward", "reward"), ("loss", "loss"), ("q_mean", "q"), ("interval_reward", "ireward")):
            for s in ("mean", "p5", "p95", "first", "last"):
                r[f"{nm}_{s}"] = get(L, key, s)
        if bend is not None and bload is not None and r["decisions_ep"] is not None and (cap is None or bend < cap):
            r["dup_excess_ep"] = (bend - bload) - r["decisions_ep"]  # only while the buffer is not full
        rows.append(r)
        prev_dec = dec if dec is not None else prev_dec
        prev_steps = steps if steps is not None else prev_steps
    return rows


def learner_diagnostics(er, cfg):
    """Derived diagnostics over the per-episode rows of one candidate."""
    if not er:
        return {}
    last, first = er[-1], er[0]
    col = lambda k: [r.get(k) for r in er]  # noqa: E731
    qm, lm = col("q_mean"), col("loss_mean")
    mono, mono_dir = longest_monotone(qm)
    q_last = last.get("q_last")
    tail = [x for x in lm[-LOSS_TAIL_K:] if x is not None]
    loss_rel_slope = (slope(tail) / max(abs(float(np.mean(tail))), 1e-9)) if len(tail) >= 2 else None
    loss_ratio = (lm[-1] / lm[0]) if (lm and lm[0] and lm[-1] is not None) else None
    late = {}
    if len(er) >= LATE_K:
        for k in ("reward_mean", "loss_mean", "q_mean", "teacher_agreement", "action_max_share", "p99_ttft"):
            a, b = er[-LATE_K].get(k), er[-1].get(k)
            late[k] = rel_change(a, b)
    agree = [x for x in col("teacher_agreement") if x is not None]
    shares = [x for x in col("action_max_share") if x is not None]
    ents = [x for x in col("action_entropy_norm") if x is not None]
    wall = [x for x in col("wall_s") if x is not None]
    tsec = [x for x in col("train_seconds") if x is not None]
    psec = [x for x in col("policy_seconds") if x is not None]
    cap = (cfg or {}).get("buffer_size")
    dup = None
    if last.get("buffer_size_end") is not None and last.get("decisions") is not None:
        full = cap is not None and last["buffer_size_end"] >= cap
        dup = None if full else last["buffer_size_end"] - last["decisions"]
    t2 = er[-2:]
    tails = {m: (float(np.mean([math.log(r[f"p99_{m}"]) for r in t2])) if all(r.get(f"p99_{m}") for r in t2)
                 else None) for m in METRICS}
    out = {
        "n_episodes": len(er),
        "decisions": last.get("decisions"), "gradient_steps": last.get("gradient_steps"),
        "decisions_per_step": (last["decisions"] / last["gradient_steps"]) if last.get("gradient_steps") else None,
        "sim_wall_s": sum(wall) if wall else None, "train_seconds": sum(tsec) if tsec else None,
        "policy_seconds": sum(psec) if psec else None,
        "grad_share_of_wall": (sum(tsec) / sum(wall)) if (wall and tsec and sum(wall) > 0) else None,
        "reward_mean_first_ep": first.get("reward_mean"), "reward_mean_last_ep": last.get("reward_mean"),
        "reward_mean_all_eps": float(np.mean([x for x in col("reward_mean") if x is not None]))
        if any(x is not None for x in col("reward_mean")) else None,
        "reward_within_last_ep": [last.get("reward_first"), last.get("reward_last")],
        "loss_mean_first_ep": lm[0] if lm else None, "loss_mean_last_ep": lm[-1] if lm else None,
        "loss_last_over_first": loss_ratio, "loss_rel_slope_last_k": loss_rel_slope,
        "loss_within_last_ep": [last.get("loss_first"), last.get("loss_last")],
        "q_mean_first_ep": qm[0], "q_mean_last_ep": qm[-1], "q_last_final": q_last,
        "q_slope_per_ep": slope(qm), "q_monotone_eps": mono, "q_monotone_dir": mono_dir,
        "epsilon_final": last.get("epsilon_now"), "explore_frac_final": last.get("explore_frac"),
        "teacher_acted_first": first.get("teacher_acted_frac"), "teacher_acted_final": last.get("teacher_acted_frac"),
        "teacher_agreement_first": agree[0] if agree else None,
        "teacher_agreement_final": last.get("teacher_agreement"),
        "teacher_agreement_max": max(agree) if agree else None,
        "action_entropy_norm_min": min(ents) if ents else None, "action_max_share_max": max(shares) if shares else None,
        "action_max_share_final": last.get("action_max_share"),
        "buffer_size_end": last.get("buffer_size_end"), "dup_excess": dup,
        "dup_excess_per_episode": [r.get("dup_excess_ep") for r in er] if any(
            r.get("dup_excess_ep") is not None for r in er) else None,
        "dup_excess_before_full": sum(r["dup_excess_ep"] for r in er if r.get("dup_excess_ep") is not None)
        if any(r.get("dup_excess_ep") is not None for r in er) else None,
        "unfinished_dropped_total": sum(r["unfinished_dropped"] or 0 for r in er),
        "late_rel_change": late,
        "train_tail_log_p99_last2": tails,
    }
    out["flags"] = {
        "q_drift": (abs(q_last) > Q_DRIFT_ABS) if q_last is not None else None,
        "q_monotone": mono >= MONOTONE_EPISODES,
        "loss_rising": (loss_ratio > LOSS_RISE_RATIO) if loss_ratio is not None else None,
        "loss_plateau": (abs(loss_rel_slope) < PLATEAU_REL_SLOPE) if loss_rel_slope is not None else None,
        "action_collapse": (max(shares) > COLLAPSE_SHARE) if shares else None,
        "dup_storage": (dup > 0) if dup is not None else (
            (out["dup_excess_before_full"] > 0) if out["dup_excess_before_full"] is not None else None),
    }
    return out


# ----------------------------------------------------------------------------- RL_LOGGING_SPEC files

def _finite(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def logging_summary(d):
    """What a logged training dir holds beyond the stats JSONs (RL_LOGGING_SPEC): manifest.json, per-episode
    <ckpt>.updates.npz (per-interval learning signals), traces, probe and held-out files. None if none exist."""
    if d is None:
        return None
    out = {"manifest": None, "episodes": []}
    man = Path(d) / "manifest.json"
    if man.exists():
        try:
            m = json.loads(man.read_text())
            out["manifest"] = {k: m.get(k) for k in ("code_version", "design_sha256", "train_seed", "options",
                                                      "prediction_bundle", "teacher_sha256", "torch", "numpy", "host",
                                                      "learner_sha256", "git_commit", "git_dirty",
                                                      "git_learner_dirty")}
        except Exception as e:
            out["manifest"] = {"error": str(e)}
    any_file = out["manifest"] is not None
    for k, p in ckpt_files(Path(d)):
        e = {"k": k}
        u = Path(str(p) + ".updates.npz")
        if u.exists():
            any_file = True
            z = np.load(u)
            n = z["n_updates"]

            def last(key, z=z):
                return _finite(z[key][-1]) if key in z.files and len(z[key]) else None

            def wmean(key, z=z, n=n):
                v = z[key] if key in z.files else np.array([])
                ok = np.isfinite(v)
                return float(np.average(v[ok], weights=n[ok])) if ok.any() and n[ok].sum() > 0 else None

            e.update({"update_rows": int(len(n)), "updates": int(n.sum()),
                      "td_sd_first": _finite(z["td_sd"][0]), "td_sd_last": last("td_sd"),
                      "td_mean_last": last("td_mean"), "loss_td_mean_last": last("loss_td_mean"),
                      "loss_imit_mean_last": last("loss_imit_mean"),
                      "grad_norm_p95_max": _finite(np.nanmax(z["grad_norm_p95"])), "clip_frac": wmean("clip_frac"),
                      "q_spread_std_last": last("q_spread_std"), "q_top2_gap_last": last("q_top2_gap"),
                      "ev_batch_last": last("ev_batch"), "target_drift_mean": wmean("target_drift"),
                      "replay_age_mean_last": last("replay_age_mean"), "buffer_fill_last": last("buffer_fill")})
            if e["q_top2_gap_last"] is not None and e["td_sd_last"]:
                e["gap_over_td_sd_last"] = e["q_top2_gap_last"] / e["td_sd_last"]  # signal-to-noise of the choice
        for suf in ("trace.npz", "probe.json", "heldout.json"):
            f = Path(str(p) + "." + suf)
            e["has_" + suf.split(".")[0]] = f.exists()
            if f.exists():
                any_file = True
                if suf.endswith(".json"):
                    try:
                        e[suf.split(".")[0]] = json.loads(f.read_text())
                    except Exception as ex:
                        e[suf.split(".")[0]] = {"error": str(ex)}
        out["episodes"].append(e)
    if not any_file:
        return None
    sd = [e.get("td_sd_last") for e in out["episodes"] if e.get("td_sd_last") is not None]
    out["td_sd_trend"] = slope(sd)
    churn = [get(e, "probe", "churn_vs_prev") for e in out["episodes"]]
    out["probe_churn"] = [c for c in churn if c is not None] or None
    return out


# ----------------------------------------------------------------------------- candidates and runs

def candidate_summary(repo, run, run_dir, row, defaults, orphan_dir=None):
    cid = row.get("id")
    d = orphan_dir or (ckpt_dir(repo, run, run_dir, cid) if cid else None)
    c = {"run": run, "id": cid, "parent": row.get("parent"), "valid": bool(row.get("valid")),
         "stage_reached": row.get("stage_reached"), "fitness": row.get("fitness") if row.get("valid") else None,
         "time": iso(row.get("time")), "seed": get(row, "proposer", "seed"),
         "proposer_cost_usd": get(row, "proposer", "cost_usd"), "orphan": orphan_dir is not None}
    if not c["valid"] and orphan_dir is None:
        c["invalid_reason"] = invalid_reason(row)
        return c
    res = row.get("result") or {}
    c.update({"code_version": res.get("code_version"), "protocol": row.get("rl_protocol") or res.get("protocol"),
              "objective": row.get("objective") or res.get("objective"), "eval_wall_s": res.get("eval_wall_s"),
              "train_wall_s": get(res, "training", "train_wall_s"), "mean_log_ratio": res.get("mean_log_ratio"),
              "worst_scenario_log_ratio": res.get("worst_scenario_log_ratio"),
              "mean_e2e_penalty": res.get("mean_e2e_penalty")})
    if res.get("details"):  # search ratios to the protocol's bar
        c["search_vs_bar"] = {m: _ratio_block([x["p99"] / x["best_heuristic_p99"] for x in res["details"]
                                               if x.get("metric") == m and x.get("best_heuristic_p99")])
                              for m in METRICS}
    try:
        dp, note = design_path(repo, run, run_dir, row)
        man = (d / "manifest.json") if d is not None else None
        if dp is None and man is not None and man.exists():  # standalone / logged training dir
            mp = json.loads(man.read_text()).get("design")
            if mp and Path(mp).exists():
                dp, note = Path(mp), "design path from manifest.json"
        di = design_info(dp, defaults)
        if note:
            di["note"] = note
    except Exception as e:
        di = {"error": str(e)}
    c["design"] = di
    cfg = di.get("config") or {}
    files = ckpt_files(d)
    meta = ckpt_meta(files[-1][1]) if files else None
    c["checkpoint"] = meta
    c["checkpoint_dir"] = str(d.relative_to(repo)) if d is not None and str(d).startswith(str(repo)) else (
        str(d) if d else None)
    c["n_checkpoints"] = len(files)
    alg = cfg.get("algorithm")
    hidden = cfg.get("hidden") or (meta or {}).get("hidden")
    c["summary"] = {
        "algorithm": alg, "gamma": ("0 (bandit)" if alg == "bandit" else
                                    (f"exp(-{cfg.get('beta')}*dt)" if alg == "dqn" else None)),
        "beta": cfg.get("beta") if alg == "dqn" else None, "hidden": hidden,
        "n_features": (meta or {}).get("feature_dim"), "params": (meta or {}).get("params"),
        "lr": cfg.get("lr"), "batch_size": cfg.get("batch_size"), "train_every": cfg.get("train_every"),
        "buffer_size": cfg.get("buffer_size"), "persist_buffer": cfg.get("persist_buffer"),
        "epsilon": [cfg.get("epsilon_start"), cfg.get("epsilon_end"), cfg.get("epsilon_decay_decisions")]
        if cfg else None,
        "teacher": ({"mix": [cfg.get("teacher_mix"), cfg.get("teacher_mix_end")],
                     "imitation_weight": [cfg.get("imitation_weight"), cfg.get("imitation_weight_end")],
                     "decay_decisions": cfg.get("teacher_decay_decisions"), "margin": cfg.get("imitation_margin"),
                     "uses_teacher": bool(cfg.get("teacher_mix") or cfg.get("imitation_weight"))} if cfg else None),
        "reward_clip": cfg.get("reward_clip"),
        "has_interval_reward": get(di, "functions", "interval_reward"),
        "has_candidate_mask": get(di, "functions", "candidate_mask"),
    }
    eps, src = episodes_of(row, d)
    er = episode_rows(eps, cfg.get("buffer_size"))  # q_mean.{mean,...} -> q_mean, q_p5, q_p95, q_first, q_last
    c["episodes_source"] = src
    c["episodes"] = er
    c["learner"] = learner_diagnostics(er, cfg)
    c["n_training_seeds"] = 1 if er else None  # torch seed = episode-0 scenario seed, fixed by the protocol
    c["logging"] = logging_summary(d)
    if c["logging"] and get(c["logging"], "manifest", "train_seed") is not None:
        c["train_seed"] = c["logging"]["manifest"]["train_seed"]
    tags = [r["tag"] for r in er if r.get("tag")] + [x.get("scenario") for x in res.get("details", [])]
    c["models"] = sorted({(parse_tag(t) or {}).get("model", "?") for t in tags if t})
    return c


def _ratio_block(rs):
    rs = [r for r in rs if r and r > 0]
    if not rs:
        return None
    return {"geo": geo(rs), "better": sum(r < 1 for r in rs), "n": len(rs), "min": min(rs), "max": max(rs)}


def invalid_reason(row):
    fb = (row.get("feedback") or "").strip().splitlines()
    s = fb[0] if fb else ""
    if row.get("stage_reached") == "proposal":
        return "proposal: " + (s or "no code returned")
    if "source is" in s and "limit" in s:
        return "static guard: source too long"
    if s.startswith("REJECTED by static guard"):
        return "static guard: " + re.sub(r"'[^']*'", "'...'", s.split(":", 1)[-1].strip())[:60]
    if s.startswith("FAILED crash check"):
        fb_all = row.get("feedback") or ""
        err = re.findall(r"(\w+(?:Error|Exception)[^\n]{0,60})", fb_all)
        return "crash check: " + (err[-1] if err else "exit != 0")
    return (s or "unknown")[:80]


def driver_log(run_dir):
    if run_dir is None or not (run_dir / "driver.log").exists():
        return None
    it = {}
    for line in (run_dir / "driver.log").read_text(errors="replace").splitlines():
        m = DRIVER_RE.match(line)
        if m:
            it[int(m.group(1))] = {"iter": int(m.group(1)), "candidates": int(m.group(2)), "best": float(m.group(3)),
                                   "best_id": m.group(4), "spent_usd": float(m.group(5))}
    return [it[k] for k in sorted(it)] or None


def run_summary(name, run, cands, current_cv):
    rows = run["rows"]
    valid = [c for c in cands if c["valid"] and not c["orphan"]]
    fits = [c["fitness"] for c in valid if c["fitness"] is not None]
    times = [r.get("time") for r in rows if r.get("time")]
    best = max(valid, key=lambda c: c["fitness"] if c["fitness"] is not None else -1e18) if valid else None
    cvs = Counter(c.get("code_version") for c in valid)
    models = sorted({m for c in valid for m in c.get("models", [])})
    protos = Counter(c.get("protocol") or "v2" for c in valid)
    objs = Counter(c.get("objective") or "balanced (default)" for c in valid)
    algs = Counter((c.get("summary") or {}).get("algorithm") or "n/a" for c in valid)
    L = [c["learner"] for c in valid if c.get("learner")]
    fl = lambda k: [l["flags"].get(k) for l in L]  # noqa: E731
    count = lambda k: {"true": sum(1 for x in fl(k) if x), "known": sum(1 for x in fl(k) if x is not None)}  # noqa
    qlast = [l["q_last_final"] for l in L if l.get("q_last_final") is not None]
    dups = [l["dup_excess"] for l in L if l.get("dup_excess") is not None]
    fit_of = [c["fitness"] for c in valid if c.get("learner")]
    agree = [c["learner"].get("teacher_agreement_final") for c in valid if c.get("learner")]
    corr = {"fitness_vs_teacher_agreement_final": spearman(fit_of, agree)}
    for m in METRICS:
        corr[f"fitness_vs_train_log_p99_{m}_last2"] = spearman(
            fit_of, [c["learner"]["train_tail_log_p99_last2"].get(m) for c in valid if c.get("learner")])
    by_alg = {}
    for a in sorted(algs):
        sub = [c for c in valid if ((c.get("summary") or {}).get("algorithm") or "n/a") == a]
        f = [c["fitness"] for c in sub if c["fitness"] is not None]
        by_alg[a] = {"n": len(sub), "best": max(f) if f else None, "median": float(np.median(f)) if f else None}
    dlog = driver_log(run["run_dir"])
    warns = []
    if any(not m.startswith("l3_") for m in models):
        warns.append(f"{name}: model(s) {models} - not l3_8b; Llama-2 profile (attention cost flat beyond ~4k KV "
                     "tokens); no number carries over to the Llama-3-8B study")
    stale = [cv for cv in cvs if cv and cv != current_cv]
    if stale:
        warns.append(f"{name}: code_version {stale} != current {current_cv}")
    if dups and max(dups) > 0:
        warns.append(f"{name}: replay double-storage in {sum(d > 0 for d in dups)}/{len(dups)} candidates "
                     f"(buffer_size_end - decisions up to +{max(dups)})")
    warns.append(f"{name}: one training seed per design (torch seed fixed by the protocol)")
    orph = [c for c in cands if c["orphan"]]
    if orph:
        warns.append(f"{name}: {len(orph)} training dir(s) without an archive row (interrupted training): "
                     + ", ".join(c["id"] for c in orph))
    if run["rows_only_in_copies"]:
        warns.append(f"{name}: {run['rows_only_in_copies']} row(s) found only in a finalist copy of the archive")
    no_design = sum(1 for c in valid if not get(c, "design", "config"))
    if no_design:
        warns.append(f"{name}: design source not on disk for {no_design}/{len(valid)} valid candidates "
                     "(CONFIG, algorithm and teacher schedule n/a)")
    cost = sum(c.get("proposer_cost_usd") or 0 for c in cands if not c["orphan"])
    return {
        "name": name, "sources": run["sources"], "run_dir": str(run["run_dir"]) if run["run_dir"] else None,
        "n_candidates": len(rows), "n_valid": len(valid),
        "invalid_reasons": dict(Counter(c.get("invalid_reason") for c in cands if not c["valid"] and not c["orphan"])),
        "n_orphan_training_dirs": len(orph),
        "best_fitness": best["fitness"] if best else None, "best_id": best["id"] if best else None,
        "median_fitness": float(np.median(fits)) if fits else None,
        "fitness_quartiles": [float(x) for x in np.percentile(fits, [25, 50, 75])] if fits else None,
        "by_algorithm": by_alg,
        "proposer_cost_usd": round(cost, 3), "driver_log_last": dlog[-1] if dlog else None,
        "best_so_far": [(d["iter"], d["candidates"], d["best"], d["best_id"]) for d in dlog] if dlog else None,
        "models": models, "not_l3_8b": any(m != "l3_8b" for m in models),
        "code_versions": dict(cvs), "protocols": dict(protos), "objectives": dict(objs),
        "n_training_seeds": 1, "first_row": iso(min(times)) if times else None,
        "last_row": iso(max(times)) if times else None,
        "span_h": (max(times) - min(times)) / 3600 if len(times) > 1 else None,
        "compute_core_h": sum((c.get("train_wall_s") or 0) + (c.get("eval_wall_s") or 0) for c in valid) / 3600,
        "train_wall_s_median": float(np.median([c["train_wall_s"] for c in valid if c.get("train_wall_s")]))
        if any(c.get("train_wall_s") for c in valid) else None,
        "decisions_median": float(np.median([l["decisions"] for l in L if l.get("decisions")]))
        if any(l.get("decisions") for l in L) else None,
        "flag_counts": {k: count(k) for k in ("q_drift", "q_monotone", "loss_rising", "loss_plateau",
                                              "action_collapse", "dup_storage")},
        "q_last_final": {"median": float(np.median(qlast)), "min": min(qlast), "max": max(qlast)} if qlast else None,
        "dup_excess": {"n": len(dups), "n_pos": sum(d > 0 for d in dups), "min": min(dups), "max": max(dups)}
        if dups else None,
        "spearman": corr,
        "warnings": warns,
    }


# ----------------------------------------------------------------------------- held-out finalists

def load_baselines(repo):
    p = repo / "evolve" / ".cache" / "baselines.json"
    if not p.exists():
        return None
    idx = {}
    for k, v in json.loads(p.read_text()).items():
        parts = k.split("|")
        if len(parts) == 4:
            idx[(parts[0], parts[2], parts[3])] = v
    return idx


def _rule_key(repo):
    p = repo / RULE_FILE
    if not p.exists():
        return None
    return "policy:" + hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def heldout_files(repo):
    seen, out = set(), []
    for pat in ("evolve/finalists/heldout*rl_*.json", "evolve/analysis/out/rl_*heldout*.json"):
        for p in sorted(glob.glob(str(repo / pat))):
            h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(Path(p))
    return out


def _group(tag, unseen):
    t = parse_tag(tag) or {}
    if t.get("R") == "16":
        return "16R"
    if t.get("transform"):
        return "stress"
    return "unseen_trace" if t.get("trace") == unseen else "seen_trace"


def heldout_summary(path, repo, cache, rule_key, cands):
    d = json.loads(path.read_text())
    m = re.search(r"(rl_\w+?)_([0-9a-f]{12})\.py$", d.get("design", ""))
    run, cid = (m.group(1), m.group(2)) if m else (None, None)
    cand = cands.get((run, cid), {})
    proto = cand.get("protocol") or ("v2" if d.get("stage") == "heldout" else None)
    unseen = HELDOUT_TRACE.get(proto)
    res = d.get("result") or {}
    cv = d.get("code_version")
    det = res.get("details") or []
    out = {"file": str(path.relative_to(repo)), "run": run, "id": cid, "stage": d.get("stage"), "code_version": cv,
           "protocol": proto, "unseen_trace": unseen, "fitness": res.get("combined_score"),
           "mean_e2e_penalty": res.get("mean_e2e_penalty"), "n_scenarios": len({x["scenario"] for x in det}),
           "vs_bar": {mm: d.get(mm) for mm in METRICS}, "search_fitness": cand.get("fitness")}
    if out["fitness"] is not None and out["search_fitness"] is not None:
        out["search_minus_heldout"] = out["search_fitness"] - out["fitness"]
    ratios = {"vs_heuristics": defaultdict(dict), "vs_rule": defaultdict(dict)}
    vref = d.get("vs_reference") or {}
    vref = next(iter(vref.values())) if vref else {}
    for x in det:
        tag, mm = x["scenario"], x["metric"]
        if cache is not None:
            hs = [cache.get((cv, tag, h)) for h in HEURISTICS]
            if all(h and h.get(f"p99_{mm}") for h in hs):
                ratios["vs_heuristics"][tag][mm] = x["p99"] / min(h[f"p99_{mm}"] for h in hs)
        if tag in vref and mm in vref[tag]:
            ratios["vs_rule"][tag][mm] = vref[tag][mm]
        elif cache is not None and rule_key and cache.get((cv, tag, rule_key)):
            ratios["vs_rule"][tag][mm] = x["p99"] / cache[(cv, tag, rule_key)][f"p99_{mm}"]
    for kind, per in ratios.items():
        if not per:
            out[kind] = None
            continue
        block = {"overall": {mm: _ratio_block([v.get(mm) for v in per.values()]) for mm in METRICS}, "groups": {}}
        for g in ("unseen_trace", "seen_trace", "stress", "16R"):
            sub = [v for t, v in per.items() if _group(t, unseen) == g]
            if sub:
                block["groups"][g] = {mm: _ratio_block([v.get(mm) for v in sub]) for mm in METRICS}
        if kind == "vs_heuristics":  # rough: draws of one configuration are not independent; clustered by -p<k>
            try:
                from evolve.analysis.cluster_ci import boot, clusters

                tags = sorted(t for t in per if all(per[t].get(mm) for mm in METRICS))
                block["cluster_boot_ci95"] = {}
                for mm in METRICS:
                    arr = np.array([math.log(per[t][mm]) for t in tags])
                    g, lo, hi = boot(arr, clusters(tags), n=2000)
                    block["cluster_boot_ci95"][mm] = [lo, hi]
            except Exception as e:
                block["cluster_boot_ci95"] = f"n/a ({type(e).__name__})"
        block["per_scenario"] = {t: v for t, v in sorted(per.items())}
        out[kind] = block
    # held-out scenarios whose arrivals repeat training arrivals (Synth has no jitter unless -j is set)
    train_tags = [r.get("tag") for r in cand.get("episodes", []) if r.get("tag")]
    tr = [parse_tag(t) for t in train_tags]
    insample = []
    for tag in sorted({x["scenario"] for x in det}):
        t = parse_tag(tag) or {}
        if t.get("trace") == "Synth" and not t.get("jit") and any(
                s and s["trace"] == "Synth" and s["scale"] == t.get("scale") and s["R"] == t.get("R") for s in tr):
            insample.append(tag)
    out["arrivals_repeat_training"] = insample
    if not train_tags:
        out["arrivals_repeat_training_note"] = "training episode tags unavailable"
    return out


# ----------------------------------------------------------------------------- offline (buffers + checkpoints)

def _q_all(torch, net, x, bs=16384):
    out = np.empty(x.shape[:2], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out[i:i + bs] = net(torch.from_numpy(np.ascontiguousarray(x[i:i + bs]))).squeeze(-1).numpy()
    return out


def _ev(r, q):
    v = float(np.var(r))
    return 1.0 - float(np.var(r - q)) / v if v > 0 else None


def offline_candidate(c, repo):
    """Buffer/checkpoint analyses for one candidate. Loads one buffer (~0.2-0.5 GB) at a time."""
    import torch

    try:
        from vidur.scheduler.global_scheduler.learned_global_scheduler import _build_net
    except Exception:
        def _build_net(torch, in_dim, hidden):
            layers, d = [], in_dim
            for h in hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
                d = h
            layers.append(torch.nn.Linear(d, 1))
            return torch.nn.Sequential(*layers)

    torch.set_num_threads(4)
    d = repo / c["checkpoint_dir"] if c.get("checkpoint_dir") else None
    files = ckpt_files(d)
    if not files:
        return {"status": "no checkpoints"}
    kmax, pmax = files[-1]
    buf = Path(str(pmax) + ".buffer.npz")
    if not buf.exists():
        return {"status": "no replay buffer (only the last episode keeps one; absent before v3.1)"}
    t0 = time.time()

    def load_net(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = _build_net(torch, int(ck["feature_dim"]), list(ck["hidden"]))
        net.load_state_dict(ck["net"])
        net.eval()
        return net

    nets = {k: load_net(p) for k, p in files}
    z = np.load(buf)
    x, a, r, ta, g = z["x"], z["a"], z["r"].astype(np.float64), z["ta"], z["g"]
    T, N, F = x.shape
    alg = get(c, "summary", "algorithm") or ("bandit" if not np.any(g) else "dqn")
    out = {"status": "ok", "buffer": str(buf.relative_to(repo)) if str(buf).startswith(str(repo)) else str(buf),
           "T": int(T), "N": int(N), "F": int(F),
           "final_ckpt_episode": kmax, "algorithm": alg, "teacher_coverage": float((ta >= 0).mean())}
    # episode chunks from the per-episode buffer_size_end (valid only while the buffer never wrapped)
    eps = c.get("episodes") or []
    ends = [e.get("buffer_size_end") for e in eps]
    loads = [e.get("buffer_loaded") for e in eps]
    chunks = None
    cap = get(c, "summary", "buffer_size")
    if ends and all(v is not None for v in ends) and ends[-1] == T and (cap is None or T < cap) \
            and all(loads[k] == ends[k - 1] for k in range(1, len(ends))) and loads[0] == 0:
        starts = [0] + ends[:-1]
        chunks = list(zip(starts, ends))
    if chunks is None and eps and all(e.get("buffer_start") is not None for e in eps):
        # logged runs: buffer_start / buffer_rows_added per episode stay exact after the buffer wraps; use them when
        # every episode's rows are still contiguous in the final buffer (no wrap during the run)
        rng_ = [(e["buffer_start"], e["buffer_start"] + e["buffer_rows_added"]) for e in eps]
        if rng_[-1][1] == T and all(rng_[k][0] == rng_[k - 1][1] for k in range(1, len(rng_))) and rng_[0][0] == 0:
            chunks = rng_
    out["chunks_exact"] = chunks is not None
    qf = _q_all(torch, nets[kmax], x)
    qa = qf[np.arange(T), a]
    greedy = qf.argmax(1)
    has_t = ta >= 0
    out["imitation_acc_final_net"] = float((greedy[has_t] == ta[has_t]).mean()) if has_t.any() else None
    out["executed_action_eq_teacher"] = float((a[has_t] == ta[has_t]).mean()) if has_t.any() else None
    spread = qf.std(1)
    srt = np.sort(qf, 1)
    out["q_spread_std_mean"] = float(spread.mean())
    out["q_top2_gap_mean"] = float((srt[:, -1] - srt[:, -2]).mean()) if N > 1 else None
    out["q_level_std_across_states"] = float(qf.mean(1).std())
    out["reward_mean"], out["reward_std"] = float(r.mean()), float(r.std())
    if alg == "bandit":
        res = r - qa
        rmse = float(np.sqrt((res ** 2).mean()))
        out.update({"ev_final_net": _ev(r, qa), "residual_rmse": rmse,
                    "advantage_snr": out["q_spread_std_mean"] / rmse if rmse > 0 else None})
    else:  # approximate one-step Bellman residual: the final net stands in for the (unsaved) target net
        nx, nm = z["nx"], z["nm"]
        qn = _q_all(torch, nets[kmax], nx)
        qn = np.where(nm, qn, -np.inf)
        boot = np.where(np.isfinite(qn.max(1)), qn.max(1), 0.0)
        y = r + g * boot
        res = y - qa
        rmse = float(np.sqrt((res ** 2).mean()))
        out.update({"bellman_residual_rmse_approx": rmse, "q_taken_mean": float(qa.mean()), "target_mean": float(y.mean()),
                    "gamma_mean": float(g.mean()), "ev_of_target_approx": _ev(y, qa),
                    "advantage_snr": out["q_spread_std_mean"] / rmse if rmse > 0 else None})
        del nx, nm, qn
    if chunks:
        per = []
        for k, (s, e) in enumerate(chunks):
            if e <= s:
                per.append({"k": k, "n": 0})
                continue
            row = {"k": k, "n": int(e - s)}
            xs, as_, rs, ts = x[s:e], a[s:e], r[s:e], ta[s:e]
            ht = ts >= 0
            row["imit_acc_final_net"] = float((greedy[s:e][ht] == ts[ht]).mean()) if ht.any() else None
            if k in nets:
                qo = _q_all(torch, nets[k], xs)
                row["imit_acc_own_net"] = float((qo.argmax(1)[ht] == ts[ht]).mean()) if ht.any() else None
                if alg == "bandit":
                    row["ev_own_net"] = _ev(rs, qo[np.arange(len(as_)), as_])
            if alg == "bandit":
                row["ev_final_net"] = _ev(rs, qa[s:e])
                if k - 1 in nets:  # out-of-sample: the net before it trained on this episode
                    qp = _q_all(torch, nets[k - 1], xs)
                    row["ev_prev_net"] = _ev(rs, qp[np.arange(len(as_)), as_])
            per.append(row)
        out["per_episode"] = per
    # policy change between consecutive checkpoints on a fixed sample of buffer states
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(T, size=min(T, 20000), replace=False))
    xs = x[idx]
    prev, changes = None, []
    for k in sorted(nets):
        gk = _q_all(torch, nets[k], xs).argmax(1)
        if prev is not None:
            changes.append({"k": k, "frac_greedy_changed": float((gk != prev).mean())})
        prev = gk
    out["policy_change"] = changes
    out["seconds"] = round(time.time() - t0, 1)
    del x, z
    return out


# ----------------------------------------------------------------------------- top level

def summarize(sources, repo=REPO, offline=False, offline_budget_s=900, log=print):
    """Main entry: dict with runs, candidates, finalists, offline, warnings, not_available."""
    defaults = learner_defaults()
    try:
        from evolve import evaluate as ev

        current_cv = ev.code_version()
    except Exception:
        current_cv = None
    runs = load_runs(sources)
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "repo": str(repo), "current_code_version": current_cv,
           "thresholds": {"q_drift_abs": Q_DRIFT_ABS, "monotone_episodes": MONOTONE_EPISODES,
                          "loss_rise_ratio": LOSS_RISE_RATIO, "loss_tail_k": LOSS_TAIL_K,
                          "plateau_rel_slope": PLATEAU_REL_SLOPE, "collapse_share": COLLAPSE_SHARE, "late_k": LATE_K},
           "runs": {}, "candidates": {}, "finalists": [], "offline": {}, "warnings": [],
           "not_available": [{"what": w, "needs": n} for w, n in NOT_AVAILABLE]}
    cands_by_key = {}
    for name in sorted(runs):
        run = runs[name]
        cands = []
        for row in run["rows"]:
            try:
                cands.append(candidate_summary(repo, name, run["run_dir"], row, defaults))
            except Exception as e:
                cands.append({"run": name, "id": row.get("id"), "valid": False, "orphan": False,
                              "invalid_reason": f"rl_stats error: {type(e).__name__}: {e}"})
                out["warnings"].append(f"{name}/{row.get('id')}: rl_stats could not summarise ({e})")
        ids = {r.get("id") for r in run["rows"]}
        if run["run_dir"] is not None and (run["run_dir"] / "training").is_dir():
            for d in sorted((run["run_dir"] / "training").iterdir()):
                if d.is_dir() and d.name not in ids:
                    try:
                        c = candidate_summary(repo, name, run["run_dir"], {"id": d.name, "valid": False}, defaults,
                                              orphan_dir=d)
                        c["invalid_reason"] = f"no archive row; training stopped after {len(c['episodes'])} episodes"
                        cands.append(c)
                    except Exception as e:
                        out["warnings"].append(f"{name}/{d.name}: orphan dir unreadable ({e})")
        rs = run_summary(name, run, cands, current_cv)
        out["runs"][name] = rs
        out["warnings"] += rs.pop("warnings")
        for c in cands:
            key = f"{name}/{c['id']}"
            out["candidates"][key] = c
            if c.get("id"):
                cands_by_key[(name, c["id"])] = c
    cache = load_baselines(repo)
    rule_key = _rule_key(repo)
    for p in heldout_files(repo):
        try:
            h = heldout_summary(p, repo, cache, rule_key, cands_by_key)
            out["finalists"].append(h)
            if h["arrivals_repeat_training"]:
                out["warnings"].append(f"held-out {h['run']}/{h['id']}: {len(h['arrivals_repeat_training'])} Synth "
                                       "scenario(s) repeat training arrivals (Synth is not jittered): "
                                       + ", ".join(h["arrivals_repeat_training"]))
            if h.get("code_version") and current_cv and h["code_version"] != current_cv:
                out["warnings"].append(f"held-out {h['run']}/{h['id']}: code_version {h['code_version']} != "
                                       f"current {current_cv}")
        except Exception as e:
            out["warnings"].append(f"{p}: held-out summary failed ({type(e).__name__}: {e})")
    if offline:
        t0 = time.time()
        todo = [c for c in out["candidates"].values() if c.get("valid") and not c.get("orphan")
                and c.get("n_checkpoints")]
        for i, c in enumerate(todo):
            key = f"{c['run']}/{c['id']}"
            if time.time() - t0 > offline_budget_s:
                out["offline"][key] = {"status": "skipped (time budget)"}
                continue
            try:
                out["offline"][key] = offline_candidate(c, repo)
            except Exception as e:
                out["offline"][key] = {"status": f"error: {type(e).__name__}: {e}"}
            log(f"[offline {i + 1}/{len(todo)}] {key}: {out['offline'][key].get('status')} "
                f"({out['offline'][key].get('seconds', 0)} s)")
        out["offline_seconds"] = round(time.time() - t0, 1)
        for name in out["runs"]:  # imitation / EV vs fitness across candidates
            ks = [k for k in out["offline"] if k.startswith(name + "/") and out["offline"][k].get("status") == "ok"]
            fit = [out["candidates"][k]["fitness"] for k in ks]
            out["runs"][name]["spearman"]["fitness_vs_offline_imitation_acc"] = spearman(
                fit, [out["offline"][k].get("imitation_acc_final_net") for k in ks])
            out["runs"][name]["spearman"]["fitness_vs_offline_advantage_snr"] = spearman(
                fit, [out["offline"][k].get("advantage_snr") for k in ks])
    return out


# ----------------------------------------------------------------------------- markdown

def _rho(s):
    if not s or s.get("rho") is None:
        return f"n/a (n={s.get('n', 0) if s else 0})"
    return f"{s['rho']:+.2f} (n={s['n']})"


def _sched(v):
    if not v:
        return "n/a"
    return f"{v[0]:g} fixed" if v[1] is None else f"{v[0]:g} -> {v[1]:g}"


def _fc(fc):
    return f"{fc['true']}/{fc['known']}" if fc and fc["known"] else "n/a"


def write_md(s, path, offline_requested):
    L = [f"# RL statistics ({s['generated']})", "",
         "Generated by `python -m evolve.analysis.rl_stats` from files on disk (archives, learner stats, checkpoints, "
         "held-out JSONs, baseline cache). No simulations. Full data in the JSON beside this file.",
         f"Current harness code version: `{s['current_code_version']}`.", "",
         "Thresholds: Q drift |q_mean.last of the final episode| > "
         f"{Q_DRIFT_ABS}; monotone Q over >= {MONOTONE_EPISODES} episodes; loss rising = final-episode mean loss / "
         f"first-episode mean loss > {LOSS_RISE_RATIO}; plateau = |slope of mean loss over the last {LOSS_TAIL_K} "
         f"episodes| < {PLATEAU_REL_SLOPE:.0%} of its level per episode; collapse = max action share > "
         f"{COLLAPSE_SHARE} in any episode. All per-episode numbers are within-episode summaries on a different "
         "scenario each episode, with exploration and the teacher acting: they are not learning curves.", ""]
    L += ["## Warnings", ""] + [f"- {w}" for w in s["warnings"]] + [""]
    L += ["## Runs", "",
          "| Run | Protocol | Model | Cand. | Valid | Invalid reasons | Best (id) | Median | Algorithms (n, best) "
          "| Cost $ | Rows (span) | Core-h | Decisions (med) | Seeds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in s["runs"].items():
        inv = "; ".join(f"{k} x{v}" for k, v in r["invalid_reasons"].items()) or "-"
        algs = "; ".join(f"{a} {b['n']}, {fnum(b['best'])}" for a, b in r["by_algorithm"].items())
        model = ", ".join(r["models"]) + (" **(not l3_8b)**" if r["not_l3_8b"] else "")
        L.append(f"| {n} | {', '.join(r['protocols']) or 'n/a'} ({', '.join(r['objectives'])}) | {model} | "
                 f"{r['n_candidates']} | {r['n_valid']} | {inv} | {fnum(r['best_fitness'])} ({r['best_id']}) | "
                 f"{fnum(r['median_fitness'])} | {algs} | {r['proposer_cost_usd']:.2f} | {r['first_row']} to "
                 f"{r['last_row']} ({fnum(r['span_h'], 1)} h) | {fnum(r['compute_core_h'], 1)} | "
                 f"{fnum(r['decisions_median'])} | {r['n_training_seeds']} |")
    L += ["", "Code versions: " + "; ".join(f"{n}: {r['code_versions']}" for n, r in s["runs"].items()), ""]
    L += ["## Learner flags and correlations per run (valid candidates)", "",
          "| Run | Q drift | Q monotone | Loss rising | Loss plateau | Collapse | Final q_mean.last med [min, max] | "
          "Dup excess (n>0/n, range) | rho(fit, teacher agree) | rho(fit, train log P99 TTFT last 2) | "
          "rho(fit, train E2E) | rho(fit, train TPOT) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in s["runs"].items():
        fc, q, dp, sp = r["flag_counts"], r["q_last_final"], r["dup_excess"], r["spearman"]
        qs = f"{q['median']:.2f} [{q['min']:.1f}, {q['max']:.2f}]" if q else "n/a"
        ds = f"{dp['n_pos']}/{dp['n']}, {dp['min']:+d} to {dp['max']:+d}" if dp else "n/a"
        L.append(f"| {n} | {_fc(fc['q_drift'])} | {_fc(fc['q_monotone'])} | {_fc(fc['loss_rising'])} | "
                 f"{_fc(fc['loss_plateau'])} | {_fc(fc['action_collapse'])} | {qs} | {ds} | "
                 f"{_rho(sp['fitness_vs_teacher_agreement_final'])} | {_rho(sp['fitness_vs_train_log_p99_ttft_last2'])} | "
                 f"{_rho(sp['fitness_vs_train_log_p99_e2e_last2'])} | {_rho(sp['fitness_vs_train_log_p99_tpot_last2'])} |")
    L += ["", "Negative rho with a training tail means candidates with lower training-episode P99 score higher; "
          "|rho| near 1 signals that the last training episodes and the search set overlap. A high rho with teacher "
          "agreement signals that fitness is earned by imitating the rule.", ""]
    L += ["## Finalists: held-out", ""]
    for h in s["finalists"]:
        L.append(f"### {h['run']}/{h['id']} ({h['stage']}, protocol {h['protocol']}, unseen trace {h['unseen_trace']})")
        L += ["", f"- File `{h['file']}`, code `{h['code_version']}`, {h['n_scenarios']} scenarios.",
              f"- Fitness: search {fnum(h['search_fitness'])}, held-out {fnum(h['fitness'])} (mean-E2E penalty "
              f"{fnum(h['mean_e2e_penalty'])}); search minus held-out {fnum(h.get('search_minus_heldout'))}.",
              f"- Held-out arrivals that repeat training arrivals: {', '.join(h['arrivals_repeat_training']) or 'none'}.",
              "", "| Ratio of P99 (geo, better/n) | Group | E2E | TTFT | TPOT |", "|---|---|---|---|---|"]

        def cell(b):
            return f"{b['geo']:.3f} ({b['better']}/{b['n']})" if b else "n/a"

        L.append(f"| vs bar (as scored) | all | {cell(h['vs_bar']['e2e'])} | {cell(h['vs_bar']['ttft'])} | "
                 f"{cell(h['vs_bar']['tpot'])} |")
        for kind, lab in (("vs_heuristics", "vs 5 heuristics"), ("vs_rule", "vs rule 76be082eb0f5")):
            b = h.get(kind)
            if not b:
                L.append(f"| {lab} | all | n/a | n/a | n/a |")
                continue
            L.append(f"| {lab} | all | {cell(b['overall']['e2e'])} | {cell(b['overall']['ttft'])} | "
                     f"{cell(b['overall']['tpot'])} |")
            for g, gb in b["groups"].items():
                L.append(f"| {lab} | {g} | {cell(gb['e2e'])} | {cell(gb['ttft'])} | {cell(gb['tpot'])} |")
            ci = b.get("cluster_boot_ci95")
            if isinstance(ci, dict):
                L.append(f"| {lab}: rough 95% cluster-bootstrap CI | all | "
                         + " | ".join(f"[{ci[m][0]:.3f}, {ci[m][1]:.3f}]" for m in METRICS) + " |")
        L.append("")
    L += ["The cluster-bootstrap CI resamples scenario configurations (draws -p<k> grouped); each scenario is one "
          "run of one trained net, so it ignores training-seed and arrival-draw variance.", ""]
    fin_ids = [(h["run"], h["id"]) for h in s["finalists"]]
    if fin_ids:
        L += ["## Finalists: design and training", "",
              "| Finalist | Alg. | Hidden | F | Params | Decisions | Grad steps | Sim wall s | Grad s (share) | "
              "Teacher mix / imit. | Final agree | Final teacher acted | q_mean ep0 -> last (slope) | Loss ep0 -> last | "
              "Late rel. change (loss, Q, agree) | Max share | Dup excess |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for run, cid in fin_ids:
            c = s["candidates"].get(f"{run}/{cid}")
            if not c or not c.get("learner"):
                continue
            sm, ln = c["summary"], c["learner"]
            t = sm.get("teacher") or {}
            lr = ln.get("late_rel_change", {})
            L.append(f"| {run}/{cid} | {sm['algorithm']} | {sm['hidden']} | {sm['n_features']} | {fnum(sm['params'])} | "
                     f"{fnum(ln['decisions'])} | {fnum(ln['gradient_steps'])} | {fnum(ln['sim_wall_s'])} | "
                     f"{fnum(ln['train_seconds'], 0)} ({fnum(ln['grad_share_of_wall'])}) | {_sched(t.get('mix'))} / "
                     f"{_sched(t.get('imitation_weight'))} | {fnum(ln['teacher_agreement_final'])} | "
                     f"{fnum(ln['teacher_acted_final'])} | {fnum(ln['q_mean_first_ep'])} -> {fnum(ln['q_mean_last_ep'])} "
                     f"({fnum(ln['q_slope_per_ep'])}) | {fnum(ln['loss_mean_first_ep'])} -> {fnum(ln['loss_mean_last_ep'])} | "
                     f"{fnum(lr.get('loss_mean'), 2)}, {fnum(lr.get('q_mean'), 2)}, {fnum(lr.get('teacher_agreement'), 2)} | "
                     f"{fnum(ln['action_max_share_max'])} | {fnum(ln['dup_excess'])} |")
        L.append("")
    L += ["## Candidates", "",
          "Valid candidates sorted by fitness, then orphans. Q = final-episode q_mean.last; slope = q_mean.mean per "
          "episode; agree = final teacher agreement; dup = buffer_size_end - decisions; flags: D Q drift, M monotone "
          "Q, R loss rising, P loss plateau, C action collapse, X double storage (for a full buffer: summed per-episode "
          "excess before it filled). M is weak (4 same-direction episode means occur by chance); read it with the "
          "slope.", ""]
    for n in s["runs"]:
        cs = [c for c in s["candidates"].values() if c["run"] == n and (c.get("valid") or c.get("orphan"))]
        cs.sort(key=lambda c: (c.get("orphan", False), -(c.get("fitness") if c.get("fitness") is not None else -1e18)))
        L += [f"### {n}", "", "| id | seed | fitness | alg | hidden | F | params | eps | decisions | agree | "
              "teacher acted | Q | Q slope | loss last | loss trend | max share | dup | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cs:
            sm, ln = c.get("summary") or {}, c.get("learner") or {}
            f = ln.get("flags", {})
            flags = "".join(ch for ch, k in (("D", "q_drift"), ("M", "q_monotone"), ("R", "loss_rising"),
                                             ("P", "loss_plateau"), ("C", "action_collapse"), ("X", "dup_storage"))
                            if f.get(k))
            L.append(f"| {c['id']}{' (orphan)' if c.get('orphan') else ''} | {c.get('seed') or ''} | "
                     f"{fnum(c.get('fitness'))} | {sm.get('algorithm') or 'n/a'} | {sm.get('hidden') or 'n/a'} | "
                     f"{sm.get('n_features') or 'n/a'} | {fnum(sm.get('params'))} | {ln.get('n_episodes', 'n/a')} | "
                     f"{fnum(ln.get('decisions'))} | {fnum(ln.get('teacher_agreement_final'))} | "
                     f"{fnum(ln.get('teacher_acted_final'))} | {fnum(ln.get('q_last_final'), 2)} | "
                     f"{fnum(ln.get('q_slope_per_ep'), 2)} | {fnum(ln.get('loss_mean_last_ep'))} | "
                     f"{fnum(ln.get('loss_rel_slope_last_k'), 2)} | {fnum(ln.get('action_max_share_max'))} | "
                     f"{fnum(ln.get('dup_excess'))} | {flags} |")
        L.append("")
    L += ["## Offline (replay buffers and checkpoints)", ""]
    if not offline_requested:
        L += ["Not run (pass `--offline`).", ""]
    elif s["offline"]:
        L += [f"Ran in {s.get('offline_seconds')} s. EV = 1 - Var(r - Q(x,a)) / Var(r) on the final episode's buffer "
              "(every transition of the run when the buffer never wrapped: mostly in-sample). EV prev = the net "
              "before an episode on that episode's transitions (out-of-sample). SNR = mean std of Q across replicas / "
              "residual RMSE. Imitation = argmax Q == teacher action on buffer states. DQN rows use an approximate "
              "Bellman residual (the final net stands in for the unsaved target net).", "",
              "| Candidate | fitness | alg | T | imitation (final net) | executed = teacher | EV final | EV prev (last 3 ep) "
              "| resid. RMSE | Q spread std | top-2 gap | SNR | greedy change ep k-1 -> k (last 3) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        items = sorted(s["offline"].items(), key=lambda kv: (kv[0].split("/")[0],
                                                             -(s["candidates"][kv[0]].get("fitness") or -1e18)))
        for k, o in items:
            if o.get("status") != "ok":
                L.append(f"| {k} | {fnum(s['candidates'][k].get('fitness'))} | {o.get('status')} |" + " |" * 10)
                continue
            per = o.get("per_episode") or []
            evp = [p.get("ev_prev_net") for p in per[-3:] if p.get("ev_prev_net") is not None]
            pc = [p["frac_greedy_changed"] for p in o.get("policy_change", [])[-3:]]
            rm = o.get("residual_rmse", o.get("bellman_residual_rmse_approx"))
            ev_f = o.get("ev_final_net", o.get("ev_of_target_approx"))
            L.append(f"| {k} | {fnum(s['candidates'][k].get('fitness'))} | {o['algorithm']} | {fnum(o['T'])} | "
                     f"{fnum(o.get('imitation_acc_final_net'))} | {fnum(o.get('executed_action_eq_teacher'))} | "
                     f"{fnum(ev_f)}{'*' if o['algorithm'] != 'bandit' else ''} | "
                     f"{', '.join(f'{v:.2f}' for v in evp) or 'n/a'} | {fnum(rm)} | {fnum(o.get('q_spread_std_mean'))} | "
                     f"{fnum(o.get('q_top2_gap_mean'))} | {fnum(o.get('advantage_snr'))} | "
                     f"{', '.join(f'{v:.2f}' for v in pc) or 'n/a'} |")
        L += ["", "\\* DQN: EV of the approximate Bellman target, not of the reward.", ""]
        L += ["Per run: " + "; ".join(
            f"{n}: rho(fit, imitation) {_rho(r['spearman'].get('fitness_vs_offline_imitation_acc'))}, "
            f"rho(fit, SNR) {_rho(r['spearman'].get('fitness_vs_offline_advantage_snr'))}"
            for n, r in s["runs"].items()), ""]
    L += ["Offline pieces that still need work:", ""] + [f"- {x}" for x in OFFLINE_NEEDS_WORK] + [""]
    L += ["## NOT AVAILABLE (needs new logging or runs)", "", "| Statistic | What it needs |", "|---|---|"]
    L += [f"| {w['what']} | {w['needs'].replace('|', '&#124;')} |" for w in s["not_available"]]
    Path(path).write_text("\n".join(L) + "\n")


def summarize_train_dirs(dirs, repo=REPO, offline=False):
    """Standalone training dirs (no search archive): smoke runs, training/<id>/seed<s>/ of train_seeds()."""
    defaults = learner_defaults()
    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "candidates": {}, "offline": {}, "warnings": []}
    for d in dirs:
        d = Path(d).resolve()
        try:
            c = candidate_summary(repo, d.parent.name, None, {"id": d.name, "valid": False}, defaults, orphan_dir=d)
            c.pop("invalid_reason", None)
            out["candidates"][str(d)] = c
            if offline and c.get("n_checkpoints"):
                out["offline"][str(d)] = offline_candidate(c, repo)
        except Exception as e:
            out["warnings"].append(f"{d}: {type(e).__name__}: {e}")
    return out


def write_train_dirs_md(s, path):
    L = [f"# RL training dirs ({s['generated']})", "", "Generated by `python -m evolve.analysis.rl_stats --train-dir`.",
         ""] + [f"- warning: {w}" for w in s["warnings"]]
    for key, c in s["candidates"].items():
        sm, lg = c.get("summary") or {}, c.get("logging") or {}
        L += [f"## {key}", "", f"algorithm {sm.get('algorithm')}, hidden {sm.get('hidden')}, features "
              f"{sm.get('n_features')}, train_seed {c.get('train_seed')}, code_version "
              f"{get(lg, 'manifest', 'code_version')}, bundle {get(lg, 'manifest', 'prediction_bundle')}, learner "
              f"sha256 {(get(lg, 'manifest', 'learner_sha256') or '?')[:12]}, git {(get(lg, 'manifest', 'git_commit') or '?')[:10]}"
              f"{' (dirty)' if get(lg, 'manifest', 'git_dirty') else ''}", "",
              "| ep | tag | decisions | steps | eps frac | teacher frac | agree | dec p50/p99 us | TD sd first/last "
              "| gap/TD sd | grad p95 max | clip | drift | age | probe agree | churn | held-out P99 TTFT |",
              "|---" * 17 + "|"]
        lge = {e["k"]: e for e in lg.get("episodes", [])}
        for r in c.get("episodes", []):
            e = lge.get(r["k"], {})
            ho = "; ".join(f"{fnum(h.get('p99_ttft'))}" for h in (e.get("heldout") or []))
            L.append(f"| {r['k']} | {r.get('tag')} | {fnum(r.get('decisions_ep'))} | {fnum(r.get('steps_ep'))} | "
                     f"{fnum(r.get('who_epsilon_frac'))} | {fnum(r.get('who_teacher_frac'))} | "
                     f"{fnum(r.get('teacher_agreement'))} | {fnum(r.get('decision_us_p50'), 0)}/"
                     f"{fnum(r.get('decision_us_p99'), 0)} | {fnum(e.get('td_sd_first'))}/{fnum(e.get('td_sd_last'))} | "
                     f"{fnum(e.get('gap_over_td_sd_last'))} | {fnum(e.get('grad_norm_p95_max'))} | "
                     f"{fnum(e.get('clip_frac'))} | {fnum(e.get('target_drift_mean'))} | "
                     f"{fnum(e.get('replay_age_mean_last'), 0)} | {fnum(get(e, 'probe', 'teacher_agreement'))} | "
                     f"{fnum(get(e, 'probe', 'churn_vs_prev'))} | {ho or 'n/a'} |")
        off = s["offline"].get(key)
        if off:
            L += ["", f"Offline: {off.get('status')}, chunks exact {off.get('chunks_exact')}, imitation acc (final net) "
                  f"{fnum(off.get('imitation_acc_final_net'))}, advantage SNR {fnum(off.get('advantage_snr'))}"]
        L.append("")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", action="append", help="summarise standalone training dirs instead of archives")
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs or archive.jsonl files (default: discover all)")
    ap.add_argument("--out", default="evolve/analysis/out/rl_stats", help="writes <out>.json and <out>.md")
    ap.add_argument("--offline", action="store_true", help="also analyse replay buffers and checkpoints")
    ap.add_argument("--offline-budget-s", type=float, default=900)
    a = ap.parse_args()
    t0 = time.time()
    if a.train_dir:
        s = summarize_train_dirs(a.train_dir, REPO, offline=a.offline)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".json").write_text(json.dumps(s, indent=1, default=str))
        write_train_dirs_md(s, str(out) + ".md")
        print(f"wrote {out}.json and {out}.md ({len(s['candidates'])} dirs, {len(s['warnings'])} warnings)")
        return
    sources = discover(REPO, a.runs)
    print("archives:", ", ".join(f"{n} <- {p.relative_to(REPO) if str(p).startswith(str(REPO)) else p}"
                                 for n, p, _ in sources))
    s = summarize(sources, REPO, offline=a.offline, offline_budget_s=a.offline_budget_s)
    s["runtime_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(s, indent=1, default=str))
    write_md(s, str(out) + ".md", a.offline)
    for n, r in s["runs"].items():
        print(f"{n}: {r['n_valid']}/{r['n_candidates']} valid, best {fnum(r['best_fitness'])} ({r['best_id']}), "
              f"Q drift {_fc(r['flag_counts']['q_drift'])}, rho(fit, agree) "
              f"{_rho(r['spearman']['fitness_vs_teacher_agreement_final'])}, dup {r['dup_excess']}")
    print(f"wrote {out}.json and {out}.md in {s['runtime_s']} s")


if __name__ == "__main__":
    main()
