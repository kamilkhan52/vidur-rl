"""
EXP12: evolve the *design* of an RL router (features, reward, learning settings) with the
same loop that evolves router code (evolve/driver.py --task rl).

A candidate is a design module for the LEARNED global scheduler
(vidur/scheduler/global_scheduler/learned_global_scheduler.py). The harness, not the
candidate, owns the learner, the training protocol and the score:

  train   TRAIN_EPISODES in order, each a fresh simulation that starts from the previous
          episode's network (8R windows of the Tool and Conv traces at scale 0.75/1.0,
          perturbations 4-7: arrival seeds never used by any evaluation scenario). The
          frozen evolved finalist is available as a teacher; the design decides whether to
          use it (CONFIG teacher_mix / imitation_weight, both 0 by default).
  score   greedy inference with the trained network on evaluate.STAGE1 (full traces,
          perturbation 1), scored exactly like an evolved router: P99 E2E/TTFT/TPOT
          relative to the per-scenario best heuristic (same fitness, same baselines).
  hold-out  frozen finalists only: evaluate.HELDOUT (perturbations 2-3, the Synth trace
          never seen in training, 16 replicas).

Training compute is part of the cost and is reported (train_wall_s).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from evolve import evaluate as ev

REPO = Path(__file__).resolve().parents[1]
TEACHER = REPO / "evolve" / "finalists" / "claude_r1_76be082eb0f5.py"
SEED_DESIGNS = ["seed_ttft_bandit", "seed_dqn_teacher", "seed_dqn_interval", "seed_dqn_queue"]

S = ev.Scenario
# Protocol v2 (2026-10-01). v1 (8 x 4000-request windows, hardest first) failed at the Tool
# knee: training opened with an exploration collapse, and windows that start with empty queues
# never show the deep backlogs of a full knee run, so the network extrapolated at evaluation.
# v2 is an easy-to-hard curriculum ending with two FULL knee traces. Arrival seeds 4-7 are never
# used for scoring (evaluation uses 1-3); request contents are the same traces (the Synth trace
# and 16 replicas in the held-out set are never trained on).
TRAIN_EPISODES = [
    S("Tool", 8, 1.0, 10000, 4000, 4), S("Conv", 8, 1.0, 6000, 4000, 4),
    S("Tool", 8, 1.0, 2000, 4000, 5), S("Conv", 8, 1.0, 1000, 4000, 5),
    S("Tool", 8, 0.75, 16000, 4000, 6), S("Conv", 8, 0.75, 7000, 4000, 6),
    S("Tool", 8, 0.75, -1, 0, 7), S("Conv", 8, 0.75, -1, 0, 7),
]
# Protocol v3 "loto_conv" (2026-10-02): leave one trace out. The RL design is trained and
# searched on Tool and Synth only, including stress workloads where fixed rules may break
# (load ramps, bursts, two-workload mixes); the Conv trace is never seen until the held-out
# test. The search is scored against the heuristics AND the evolved router (a reference
# policy), so imitating the evolved router earns nothing: fitness > 0 requires beating it.
LOTO_TRAIN_EPISODES = [  # Tool and Synth alternate to the end (r1 ended on three Tool episodes and forgot Synth)
    S("Tool", 8, 1.0, 10000, 4000, 4), S("Synth", 8, 1.0, -1, 0, 4),
    S("Tool", 8, 1.0, 2000, 4000, 5, transform="ramp"), S("Synth", 8, 1.0, -1, 0, 5, transform="burst"),
    S("Tool", 8, 0.75, 16000, 4000, 6), S("Synth", 8, 0.75, -1, 0, 6),
    S("Tool", 8, 0.75, 6000, 4000, 6, transform="mix-Synth"), S("Synth", 8, 0.75, -1, 0, 7, transform="ramp"),
    S("Tool", 8, 0.75, -1, 0, 7, transform="burst"), S("Synth", 8, 0.75, -1, 0, 8, transform="mix-Tool"),
    S("Tool", 8, 0.75, -1, 0, 7), S("Synth", 8, 0.75, -1, 0, 9),
]
LOTO_SEARCH = [
    S("Tool", 8, 1.0, -1, 0, 1), S("Tool", 8, 0.75, -1, 0, 1),
    S("Synth", 8, 1.0, -1, 0, 1), S("Synth", 8, 0.75, -1, 0, 1),
    S("Tool", 8, 0.75, -1, 0, 1, transform="burst"), S("Synth", 8, 0.75, -1, 0, 1, transform="ramp"),
]
LOTO_HELDOUT = (
    [S("Conv", 8, sc, -1, 0, p) for sc in (1.0, 0.75) for p in (1, 2, 3)]
    + [S("Conv", 8, 0.75, -1, 0, 1, transform=x) for x in ("ramp", "burst", "mix-Synth")]
    + [S("Conv", 16, 0.5, -1, 0, p) for p in (1, 2)]
    + [S(tr, 8, 0.75, -1, 0, p) for tr in ("Tool", "Synth") for p in (2, 3)]
    + [S(tr, 16, 0.5, -1, 0, 1) for tr in ("Tool", "Synth")]
)
EVOLVED_REFERENCE = REPO / "evolve" / "finalists" / "claude_r1_76be082eb0f5.py"


def _ep(trace, scale, seed, k, transform=""):
    """A training episode: a 4,000-request window of a large trace, or the whole Synth trace."""
    if trace == "Synth":
        return S("Synth", 8, scale, -1, 0, seed, transform=transform)
    start = (2000, 10000, 6000, 16000)[k % 4] if trace == "Tool" else (1000, 6000, 3000, 7000)[k % 4]
    return S(trace, 8, scale, start, 4000, seed, transform=transform)


def loto_protocol(holdout: str) -> dict:
    """Leave-one-trace-out protocol holding out ``holdout`` (same structure as loto_conv).

    Train: 12 episodes alternating the two remaining traces to the end, easy to hard, with
    ramps, bursts and a two-workload mix; the last two are full traces at the knee. Search: both
    traces at 1.0 and 0.75 plus one stressed scenario each. Held out: everything on ``holdout``
    (8R seeds 1-3, stress at 0.75, 16R) plus new seeds / 16R of the training traces.
    """
    a, b = [t for t in ("Tool", "Conv", "Synth") if t != holdout]
    mix_a, mix_b = f"mix-{b}", f"mix-{a}"
    train = [
        _ep(a, 1.0, 4, 0), _ep(b, 1.0, 4, 0),
        _ep(a, 1.0, 5, 1, "ramp"), _ep(b, 1.0, 5, 1, "burst"),
        _ep(a, 0.75, 6, 2), _ep(b, 0.75, 6, 2),
        _ep(a, 0.75, 6, 3, mix_a), _ep(b, 0.75, 7, 3, "ramp"),
        _ep(a, 0.75, 7, 0, "burst"), _ep(b, 0.75, 8, 1, mix_b),
        S(a, 8, 0.75, -1, 0, 7), S(b, 8, 0.75, -1, 0, 9),
    ]
    search = [S(t, 8, sc, -1, 0, 1) for t in (a, b) for sc in (1.0, 0.75)] + [
        S(a, 8, 0.75, -1, 0, 1, transform="burst"), S(b, 8, 0.75, -1, 0, 1, transform="ramp")]
    heldout = (
        [S(holdout, 8, sc, -1, 0, p) for sc in (1.0, 0.75) for p in (1, 2, 3)]
        + [S(holdout, 8, 0.75, -1, 0, 1, transform=x)
           for x in ("ramp", "burst", "mix-Synth" if holdout != "Synth" else "mix-Tool")]
        + [S(holdout, 16, 0.5, -1, 0, p) for p in (1, 2)]
        + [S(t, 8, 0.75, -1, 0, p) for t in (a, b) for p in (2, 3)]
        + [S(t, 16, 0.5, -1, 0, 1) for t in (a, b)]
    )
    return {"train": train, "search": search, "heldout": heldout,
            "references": (f"policy:{EVOLVED_REFERENCE}",),
            "seeds": ["seed_ttft_bandit", "seed_dqn_teacher", "seed_dqn_queue", "seed_v3_kickstart"]}
PROTOCOLS = {
    "v2": {"train": None, "search": None, "references": (), "seeds": SEED_DESIGNS},
    "loto_synth": loto_protocol("Synth"),
    "loto_tool": loto_protocol("Tool"),
    "loto_conv": {"train": LOTO_TRAIN_EPISODES, "search": LOTO_SEARCH, "heldout": LOTO_HELDOUT,
                  "references": (f"policy:{EVOLVED_REFERENCE}",),
                  "seeds": ["seed_ttft_bandit", "seed_dqn_teacher", "seed_dqn_queue", "seed_v3_kickstart"]},
}

CRASH_EPISODE = S("Tool", 4, 1.5, 12000, 150, 1)
# Plumbing check only (driver --scenarios smoke): two tiny training episodes.
SMOKE_EPISODES = [S("Tool", 4, 1.5, 12000, 300, 4), S("Conv", 4, 2.0, 6000, 300, 4)]


def learned_args(mode: str, ckpt_in: Optional[Path] = None, ckpt_out: Optional[Path] = None,
                 teacher: Optional[Path] = TEACHER) -> List[str]:
    args = ["--learned_global_scheduler_config_mode", mode]
    if ckpt_in:
        args += ["--learned_global_scheduler_config_checkpoint_in", str(ckpt_in)]
    if ckpt_out:
        args += ["--learned_global_scheduler_config_checkpoint_out", str(ckpt_out)]
    if teacher and mode == "training":
        args += ["--learned_global_scheduler_config_teacher_path", str(teacher)]
    return args


def crash_check(design: str, launcher=None) -> dict:
    """Train one tiny episode and run it greedily; catches contract errors in seconds."""
    with tempfile.TemporaryDirectory(prefix="rl_crash_") as tmp:
        ck = Path(tmp) / "net.pt"
        r = ev.run_one("learned", CRASH_EPISODE, policy=design, timeout=900, launcher=launcher,
                       extra=learned_args("training", ckpt_out=ck))
        if not r.get("ok"):
            return r
        if not ck.exists():  # a design whose features were never called cannot be scored
            return {"ok": False, "reason": "no network was created (replica_features never called?)"}
        return ev.run_one("learned", CRASH_EPISODE, policy=design, timeout=900, launcher=launcher,
                          extra=learned_args("inference", ckpt_in=ck))


def _sha256(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


LEARNER = REPO / "vidur" / "scheduler" / "global_scheduler" / "learned_global_scheduler.py"


def _git(*args) -> Optional[str]:
    import subprocess

    try:
        # GIT_OPTIONAL_LOCKS=0: `git status` must not refresh .git/index from parallel driver threads (review F_r2).
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def learner_provenance() -> dict:
    """Which learner produced an RL run. The learner file is excluded from code_version (evaluate._VERSION_EXCLUDE),
    so code_version alone cannot tell two learners apart: record its sha256 (and this harness file's), plus the git
    commit and dirty flags when the repo is a git checkout (None otherwise). Computed at call time."""
    def sha(p):
        try:
            return _sha256(p)
        except OSError:
            return None

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if head is not None else None
    rel = str(LEARNER.relative_to(REPO))
    learner_status = _git("status", "--porcelain", "--", rel) if head is not None else None
    return {"learner_file": rel, "learner_sha256": sha(LEARNER),
            "rl_task_sha256": sha(Path(__file__).resolve()),
            "git_commit": head.strip() if head else None,
            "git_dirty": bool(status.strip()) if status is not None else None,  # tracked files modified
            "git_learner_dirty": bool(learner_status.strip()) if learner_status is not None else None}


def run_design(design: str, work_dir: Path, overrides: dict) -> Path:
    """Copy of ``design`` with CONFIG run options appended (RL_LOGGING_SPEC: train_seed, log, ...). The options go
    through the design's CONFIG (resolve_config validates them), so no simulator flag or vidur/config change is
    needed; the original design file and its hash are untouched (the manifest records both)."""
    src = Path(design).read_text()
    out = work_dir / "design_run.py"
    out.write_text(src + "\n\n# --- appended by evolve.rl_task.train: run options, not part of the design ---\n"
                   f"CONFIG = dict(globals().get('CONFIG') or {{}}, **{overrides!r})\n")
    return out


def _resolved_config(design_file: Path) -> Optional[dict]:
    try:
        import types

        from vidur.scheduler.global_scheduler.learned_global_scheduler import resolve_config

        mod = types.ModuleType("rl_design_manifest")  # exec, not import: no __pycache__ beside the run
        mod.__file__ = str(design_file)
        exec(compile(Path(design_file).read_text(), str(design_file), "exec"), mod.__dict__)
        return resolve_config(getattr(mod, "CONFIG", None))
    except Exception as e:  # recorded, never fatal: the simulator validates the design itself
        return {"error": f"{type(e).__name__}: {e}"}


def write_manifest(work_dir: Path, design: str, run_file: Path, episodes, teacher: Optional[Path],
                   options: dict, bundle: Optional[str] = None, extra: Optional[dict] = None) -> Path:
    """training/<id>/manifest.json (RL_LOGGING_SPEC section 0): everything needed to identify the run."""
    import datetime
    import platform

    import numpy
    import torch

    prov = ev.provenance()
    m = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "code_version": ev.code_version(),
         **learner_provenance(),
         "design": str(design), "design_sha256": _sha256(Path(design)),
         "run_design": str(run_file), "run_design_sha256": _sha256(run_file),
         "config": _resolved_config(run_file),
         "teacher": str(teacher) if teacher else None, "teacher_sha256": _sha256(teacher) if teacher else None,
         "episodes": [{"k": k, "tag": sc.tag, "scenario_seed": sc.seed, "trace": sc.trace, "replicas": sc.replicas,
                       "scale": sc.scale, "window_start": sc.window_start, "window_len": sc.window_len,
                       "perturbation": sc.perturbation, "transform": sc.transform, "model": sc.model}
                      for k, sc in enumerate(episodes)],
         "train_seed": options.get("train_seed"), "options": options, "prediction_bundle": bundle,
         "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
         "torch": torch.__version__, "numpy": numpy.__version__, "pandas": prov.get("pandas"),
         "sklearn": prov.get("sklearn")}
    m.update(extra or {})
    p = work_dir / "manifest.json"
    p.write_text(json.dumps(m, indent=1, default=str))
    return p


def make_probe_set(trace_files, out: Path, n: int = 5000, seed: int = 0) -> Path:
    """Fixed probe set of saved states (RL_LOGGING_SPEC section 2): n decisions with a recorded teacher action, drawn
    from <ckpt>.trace.npz files. Reuse the same file across runs so the probe metrics are comparable."""
    import numpy as np

    xs, tas, ms = [], [], []
    for f in trace_files:
        z = np.load(f)
        if "x" not in z.files:
            continue
        keep = z["ta"] >= 0
        xs.append(z["x"][keep])
        tas.append(z["ta"][keep])
        ms.append(z["mask"][keep])
    x, ta, mask = np.concatenate(xs), np.concatenate(tas), np.concatenate(ms)
    idx = np.sort(np.random.default_rng(seed).choice(len(ta), size=min(n, len(ta)), replace=False))
    np.savez_compressed(out, x=x[idx], ta=ta[idx], mask=mask[idx])
    return out


def probe_metrics(ckpt: Path, probe: Path, prev_greedy=None):
    """Greedy agreement of a checkpoint with the teacher on the probe set, and the fraction of greedy actions that
    changed since the previous checkpoint (policy churn). Returns (metrics, greedy actions). No simulation."""
    import numpy as np
    import torch

    from vidur.scheduler.global_scheduler.learned_global_scheduler import _build_net

    z = np.load(probe)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = _build_net(torch, int(ck["feature_dim"]), list(ck["hidden"]))
    net.load_state_dict(ck["net"])
    net.eval()
    if z["x"].shape[2] != int(ck["feature_dim"]):
        return {"error": f"probe has {z['x'].shape[2]} features, net {ck['feature_dim']}"}, None
    with torch.no_grad():
        q = net(torch.from_numpy(np.ascontiguousarray(z["x"]))).squeeze(-1).numpy()
    q = np.where(z["mask"], q, -np.inf)
    g = q.argmax(1)
    m = {"n": int(len(g)), "teacher_agreement": float((g == z["ta"]).mean()),
         "churn_vs_prev": float((g != prev_greedy).mean()) if prev_greedy is not None else None}
    return m, g


def heldout_eval(design: str, ckpt: Path, scenarios, launcher=None, timeout=3600, workers=1) -> list:
    """Greedy inference of one checkpoint on fixed unseen scenarios (RL_LOGGING_SPEC section 2, plan E3)."""
    import concurrent.futures as cf

    extra = learned_args("inference", ckpt_in=ckpt)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda sc: ev.run_one("learned", sc, design, timeout, launcher, extra), scenarios))
    keys = ("p99_e2e", "p99_ttft", "p99_tpot", "p99_tbt", "mean_e2e", "wall_s")
    return [{"scenario": sc.tag, "ok": bool(r.get("ok")), **({k: r.get(k) for k in keys} if r.get("ok") else
                                                              {"reason": r.get("reason")})}
            for sc, r in zip(scenarios, res)]


def train(design: str, work_dir: Path, episodes=None, launcher=None, timeout=3600, train_seed: Optional[int] = None,
          log: bool = False, log_interval: int = 200, log_trace_frac: float = 1.0, keep_buffers: bool = False,
          heldout=None, probe: Optional[Path] = None, heldout_workers: int = 1, manifest_extra: Optional[dict] = None
          ) -> dict:
    """Run the training episodes in sequence; return per-episode tails and learner diagnostics.

    RL_LOGGING_SPEC options (all off by default; off = exactly the behaviour and outputs of the learner before the
    logging change, i.e. the v4 learner of af87600; not v3: v4 stores each decision once, v3 bandits stored a decision
    twice when its request finished before the next routing decision):
      train_seed      seed of the learner (network initialisation, exploration, batch sampling) separate from the
                      scenario seed; passed through the design CONFIG (see run_design).
      log             learner logs per episode: <ckpt>.updates.npz, <ckpt>.trace.npz, stats["logging"]; also
                      writes <work_dir>/manifest.json and keeps per-episode metadata in the returned log.
      keep_buffers    keep every episode's replay buffer (default: only the newest; the traces hold each
                      episode's rows anyway when log_trace_frac = 1).
      heldout         Scenarios for a greedy held-out evaluation of every checkpoint (costs one simulation per
                      scenario and episode; off by default).
      probe           probe-set file (make_probe_set) for teacher agreement and policy churn per checkpoint.
    """
    episodes = episodes or TRAIN_EPISODES
    work_dir.mkdir(parents=True, exist_ok=True)
    learner = learner_provenance()  # returned with the result (archive rows), whatever the options
    options = {}
    if train_seed is not None:
        options["train_seed"] = int(train_seed)
    if log:
        options.update({"log": True, "log_interval": int(log_interval), "log_trace_frac": float(log_trace_frac)})
    run_file = run_design(design, work_dir, options) if options else None
    policy = str(run_file) if run_file else design
    if options:
        write_manifest(work_dir, design, run_file, episodes, TEACHER, options, extra=manifest_extra)
    prev, log_rows, prev_greedy = None, [], None
    for k, sc in enumerate(episodes):
        ck = work_dir / f"net_ep{k}.pt"
        r = ev.run_one("learned", sc, policy=policy, timeout=timeout, launcher=launcher,
                       extra=learned_args("training", ckpt_in=prev, ckpt_out=ck))
        if not r.get("ok"):
            return {"ok": False, "reason": f"training episode {k} ({sc.tag}): {r.get('reason')}",
                    "stderr": r.get("stderr", ""), "episodes": log_rows, "learner_provenance": learner}
        if prev is not None and not keep_buffers:  # the replay buffer now lives beside the newest checkpoint only
            Path(str(prev) + ".buffer.npz").unlink(missing_ok=True)
            Path(str(prev) + ".buffer.meta.npz").unlink(missing_ok=True)
        stats = json.loads((Path(str(ck) + ".stats.json")).read_text())
        row = {"episode": sc.tag, "p99_e2e": r["p99_e2e"], "p99_ttft": r["p99_ttft"],
               "p99_tpot": r["p99_tpot"], "mean_e2e": r["mean_e2e"], "wall_s": r["wall_s"],
               "learner": stats}
        if options:
            row.update({"scenario_seed": sc.seed, "train_seed": options.get("train_seed"), "bundle": r.get("bundle")})
            if k == 0:
                write_manifest(work_dir, design, run_file, episodes, TEACHER, options, bundle=r.get("bundle"),
                               extra=manifest_extra)
        if heldout:
            row["heldout"] = heldout_eval(policy, ck, heldout, launcher, timeout, heldout_workers)
            Path(str(ck) + ".heldout.json").write_text(json.dumps(row["heldout"], indent=1))
        if probe is not None:
            pm, prev_greedy = probe_metrics(ck, Path(probe), prev_greedy)
            row["probe"] = pm
            Path(str(ck) + ".probe.json").write_text(json.dumps(pm, indent=1))
        log_rows.append(row)
        prev = ck
    end_sha = learner_provenance()["learner_sha256"]
    if end_sha != learner["learner_sha256"]:  # each episode re-imports the learner: an edit mid-run mixes learners
        learner = dict(learner, learner_sha256_end=end_sha, learner_changed_during_run=True)
        print(f"WARNING: {learner['learner_file']} changed during training ({learner['learner_sha256'][:12]} -> "
              f"{end_sha[:12]})", flush=True)
    return {"ok": True, "checkpoint": str(prev), "episodes": log_rows,
            "train_wall_s": sum(e["wall_s"] for e in log_rows), "learner_provenance": learner}


def train_seeds(design: str, work_dir: Path, seeds=(0, 1, 2, 3, 4), **kw) -> dict:
    """Train one design with several training seeds, one after the other, into work_dir/seed<s>/ (>= 5 seeds for
    every finalist and key ablation; report mean +- CI across seeds, never the best seed)."""
    return {int(s): train(design, work_dir / f"seed{int(s)}", train_seed=int(s), **kw) for s in seeds}


def evaluate_design(design: str, work_dir: Path, scenarios=None, workers=4, launcher=None,
                    timeout=3600, episodes=None, protocol: str = "v2", objective: str = "balanced",
                    train_kw: Optional[dict] = None) -> dict:
    proto = PROTOCOLS[protocol]
    scenarios = scenarios or proto["search"] or ev.STAGE1
    episodes = episodes or proto["train"]
    baselines = tuple(ev.DEFAULT_BASELINES) + tuple(proto["references"])
    tr = train(design, work_dir, episodes=episodes, launcher=launcher, timeout=timeout, **(train_kw or {}))
    if not tr["ok"]:
        return {"combined_score": -1e9, "valid": False, "failures": {"training": tr["reason"]},
                "training": tr}
    base = ev.ensure_baselines(scenarios, baselines, workers=workers, launcher=launcher, timeout=timeout)
    extra = learned_args("inference", ckpt_in=Path(tr["checkpoint"]))
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(ev.run_one, "learned", sc, design, timeout, launcher, extra) for sc in scenarios]
        results = [f.result() for f in futs]
    failures = {sc.tag: r for sc, r in zip(scenarios, results) if not r.get("ok")}
    if failures:
        return {"combined_score": -1e9, "valid": False, "failures": failures, "training": tr}
    out = ev.score(dict(zip(scenarios, results)), base, baselines, objective)
    out.update({"valid": True, "code_version": ev.code_version(), "training": tr, "protocol": protocol,
                "eval_wall_s": sum(r["wall_s"] for r in results),
                "scenarios": [sc.tag for sc in scenarios]})
    return out


def feedback_text(res: dict) -> str:
    """Score feedback plus 'reward reflection' (Eureka): what the learner did in training."""
    lines = [ev.feedback_text(res)] if res.get("valid") else [f"INVALID: {res.get('failures')}"]
    tr = res.get("training") or {}
    eps = tr.get("episodes") or []
    if eps:
        lines.append("Training episodes (P99 E2E / TTFT / TPOT in s; learner diagnostics):")
        for e in eps:
            L = e["learner"]
            rw = L.get("reward") or L.get("interval_reward") or {}
            ls, q = L.get("loss") or {}, L.get("q_mean") or {}
            lines.append(
                f"  {e['episode']}: {e['p99_e2e']:.1f} / {e['p99_ttft']:.1f} / {e['p99_tpot']:.3f}; "
                f"steps={L['gradient_steps']} eps={L['epsilon_now']:.3f} explore={L['explore_frac']:.3f} "
                f"teacher_acted={L['teacher_acted_frac']:.3f} agree={L['teacher_agreement']} "
                f"reward mean={rw.get('mean')} first/last={rw.get('first')}/{rw.get('last')} "
                f"loss first/last={ls.get('first')}/{ls.get('last')} Q first/last={q.get('first')}/{q.get('last')} "
                f"action_share={L['action_share']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a systems and reinforcement-learning researcher designing an RL-trained request
router for an LLM inference cluster. You write a DESIGN MODULE that defines what the router
learns from and how; a fixed harness trains it in a simulator and scores it.

## Setting
- Vidur discrete-event simulator; Llama-2-7b replicas (A100, vLLM-v1 scheduler with chunked
  prefill and prefix caching), 122,752 KV-cache tokens per replica.
- Mooncake production traces (conversation and tool/agent). Requests arrive in bursts;
  prompts are long (median ~7k tokens) and share prefixes, so prefix-cache hits matter.
- The router assigns each arriving request to one replica immediately; replicas run FCFS.

## Objective (fixed by the harness)
Minimise TAIL latency: P99 end-to-end, P99 time-to-first-token and P99 time-per-output-token
per scenario, relative to the best of five heuristic routers (sticky-LOR, LOR, round-robin,
random, least-uncached-prefill). Mean E2E more than 5% worse than the best heuristic is
penalised. Fitness 0 = matches the best heuristic on every tail; positive = better.
A hand-designed (evolved) router already reaches about +0.16; its key idea is that a cache
miss is expensive even at light load because its prefill chunks steal steps from requests
that are already decoding, so it prices uncached tokens higher when decodes are resident,
plus convex guards against load imbalance, decode-batch growth and KV pressure.

## The learner (fixed; you configure it)
One MLP with shared weights scores every replica: Q(s, a) = f(replica_features(request, a, ...)).
The router picks argmax Q (epsilon-greedy while training). Algorithms (CONFIG["algorithm"]):
- "bandit": regress Q(s, a) on the decision's own reward (no bootstrapping).
- "dqn": Double DQN on a semi-MDP. Successor state = state at the next routing decision;
  discount exp(-beta * dt) with dt the simulated seconds between decisions; target network
  copied every target_update gradient steps; Huber loss.
A teacher (the evolved router above) is available during training only:
  teacher_mix = probability the teacher's action is executed instead of the learner's;
  imitation_weight / imitation_margin = DQfD large-margin loss towards the teacher's action.
At evaluation the learner acts alone and greedily.
Training protocol: 8 episodes in order, easy to hard: four 4,000-request windows at load 1.0,
two windows at load 0.75 (the knee), then the FULL Tool and Conv traces at 0.75 (deep backlogs).
8 replicas; arrival seeds never used for scoring. Each episode continues from the previous
network (optimizer state kept; replay buffer starts empty each episode).
Scoring: full Tool and Conv traces, 8 replicas, loads 1.0 and 0.75 (~24k and ~12k requests).
About 60k training decisions in total. Backlogs at the knee reach hundreds of thousands of
queued tokens: scale features so they stay O(1) there (e.g. log1p), or the network extrapolates.
Optional: def candidate_mask(request, replicas, ctx) -> list of bools restricts which replicas
the learner may pick (greedy, exploration and the DQN successor argmax), e.g. a safety filter.

## Contract (strict)
Define at module level:
    def replica_features(request, replica, replicas, ctx) -> list of floats
        Same length on every call (1..64). Keep values O(1) (scale them).
    def reward(outcome) -> float                                          (optional*)
        Reward of the decision that routed this request, known when it completes.
    def interval_reward(replicas_before, replicas_after, dt, ctx) -> float (optional*)
        Reward accrued between this decision and the next (dt simulated seconds), e.g.
        -dt * requests-in-system (Bradtke & Duff interval cost; sums to total latency by
        Little's law). Known at the next decision; charges a decision for slowing OTHER
        requests through the bootstrap. Not available to "bandit" credit beyond one step.
    * define at least one; if both, they add. |reward| is clipped at reward_clip.
    CONFIG = {...}   # optional; keys and allowed ranges below
Views are frozen dataclasses:
  request: arrived_at, num_prefill_tokens, session_id, cached_prefix_tokens (tuple, per replica),
           session_last_replica (-1 if new session), position_in_burst, burst_size
  replica / replicas[i]: index, outstanding_requests, pending_prefill_tokens (uncached prompt
           tokens routed here, not yet prefilled), waiting, running, kv_usage (0..1), kv_capacity_tokens
  ctx: rng, now (simulated s), num_replicas, state (dict for memory across calls)
  outcome (in reward): ttft, e2e, tpot (None if one output token), num_prefill_tokens,
           num_decode_tokens, cached_tokens_at_routing, replica, routed_at
Output lengths are NOT visible when routing. Do not make features depend on replica.index
itself (weights are shared across replicas; the index carries no meaning).
CONFIG keys (defaults): algorithm "bandit"|"dqn"; beta 0.05 [0,10]; hidden [64, 64] (1-4 layers,
each 1..512); lr 1e-3 [1e-5,0.1]; batch_size 64 [8,1024]; buffer_size 50000 [1e3,5e5];
train_every 4 [1,64]; learning_starts 500; target_update 500; epsilon_start 0.2, epsilon_end 0.02,
epsilon_decay_decisions 20000; teacher_mix 0.0 [0,1]; imitation_weight 0.0 [0,100];
imitation_margin 0.5; grad_clip 10; reward_clip 100.
Allowed imports: math, heapq, collections, itertools, functools, statistics, bisect.
No file, clock, network, os/sys access, no global/nonlocal, no dunder attributes,
no getattr/eval/exec. Keep per-call work O(replicas).

## Answer format
Read the parent's score AND its training diagnostics (reward and loss trends, Q values,
exploration, teacher agreement, action shares across replicas). Form one concrete hypothesis
about why it learns what it learns, and change the design to test it. Reply with a short
paragraph stating the hypothesis, then the COMPLETE new design module in one ```python block.
"""


SYSTEM_PROMPT_V3_HEADER = """\
You are a systems and reinforcement-learning researcher designing an RL-trained request
router for an LLM inference cluster. You write a DESIGN MODULE that defines what the router
learns from and how; a fixed harness trains it in a simulator and scores it.

## Why RL here (read this first)
Heuristic routers, including a strong hand-evolved one, encode simplifications: fixed
constants tuned on particular traces and loads, a fixed trade-off between cache affinity and
load, linear guesses about how queues grow. Those simplifications break when the workload
changes (load ramps, bursts, two workloads at once, a new trace). An RL router is only worth
its cost if it learns a policy that does NOT rely on such a breakable simplification. An RL
router that merely imitates the evolved router is an expensive heuristic and scores 0 here.
You may use the evolved router as a teacher to start (it is a good prior), but the design
must phase it out and learn where it is wrong.

What we know about where fixed rules break (measured):
- Tails at the knee are made of QUEUEING: under heuristics, 38-94 % of a P99 request's time is
  waiting. The evolved router removes most of it by pricing a cache miss higher as load rises
  (pricing uncached tokens at face value is worse than the best heuristic).
- Its gains depend on a few hand-set mechanisms, each tuned for one regime: a per-replica
  surcharge on cache misses where decodes are resident (removing it doubles P99 TTFT at the
  Tool knee), and memory of a session's previous turn still queued on its replica (removing it
  halves the cache hit ratio on the Synth trace). Its imbalance guards are mis-tuned for Synth.
- It wins TTFT by packing prefix-sharing sessions, which grows decode batches: P99 TPOT is
  1.04-1.23x the best heuristic. A request's TPOT is caused by LATER routing decisions.
- A previous RL search (trained and scored on Tool+Conv) produced a policy that matched the
  evolved router on those traces but failed on the unseen Synth trace: it imitated behaviour
  instead of learning the rule. Reward credit mattered most: rewards that saturate in the
  regime that decides P99, or that are dominated by output length (E2E), carry no signal.
"""


def system_prompt_v3(objective: str = "balanced") -> str:
    from evolve import evaluate as _ev

    w = _ev.OBJECTIVES[objective]
    base = SYSTEM_PROMPT
    contract = base[base.index("## The learner (fixed; you configure it)"):]
    contract = contract.replace(contract[contract.index("Training protocol:"):contract.index("## Contract (strict)")], """\
Training protocol (leave one trace out): 12 episodes in order, easy to hard, alternating the Tool
and Synth traces to the end, including stress workloads: load ramps (+-35 % sinusoidal), bursts (30 % of
requests in 15 s bursts at 3x), and two workloads mixed (Tool + Synth). 8 replicas, loads 1.0
and 0.75, arrival seeds never used for scoring; each episode continues from the previous network
AND the previous replay buffer (CONFIG persist_buffer, default True; buffer_size caps it).
Scoring: Tool and Synth full traces at 1.0 and 0.75, plus Tool-0.75 with bursts and Synth-0.75
with a load ramp. The Conv trace, 16-replica clusters and new arrival seeds are HELD OUT and only
used to test frozen finalists: design for generalisation, not for these six scenarios.
About 60k training decisions. Backlogs at the knee reach hundreds of thousands of queued tokens:
scale features to stay O(1) (e.g. log1p). Teacher phase-out: CONFIG teacher_mix_end and
imitation_weight_end decay the teacher linearly to those values over teacher_decay_decisions
(default 30000); at scoring the learner always acts alone.

""")
    objective_text = f"""## Objective (fixed by the harness)
Fitness weights the per-scenario log P99 ratios {w['e2e']:.2f} * end-to-end + {w['ttft']:.2f} *
time-to-first-token + {w['tpot']:.2f} * time-per-output-token, each relative to the BEST of the five
heuristic routers AND the evolved router, minus half the worst scenario; a mean E2E more than 5 %
worse than the best of them is penalised. Fitness 0 = as good as the best of all of them on every
tail; positive = better than all of them. Imitating the evolved router scores at most 0.

"""
    return SYSTEM_PROMPT_V3_HEADER + "\n" + objective_text + contract
