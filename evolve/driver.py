"""
Evolution driver for routing policies (the automated research loop).

Each iteration: pick a parent by tournament over the best valid candidates, show the
proposer the parent's code and per-scenario tail feedback plus two inspirations, then
screen the proposal through
    stage 0a  static guard (evolve/guard.py)
    stage 0b  crash check on a 150-request window (seconds)
    stage 1   paired tail screen against heuristic baselines (evolve/evaluate.py STAGE1)
Every candidate, rejected or not, is appended to <run_dir>/archive.jsonl with its lineage,
so the full search is auditable and the number of candidates tried is reported.

Usage (after the stage-1 fidelity check has passed; see evolve/README.md):
    python -m evolve.driver --run-dir evolve/runs/r1 --iterations 50 --batch 4 --workers 8
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from evolve import evaluate as ev
from evolve.guard import check_policy_source

REPO = Path(__file__).resolve().parents[1]
SEEDS = ["lor", "sticky_lor", "lop_uncached", "round_robin"]
STAGE0 = [ev.Scenario("Tool", 4, 1.5, 12000, 150, 1)]
# Plumbing check only (seconds per simulation); not a research scenario set.
SMOKE = [ev.Scenario("Tool", 4, 1.5, 12000, 200, 1), ev.Scenario("Conv", 4, 2.0, 6000, 200, 1)]


class Archive:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        (run_dir / "programs").mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "archive.jsonl"
        self.lock = threading.Lock()
        self.rows = [json.loads(l) for l in self.path.read_text().splitlines()] if self.path.exists() else []

    def add(self, row: dict) -> None:
        with self.lock:
            self.rows.append(row)
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    def valid(self) -> List[dict]:
        return [r for r in self.rows if r.get("valid")]

    def best(self) -> Optional[dict]:
        v = self.valid()
        return max(v, key=lambda r: r["fitness"]) if v else None


def save_program(archive: Archive, code: str) -> tuple:
    pid = hashlib.sha256(code.encode()).hexdigest()[:12]
    path = archive.dir / "programs" / f"{pid}.py"
    if not path.exists():
        path.write_text(code)
    return pid, path


def evaluate_candidate(code: str, archive: Archive, parent: Optional[str], stage1, workers, launcher,
                       meta: Optional[dict] = None, task: str = "router") -> dict:
    pid, path = save_program(archive, code)
    row = {"id": pid, "parent": parent, "time": time.time(), "valid": False, "stage_reached": "0a",
           "proposer": meta or {}, "task": task}
    problems = check_policy_source(code, "rl_design" if task == "rl" else "router")
    if problems:
        row["feedback"] = "REJECTED by static guard: " + "; ".join(problems)
        archive.add(row)
        return row
    row["stage_reached"] = "0b"
    if task == "rl":
        from evolve import rl_task

        crash = rl_task.crash_check(str(path), launcher=launcher)
    else:
        crash = ev.run_one("evolved", STAGE0[0], policy=str(path), timeout=600, launcher=launcher)
    if not crash.get("ok"):
        row["feedback"] = f"FAILED crash check: {crash.get('reason')} {crash.get('stderr', '')[-800:]}"
        archive.add(row)
        return row
    row["stage_reached"] = "1"
    if task == "rl":
        from evolve import rl_task

        res = rl_task.evaluate_design(str(path), archive.dir / "training" / pid, stage1, workers=workers,
                                      launcher=launcher,
                                      episodes=rl_task.SMOKE_EPISODES if stage1 == SMOKE else None)
        feedback = rl_task.feedback_text(res)
    else:
        res = ev.evaluate_policy(str(path), stage1, workers=workers, launcher=launcher)
        feedback = ev.feedback_text(res)
    row.update({"valid": res.get("valid", False), "fitness": res["combined_score"],
                "feedback": feedback, "result": res})
    archive.add(row)
    return row


def build_prompt(parent: dict, inspirations: List[dict], archive: Archive, task: str = "router") -> str:
    def code_of(r):
        return (archive.dir / "programs" / f"{r['id']}.py").read_text()

    what = "design" if task == "rl" else "policy"
    parts = [f"## Parent {what} (improve this one)", f"```python\n{code_of(parent)}```",
             "### Its stage-1 feedback", parent["feedback"]]
    for k, r in enumerate(inspirations, 1):
        parts += [f"## Inspiration {k} (fitness {r['fitness']:.4f})", f"```python\n{code_of(r)}```"]
    recent_fail = [r for r in archive.rows[-20:] if not r.get("valid")][-3:]
    if recent_fail:
        parts.append("## Recent failed attempts (avoid these mistakes)")
        parts += [f"- {r['feedback'][:300]}" for r in recent_fail]
    b = archive.best()
    parts.append(f"## Search status\n{len(archive.rows)} candidates tried; best fitness so far "
                 f"{b['fitness']:.4f}." if b else "")
    parts.append(f"Write an improved {what}.")
    return "\n\n".join(p for p in parts if p)


def select(archive: Archive, rng: random.Random, k: int = 3, top: int = 10) -> tuple:
    pool = sorted(archive.valid(), key=lambda r: -r["fitness"])[:top]
    parent = max(rng.sample(pool, min(k, len(pool))), key=lambda r: r["fitness"])
    others = [r for r in pool if r["id"] != parent["id"]]
    return parent, rng.sample(others, min(2, len(others)))


def spent_usd(archive: Archive) -> float:
    return sum((r.get("proposer") or {}).get("cost_usd") or 0.0 for r in archive.rows)


def seed_programs(task: str) -> List[tuple]:
    if task == "rl":
        from evolve import rl_task

        return [(n, (REPO / "evolve" / "rl_designs" / f"{n}.py").read_text()) for n in rl_task.SEED_DESIGNS]
    return [(n, (REPO / "evolve" / "policies" / f"{n}.py").read_text()) for n in SEEDS]


def run(run_dir: Path, iterations: int, batch: int, workers: int, propose: Callable[[str], tuple],
        stage1=None, launcher=None, seed: int = 0, max_cost_usd: Optional[float] = None,
        task: str = "router") -> Archive:
    stage1 = stage1 or ev.STAGE1
    archive = Archive(run_dir)
    rng = random.Random(seed)
    seeded = {r["id"] for r in archive.rows}
    for name, code in seed_programs(task):  # seeds enter the archive as ordinary candidates
        pid = hashlib.sha256(code.encode()).hexdigest()[:12]
        if pid not in seeded:
            evaluate_candidate(code, archive, None, stage1, workers, launcher, {"seed": name}, task)
    for it in range(iterations):
        if max_cost_usd is not None and spent_usd(archive) >= max_cost_usd:
            print(f"[stop] proposer spend {spent_usd(archive):.2f} USD reached cap {max_cost_usd}", flush=True)
            break
        jobs = []
        for _ in range(batch):
            parent, insp = select(archive, rng)
            jobs.append((parent, build_prompt(parent, insp, archive, task)))

        def one(job):
            parent, prompt = job
            t0 = time.time()
            try:
                code, meta = propose(prompt)
            except Exception as e:  # API errors are recorded, never fatal to the run
                code, meta = None, {"error": f"{type(e).__name__}: {e}"}
            meta["proposal_s"] = round(time.time() - t0, 1)
            if not code:
                why = "output limit reached before the answer" if meta.get("stop_reason") in ("length", "max_tokens") \
                    else "no code returned"
                archive.add({"id": None, "parent": parent["id"], "time": time.time(), "valid": False,
                             "stage_reached": "proposal", "feedback": why, "proposer": meta})
                return None
            meta["iteration"] = it
            return evaluate_candidate(code, archive, parent["id"], stage1, workers, launcher, meta, task)

        with cf.ThreadPoolExecutor(max_workers=batch) as ex:
            list(ex.map(one, jobs))
        b = archive.best()
        print(f"[iter {it}] candidates={len(archive.rows)} best={b['fitness']:.4f} ({b['id']}) "
              f"spent={spent_usd(archive):.3f} USD", flush=True)
    return archive


def make_proposer(kind: str, model: Optional[str]):
    from evolve import proposer as P

    if kind == "openrouter":
        return P.OpenRouterProposer(model=model or "deepseek/deepseek-v4.1-flash")
    if kind == "claude-code":
        return P.ClaudeCodeProposer(model=model or "claude-opus-5")
    if kind == "anthropic":
        return P.ClaudeProposer(model=model or "claude-opus-5")
    raise ValueError(f"unknown proposer {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4, help="proposals evaluated in parallel per iteration")
    ap.add_argument("--workers", type=int, default=8, help="parallel simulations per candidate")
    ap.add_argument("--proposer", choices=["openrouter", "claude-code", "anthropic"], default="openrouter")
    ap.add_argument("--model", default=None, help="defaults: deepseek/deepseek-v4.1-flash | claude-opus-5")
    ap.add_argument("--scenarios", choices=["stage1", "smoke"], default="stage1")
    ap.add_argument("--max-cost-usd", type=float, default=None, help="stop once proposer spend reaches this")
    ap.add_argument("--srun", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["router", "rl"], default="router",
                    help="router: evolve routing code (EXP11); rl: evolve an RL router design (EXP12)")
    a = ap.parse_args()
    proposer = make_proposer(a.proposer, a.model)
    if a.task == "rl":
        from evolve import rl_task

        proposer.system_prompt = rl_task.SYSTEM_PROMPT
    launcher = ["srun", "--exact", "-n1", "-c1", "--mem=6G"] if a.srun else None
    stage1 = SMOKE if a.scenarios == "smoke" else ev.STAGE1
    run(Path(a.run_dir), a.iterations, a.batch, a.workers, proposer.propose, stage1=stage1,
        launcher=launcher, seed=a.seed, max_cost_usd=a.max_cost_usd, task=a.task)


if __name__ == "__main__":
    main()
