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

Resuming (same --run-dir): --iterations is the TOTAL budget of the run, iterations x batch proposals, counted from
the archive (each proposal row records its iteration and slot), so a chain of time-limited jobs stops at the budget.
Parent selection in iteration i uses an RNG seeded by (seed, i) and sees only the seeds and the rows of iterations
< i, so a resumed run makes the same selections as an uninterrupted one. <run_dir>/run.json records the settings
that define the run (task, stage, bar, objective, seed, batch, code_version); resuming with other values is refused.

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
# Router search protocol: v1 = EXP11 (STAGE1, five heuristics); v2 = SEARCH_V2 against BAR_V2 (set in main()).
BASELINES = ev.DEFAULT_BASELINES
NO_DEFAULT_SEEDS = False
RL_TRAIN_KW: dict = {}  # --rl-log / --rl-train-seed: RL_LOGGING_SPEC options passed to rl_task.train
STAGE0 = [ev.Scenario("Tool", 4, 1.5, 12000, 150, 1)]
# Plumbing check only (seconds per simulation); not a research scenario set.
SMOKE = [ev.Scenario("Tool", 4, 1.5, 12000, 200, 1), ev.Scenario("Conv", 4, 2.0, 6000, 200, 1)]


class Archive:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        (run_dir / "programs").mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "archive.jsonl"
        self.lock = threading.Lock()
        lines = self.path.read_text().splitlines() if self.path.exists() else []
        try:
            self.rows = [json.loads(l) for l in lines]
        except json.JSONDecodeError:  # a job killed mid-write: drop the truncated last line, refuse anything else
            self.rows = [json.loads(l) for l in lines[:-1]]
            print(f"[resume] dropped a truncated last line of {self.path}", flush=True)
            self.path.write_text("".join(json.dumps(r) + "\n" for r in self.rows))

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
                       meta: Optional[dict] = None, task: str = "router", objective: str = "balanced",
                       rl_protocol: str = "v2") -> dict:
    pid, path = save_program(archive, code)
    row = {"id": pid, "parent": parent, "time": time.time(), "valid": False, "stage_reached": "0a",
           "proposer": meta or {}, "task": task, "objective": objective,
           **({"rl_protocol": rl_protocol} if task == "rl" else {})}
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

        smoke = stage1 == SMOKE
        res = rl_task.evaluate_design(str(path), archive.dir / "training" / pid,
                                      stage1 if (smoke or rl_protocol == "v2") else None, workers=workers,
                                      launcher=launcher, episodes=rl_task.SMOKE_EPISODES if smoke else None,
                                      protocol=rl_protocol, objective=objective, train_kw=RL_TRAIN_KW or None)
        feedback = rl_task.feedback_text(res)
    else:
        res = ev.evaluate_policy(str(path), stage1, workers=workers, launcher=launcher, objective=objective,
                                 baselines=BASELINES)
        feedback = ev.feedback_text(res)
    row.update({"valid": res.get("valid", False), "fitness": res["combined_score"],
                "feedback": feedback, "result": res})
    archive.add(row)
    return row


def build_prompt(parent: dict, inspirations: List[dict], archive: Archive, task: str = "router",
                 rows: Optional[List[dict]] = None) -> str:
    """rows: the archive as the iteration sees it (default: all rows)."""
    rows = archive.rows if rows is None else rows

    def code_of(r):
        return (archive.dir / "programs" / f"{r['id']}.py").read_text()

    what = "design" if task == "rl" else "policy"
    parts = [f"## Parent {what} (improve this one)", f"```python\n{code_of(parent)}```",
             "### Its stage-1 feedback", parent["feedback"]]
    for k, r in enumerate(inspirations, 1):
        parts += [f"## Inspiration {k} (fitness {r['fitness']:.4f})", f"```python\n{code_of(r)}```"]
    recent_fail = [r for r in rows[-20:] if not r.get("valid")][-3:]
    if recent_fail:
        parts.append("## Recent failed attempts (avoid these mistakes)")
        parts += [f"- {r['feedback'][:300]}" for r in recent_fail]
    valid = [r for r in rows if r.get("valid")]
    b = max(valid, key=lambda r: r["fitness"]) if valid else None
    parts.append(f"## Search status\n{len(rows)} candidates tried; best fitness so far "
                 f"{b['fitness']:.4f}." if b else "")
    parts.append(f"Write an improved {what}.")
    return "\n\n".join(p for p in parts if p)


def select(rows, rng: random.Random, k: int = 3, top: int = 10) -> tuple:
    """Tournament parent and two inspirations from the valid rows (an Archive or a list of rows)."""
    rows = rows.rows if isinstance(rows, Archive) else rows
    pool = sorted((r for r in rows if r.get("valid")), key=lambda r: -r["fitness"])[:top]
    parent = max(rng.sample(pool, min(k, len(pool))), key=lambda r: r["fitness"])
    others = [r for r in pool if r["id"] != parent["id"]]
    return parent, rng.sample(others, min(2, len(others)))


def _iteration(row: dict) -> Optional[int]:
    return (row.get("proposer") or {}).get("iteration")


def iteration_rng(seed: int, it: int) -> random.Random:
    """Selection RNG of iteration ``it``: a function of (seed, it) only, so resumed runs select like uninterrupted ones."""
    return random.Random(f"evolve-driver:{seed}:{it}")


def missing_slots(archive: Archive, it: int, batch: int) -> List[int]:
    """Proposal slots of iteration ``it`` not yet in the archive (a proposal with no code fills its slot too)."""
    rows = [r for r in archive.rows if _iteration(r) == it]
    done = {(r.get("proposer") or {}).get("slot") for r in rows}
    return [j for j in range(batch) if j not in done][:max(0, batch - len(rows))]


def _member(b: str) -> str:
    if b.startswith("policy:"):
        path = Path(b[len("policy:"):])
        return f"policy:{path.stem}:{ev.file_sha(path.resolve())[:12]}"
    return b


def check_run_config(run_dir: Path, config: dict) -> None:
    """Write <run_dir>/run.json on the first start; on resume refuse settings that differ from it."""
    path = run_dir / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{run_dir} was started with other settings (recorded, now): {diff}. Resume with the "
                             f"recorded settings, or start a new --run-dir.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1))


def spent_usd(archive: Archive) -> float:
    return sum((r.get("proposer") or {}).get("cost_usd") or 0.0 for r in archive.rows)


def seed_programs(task: str, extra: Optional[List[str]] = None, rl_protocol: str = "v2") -> List[tuple]:
    extra_rows = [(Path(p).stem, Path(p).read_text()) for p in (extra or [])]
    if task == "rl":
        from evolve import rl_task

        names = rl_task.PROTOCOLS[rl_protocol]["seeds"]
        return [(n, (REPO / "evolve" / "rl_designs" / f"{n}.py").read_text()) for n in names] + extra_rows
    base = [] if NO_DEFAULT_SEEDS else [(n, (REPO / "evolve" / "policies" / f"{n}.py").read_text()) for n in SEEDS]
    return base + extra_rows


def run(run_dir: Path, iterations: int, batch: int, workers: int, propose: Callable[[str], tuple],
        stage1=None, launcher=None, seed: int = 0, max_cost_usd: Optional[float] = None,
        task: str = "router", objective: str = "balanced", extra_seeds: Optional[List[str]] = None,
        rl_protocol: str = "v2") -> Archive:
    stage1 = stage1 or ev.STAGE1
    check_run_config(run_dir, {
        "task": task, "objective": objective, "seed": seed, "batch": batch,
        "rl_protocol": rl_protocol if task == "rl" else None, "stage": [sc.tag for sc in stage1],
        "bar": [_member(b) for b in BASELINES] if task == "router" else None, "code_version": ev.code_version()})
    archive = Archive(run_dir)
    seeded = {r["id"] for r in archive.rows}
    todo = [(name, code) for name, code in seed_programs(task, extra_seeds, rl_protocol)  # seeds enter the archive as candidates
            if hashlib.sha256(code.encode()).hexdigest()[:12] not in seeded]
    with cf.ThreadPoolExecutor(max_workers=max(1, batch)) as ex:  # in parallel: RL seeds take ~1 h each
        list(ex.map(lambda nc: evaluate_candidate(nc[1], archive, None, stage1, workers, launcher,
                                                  {"seed": nc[0]}, task, objective, rl_protocol), todo))
    for it in range(iterations):  # a total: iterations already in the archive are skipped
        slots = missing_slots(archive, it, batch)
        if not slots:
            continue
        if max_cost_usd is not None and spent_usd(archive) >= max_cost_usd:
            print(f"[stop] proposer spend {spent_usd(archive):.2f} USD reached cap {max_cost_usd}", flush=True)
            break
        rng = iteration_rng(seed, it)
        view = [r for r in archive.rows if _iteration(r) is None or _iteration(r) < it]  # as at the iteration's start
        jobs = []
        for j in range(batch):  # draw every slot's selection, in order, so the RNG stream does not depend on resumes
            parent, insp = select(view, rng)
            if j in slots:
                jobs.append((j, parent, build_prompt(parent, insp, archive, task, view)))

        def one(job):
            slot, parent, prompt = job
            t0 = time.time()
            try:
                code, meta = propose(prompt)
            except Exception as e:  # API errors are recorded, never fatal to the run
                code, meta = None, {"error": f"{type(e).__name__}: {e}"}
            meta["proposal_s"] = round(time.time() - t0, 1)
            meta.update(iteration=it, slot=slot)
            if not code:
                why = "output limit reached before the answer" if meta.get("stop_reason") in ("length", "max_tokens") \
                    else "no code returned"
                archive.add({"id": None, "parent": parent["id"], "time": time.time(), "valid": False,
                             "stage_reached": "proposal", "feedback": why, "proposer": meta})
                return None
            return evaluate_candidate(code, archive, parent["id"], stage1, workers, launcher, meta, task,
                                      objective, rl_protocol)

        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            list(ex.map(one, jobs))
        b = archive.best()
        print(f"[iter {it}] candidates={len(archive.rows)} best={b['fitness']:.4f} ({b['id']}) "
              f"spent={spent_usd(archive):.3f} USD", flush=True)
    return archive


_HEURISTICS = ("sticky_lor", "lor", "round_robin", "random", "lop_uncached")
_MEMBER_TEXT = {
    "sticky_lor": "sticky-LOR", "lor": "LOR", "round_robin": "round-robin", "random": "random",
    "lop_uncached": "least-uncached-prefill",
    "mooncake_min_ttft": "a Mooncake-style min-estimated-TTFT router (queued prefill + this request's uncached tokens, "
                         "counting prefixes already routed but not yet prefilled as cached)",
    "dynamo_kv_cost": "a Dynamo-style KV-cost router (prefill cost + decode-block load, crediting overlap with active "
                      "sequences)",
    "p2c_outstanding": "power-of-two choices on outstanding requests (two random replicas; the less loaded wins)",
    "p2c_kv": "power-of-two choices on KV load (KV blocks in use + uncached prompt tokens routed there and not yet "
              "prefilled)",
    "jsq_kv": "join-the-shortest-queue on that KV load (prefix-cache blind)",
    "mooncake_min_ttft_tuned_best": "the Mooncake-style router with its cost constants tuned on this search set by an "
                                    "equal-budget evolution strategy",
    "dynamo_kv_cost_tuned_best": "the Dynamo-style router with its cost constants tuned on this search set by an "
                                 "equal-budget evolution strategy",
}
_BAR_V2_TEXT = ("a strong bar: the best (per scenario and per metric) of five\nheuristic routers (sticky-LOR, LOR, "
                "round-robin, random, least-uncached-prefill) AND two\nproduction-style routers: a Mooncake-style "
                "min-estimated-TTFT router (queued prefill + this\nrequest's uncached tokens, counting prefixes already "
                "routed but not yet prefilled as cached) and\na Dynamo-style KV-cost router (prefill cost + decode-block "
                "load, crediting overlap with active\nsequences).")
_DRAWS_V2_TEXT = ("Each load point is scored on two independent arrival draws, so gains that\ndepend on one lucky draw "
                  "do not count.")
_WORKLOAD_V2_TEXT = "Mooncake production traces (conversation and tool/agent)"


def describe_bar(baselines) -> str:
    names = [b.rsplit("/", 1)[-1].removesuffix(".py") if b.startswith("policy:") else b for b in baselines]
    parts = []
    if all(h in names for h in _HEURISTICS):
        parts.append("five heuristic routers (" + ", ".join(_MEMBER_TEXT[h] for h in _HEURISTICS) + ")")
        names = [n for n in names if n not in _HEURISTICS]
    parts += [_MEMBER_TEXT.get(n, n) for n in names]
    return (f"a strong bar: the best (per scenario and per metric) of these {len(baselines)} routers: "
            + "; ".join(parts) + ".")


def adapt_prompt(prompt: str, baselines, stage) -> str:
    """Make the protocol-v2 system prompt describe the bar and the search set actually used. Unchanged for the v2 bar
    on SEARCH_V2 (so earlier runs' prompts are reproduced). Raises if the prompt lacks the text it rewrites."""
    for anchor in (_BAR_V2_TEXT, _DRAWS_V2_TEXT, _WORKLOAD_V2_TEXT):
        if anchor not in prompt:
            raise ValueError(f"system prompt lacks the text to adapt: {anchor[:60]!r}")
    if tuple(baselines) != tuple(ev.BAR_V2):
        prompt = prompt.replace(_BAR_V2_TEXT, describe_bar(baselines))
    traces = sorted({sc.trace for sc in stage}, key=["Conv", "Tool", "Synth"].index)
    if "Synth" in traces:
        prompt = prompt.replace(_WORKLOAD_V2_TEXT, "Mooncake production traces (conversation and tool/agent) and "
                                                   "Mooncake's synthetic trace")
    draws = {}
    for sc in stage:
        draws.setdefault((sc.trace, sc.replicas, sc.scale, sc.model, sc.device), set()).add(sc.perturbation)
    by_trace = {t: sorted({len(v) for k, v in draws.items() if k[0] == t}) for t in traces}
    if any(n != [2] for n in by_trace.values()):
        desc = ", ".join(f"{t}: {'/'.join(map(str, n))}" for t, n in by_trace.items())
        prompt = prompt.replace(_DRAWS_V2_TEXT, f"Each load point is scored on independent arrival draws ({desc} per "
                                                f"load point), so gains\nthat depend on one lucky draw count less.")
    return prompt


def make_proposer(kind: str, model: Optional[str], relay_dir: Optional[str] = None):
    from evolve import proposer as P

    if kind == "relay":  # proposals answered elsewhere (e.g. Claude subscription on a workstation)
        from evolve.relay import RelayProposer

        if not relay_dir:
            raise ValueError("--proposer relay needs --relay-dir")
        return RelayProposer(relay_dir, model=model)

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
    ap.add_argument("--iterations", type=int, default=10,
                    help="TOTAL iterations of the run (a resumed run continues to this count, not past it)")
    ap.add_argument("--batch", type=int, default=4, help="proposals evaluated in parallel per iteration")
    ap.add_argument("--workers", type=int, default=8, help="parallel simulations per candidate")
    ap.add_argument("--proposer", choices=["openrouter", "claude-code", "anthropic", "relay"], default="openrouter")
    ap.add_argument("--relay-dir", default=None, help="with --proposer relay: shared request/response directory")
    ap.add_argument("--model", default=None, help="defaults: deepseek/deepseek-v4.1-flash | claude-opus-5")
    ap.add_argument("--scenarios", choices=["stage1", "smoke"], default="stage1")
    ap.add_argument("--max-cost-usd", type=float, default=None, help="stop once proposer spend reaches this")
    ap.add_argument("--srun", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1",
                    help="router task: v1 = EXP11 search set and heuristic bar; v2 = two draws per scenario, "
                         "bar = heuristics + Mooncake + Dynamo, seeds only from --seed-policy")
    ap.add_argument("--rl-protocol", default="v2", help="with --task rl: v2 | loto_conv (see rl_task.PROTOCOLS)")
    ap.add_argument("--rl-log", action="store_true",
                    help="with --task rl: learner logs per training episode (RL_LOGGING_SPEC: manifest.json, "
                         "<ckpt>.updates.npz, <ckpt>.trace.npz); behaviour unchanged")
    ap.add_argument("--rl-log-trace-frac", type=float, default=1.0,
                    help="with --rl-log: fraction of transitions kept in the per-episode traces")
    ap.add_argument("--rl-train-seed", type=int, default=None,
                    help="with --task rl: training seed separate from the scenario seeds (default: scenario seed)")
    ap.add_argument("--seed-policy", action="append", default=[],
                    help="extra seed program file (repeatable), e.g. an earlier finalist")
    ap.add_argument("--objective", choices=sorted(ev.OBJECTIVES), default="balanced",
                    help="tail objective (weights on P99 E2E/TTFT/TPOT); see evaluate.OBJECTIVES")
    ap.add_argument("--bar", default=None, choices=sorted(ev.BARS),
                    help="with --protocol v2: the bar fitness is computed against (default v2, kept so earlier runs "
                         "reproduce; the v4 protocol uses v4t = v4 + tuned Mooncake/Dynamo)")
    ap.add_argument("--search-stage", default=None, choices=sorted(ev.STAGES),
                    help="router v2: search on this evaluate.STAGES set instead of SEARCH_V2 (e.g. cal70)")
    ap.add_argument("--task", choices=["router", "rl"], default="router",
                    help="router: evolve routing code (EXP11); rl: evolve an RL router design (EXP12)")
    a = ap.parse_args()
    if not 0 < a.rl_log_trace_frac <= 1:  # also rejects nan
        ap.error(f"--rl-log-trace-frac must be in (0, 1], got {a.rl_log_trace_frac}")
    if a.bar is None:
        a.bar = "v2"
        if a.task == "router" and a.protocol == "v2":
            print("WARNING: --bar not given; scoring against the v2 bar (5 heuristics + Mooncake + Dynamo). The v4 "
                  "protocol (evolve/SELECTION.md) uses --bar v4t.", flush=True)
    proposer = make_proposer(a.proposer, a.model, a.relay_dir)
    from evolve.proposer import system_prompt_for

    proposer.system_prompt = system_prompt_for(a.objective)
    if a.task == "rl":
        from evolve import rl_task

        proposer.system_prompt = rl_task.SYSTEM_PROMPT if a.rl_protocol == "v2" else rl_task.system_prompt_v3(a.objective)
    launcher = ["srun", "--exact", "-n1", "-c1", "--mem=6G"] if a.srun else None
    if a.rl_log:
        RL_TRAIN_KW.update(log=True, log_trace_frac=a.rl_log_trace_frac)
    if a.rl_train_seed is not None:
        RL_TRAIN_KW["train_seed"] = a.rl_train_seed
    stage1 = SMOKE if a.scenarios == "smoke" else ev.STAGE1
    if a.task == "router" and a.protocol == "v2":
        global BASELINES, NO_DEFAULT_SEEDS
        BASELINES = ev.BARS[a.bar]
        NO_DEFAULT_SEEDS = True
        if a.scenarios != "smoke":
            stage1 = ev.SEARCH_V2
        from evolve.proposer import system_prompt_v2

        proposer.system_prompt = system_prompt_v2(a.objective)
        if a.search_stage:
            stage1 = ev.STAGES[a.search_stage]
            models = {sc.model for sc in stage1}  # describe the deployment the candidates are scored on
            if models == {"70b"}:
                proposer.system_prompt = proposer.system_prompt.replace(
                    "Llama-2-7b replicas (A100,", "Llama-2-70b replicas (4x A100 tensor parallel, grouped-query attention,"
                ).replace("122,752 KV-cache tokens", "525,920 KV-cache tokens")
            elif models == {"l3_8b"}:
                proposer.system_prompt = proposer.system_prompt.replace(
                    "Llama-2-7b replicas (A100,", "Llama-3.1-8B replicas (A100, Meta-Llama-3-8B performance profile, "
                    "128K context, grouped-query attention,"
                ).replace("122,752 KV-cache tokens", "483,328 KV-cache tokens")
        proposer.system_prompt = adapt_prompt(proposer.system_prompt, BASELINES, stage1)
    run(Path(a.run_dir), a.iterations, a.batch, a.workers, proposer.propose, stage1=stage1,
        launcher=launcher, seed=a.seed, max_cost_usd=a.max_cost_usd, task=a.task, objective=a.objective,
        extra_seeds=a.seed_policy, rl_protocol=a.rl_protocol)


if __name__ == "__main__":
    main()
