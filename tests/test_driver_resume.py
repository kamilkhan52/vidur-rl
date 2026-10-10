"""Driver budget and reproducibility across resumes (review D_r1 #1), run config guard, prompt adaptation (B_r1)."""
import hashlib
import sys

import pytest

from evolve import driver
from evolve import evaluate as ev

STAGE = [ev.Scenario("Tool", 4, 1.5, 12000, 150, 1)]


class Killed(BaseException):
    """Stands in for a job time limit: not an Exception, so nothing in the driver swallows it."""


def _stub(monkeypatch, kill_at=None):
    """Seeds and a deterministic stub evaluator (fitness from the program id); kill_at=(iteration, slot) raises once."""
    state = {"kill_at": kill_at}
    monkeypatch.setattr(driver, "seed_programs", lambda task, extra=None, rl_protocol="v2":
                        [("s1", "# seed one\n"), ("s2", "# seed two\n"), ("s3", "# seed three\n")])

    def evaluate_candidate(code, archive, parent, stage1, workers, launcher, meta=None, task="router",
                           objective="balanced", rl_protocol="v2"):
        meta = meta or {}
        if state["kill_at"] is not None and (meta.get("iteration"), meta.get("slot")) == state["kill_at"]:
            state["kill_at"] = None
            raise Killed()
        pid, _ = driver.save_program(archive, code)
        fit = int(pid, 16) % 10**6 / 10**6
        row = {"id": pid, "parent": parent, "valid": True, "fitness": fit, "feedback": f"fitness {fit:.6f}",
               "proposer": meta, "stage_reached": "1"}
        archive.add(row)
        return row

    monkeypatch.setattr(driver, "evaluate_candidate", evaluate_candidate)
    calls = []

    def propose(prompt):
        calls.append(prompt)
        return f"# child of prompt {hashlib.sha256(prompt.encode()).hexdigest()[:16]}\n", {}

    return propose, calls


def _proposals(archive):
    rows = [r for r in archive.rows if r.get("proposer", {}).get("iteration") is not None]
    return {(r["proposer"]["iteration"], r["proposer"]["slot"]): r["id"] for r in rows}, len(rows)


def _run(run_dir, n, propose, **kw):
    return driver.run(run_dir, iterations=n, batch=2, workers=1, propose=propose, stage1=STAGE, seed=7, **kw)


def test_iterations_are_a_total_and_resumes_reproduce_the_uninterrupted_run(tmp_path, monkeypatch):
    propose, calls = _stub(monkeypatch)
    ref, n_ref = _proposals(_run(tmp_path / "a", 5, propose))
    assert n_ref == 10 and len(ref) == 10
    _run(tmp_path / "b", 2, propose)
    _run(tmp_path / "b", 5, propose)
    got, n_got = _proposals(_run(tmp_path / "b", 5, propose))  # a third link adds nothing
    assert n_got == 10 and got == ref
    k = len(calls)
    _run(tmp_path / "b", 5, propose)
    assert len(calls) == k


def test_kill_mid_iteration_fills_only_the_missing_slot(tmp_path, monkeypatch):
    propose, _ = _stub(monkeypatch)
    ref, _ = _proposals(_run(tmp_path / "a", 5, propose))
    propose, _ = _stub(monkeypatch, kill_at=(3, 1))
    with pytest.raises(Killed):
        _run(tmp_path / "b", 5, propose)
    partial, _ = _proposals(driver.Archive(tmp_path / "b"))
    assert (3, 1) not in partial and max(i for i, _ in partial) == 3
    got, n_got = _proposals(_run(tmp_path / "b", 5, propose))
    assert n_got == 10 and got == ref


def test_resume_with_other_settings_is_refused(tmp_path, monkeypatch):
    propose, _ = _stub(monkeypatch)
    _run(tmp_path / "a", 1, propose)
    with pytest.raises(SystemExit, match="seed"):
        driver.run(tmp_path / "a", iterations=2, batch=2, workers=1, propose=propose, stage1=STAGE, seed=8)
    with pytest.raises(SystemExit, match="batch"):
        driver.run(tmp_path / "a", iterations=2, batch=3, workers=1, propose=propose, stage1=STAGE, seed=7)


def test_truncated_last_archive_line_is_dropped(tmp_path, monkeypatch):
    propose, _ = _stub(monkeypatch)
    _run(tmp_path / "a", 1, propose)
    path = tmp_path / "a" / "archive.jsonl"
    n = len(path.read_text().splitlines())
    with open(path, "a") as f:
        f.write('{"id": "trunc')
    assert len(driver.Archive(tmp_path / "a").rows) == n
    assert len(path.read_text().splitlines()) == n


@pytest.mark.parametrize("frac", ["0", "1.5", "nan", "-0.1"])
def test_rl_log_trace_frac_is_validated(monkeypatch, frac):
    monkeypatch.setattr(sys, "argv", ["driver", "--run-dir", "x", "--rl-log-trace-frac", frac])
    with pytest.raises(SystemExit):
        driver.main()


def test_prompt_describes_the_bar_and_search_set_used():
    from evolve.proposer import system_prompt_v2
    s = system_prompt_v2("tail4")
    assert driver.adapt_prompt(s, ev.BAR_V2, ev.SEARCH_V2) == s  # earlier v2 runs' prompt unchanged
    t = driver.adapt_prompt(s, ev.BAR_V4, ev.STAGES["search3b"])
    assert "AND two\nproduction-style" not in t and "power-of-two choices on outstanding" in t
    assert "join-the-shortest-queue" in t and "synthetic trace" in t and "Synth: 1" in t
    tuned = ev.BAR_V4 + tuple(f"policy:{ev.TUNED / f}" for f in ("dynamo_kv_cost_tuned_best.py",
                                                                   "mooncake_min_ttft_tuned_best.py"))
    assert "tuned on this search set" in driver.adapt_prompt(s, tuned, ev.STAGES["search3b"])
    with pytest.raises(ValueError):
        driver.adapt_prompt("unrelated prompt", ev.BAR_V4, ev.STAGES["search3b"])
