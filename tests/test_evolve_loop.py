"""The research loop: static guard, and the driver end to end with a scripted proposer."""

from pathlib import Path

import pytest

import evolve.evaluate as ev
from evolve import driver
from evolve.guard import check_policy_source
from evolve.proposer import extract_code

POLICIES = Path(__file__).resolve().parents[1] / "evolve" / "policies"
GOOD = (
    "import math\n\n"
    "def choose(request, replicas, ctx):\n"
    "    # prefer cached prefixes, penalise queued prefill work\n"
    "    def cost(r):\n"
    "        return r.pending_prefill_tokens + request.num_prefill_tokens - request.cached_prefix_tokens[r.index]\n"
    "    best = min(cost(r) for r in replicas)\n"
    "    return ctx.rng.choice([r.index for r in replicas if cost(r) == best])\n"
)


@pytest.mark.parametrize("name", ["lor", "sticky_lor", "lop_uncached", "round_robin"])
def test_seed_policies_pass_guard(name):
    assert check_policy_source((POLICIES / f"{name}.py").read_text()) == []


@pytest.mark.parametrize(
    "src,needle",
    [("import os\ndef choose(a, b, c):\n    return 0\n", "import of 'os'"),
     ("import random\ndef choose(a, b, c):\n    return random.randint(0, 1)\n", "import of 'random'"),
     ("def choose(a, b, c):\n    return open('x') and 0\n", "'open'"),
     ("def choose(a, b, c):\n    return a.__class__ and 0\n", "dunder"),
     ("def choose(a, b, c):\n    global X\n    return 0\n", "global"),
     ("def choose(a, b):\n    return 0\n", "3 positional"),
     ("def helper():\n    return 0\n", "must define"),
     ("def choose(a, b, c) return 0\n", "syntax error")],
)
def test_guard_rejects(src, needle):
    assert any(needle in p for p in check_policy_source(src))


def test_extract_code_takes_last_block():
    text = "Hypothesis.\n```python\nx = 1\n```\nRevised:\n```python\ndef choose(a, b, c):\n    return 0\n```"
    assert extract_code(text).startswith("def choose")
    assert extract_code("no code here") is None


def test_driver_end_to_end_with_scripted_proposer(tmp_path, monkeypatch):
    for t in ev.TRACES.values():
        if not t.exists():
            pytest.skip("traces not available")
    monkeypatch.setattr(ev, "CACHE_DIR", tmp_path / "cache")
    stage1 = [ev.Scenario("Tool", 4, 1.5, 12000, 150, 1)]
    answers = iter([
        (GOOD, {"note": "good"}),
        ("import os\ndef choose(a, b, c):\n    return 0\n", {"note": "guard"}),
        ("def choose(request, replicas, ctx):\n    return len(replicas)\n", {"note": "crash"}),
        (None, {"note": "empty"}),
    ])
    prompts = []

    def fake_propose(prompt):
        prompts.append(prompt)
        return next(answers)

    archive = driver.run(tmp_path / "run", iterations=1, batch=4, workers=4,
                         propose=fake_propose, stage1=stage1)
    rows = archive.rows
    assert len(rows) == 4 + 4  # four seeds + four proposals
    assert sum(1 for r in rows if r.get("proposer", {}).get("seed")) == 4
    by_note = {r["proposer"].get("note"): r for r in rows if "note" in r.get("proposer", {})}
    assert by_note["good"]["valid"] and by_note["good"]["stage_reached"] == "1"
    assert by_note["guard"]["stage_reached"] == "0a" and "static guard" in by_note["guard"]["feedback"]
    assert by_note["crash"]["stage_reached"] == "0b" and not by_note["crash"]["valid"]
    assert by_note["empty"]["stage_reached"] == "proposal"
    # lineage: every proposal names a seed parent; prompts carry parent code and feedback
    assert all(r["parent"] for r in by_note.values())
    assert all("## Parent policy" in p and "fitness" in p for p in prompts)
    # archive is append-only JSONL and reloads identically
    assert driver.Archive(tmp_path / "run").rows == rows
