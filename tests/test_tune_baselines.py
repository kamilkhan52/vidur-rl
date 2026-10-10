import random
import re
from pathlib import Path

import numpy as np

from evolve.analysis.tune_baselines import FAMILIES, ask_tell, render, to_unit, to_value
from vidur.scheduler.global_scheduler.evolved_global_scheduler import (
    PolicyContext, ReplicaView, RequestView, load_policy)

SYSTEMS = Path(__file__).resolve().parents[1] / "evolve" / "policies" / "systems"


def _random_case(rng, n=4):
    reps = [ReplicaView(i, rng.randint(0, 9), rng.randint(0, 5000), 0, 0, rng.random(), 100000) for i in range(n)]
    hashes = tuple(range(rng.randint(1, 6)))
    req = RequestView(0.0, 16 * len(hashes) + rng.randint(0, 15), None,
                      tuple(rng.choice([0, 16, 32]) for _ in range(n)), -1, 0, 1, hashes, 16,
                      tuple(rng.choice([0, 16]) for _ in range(n)), tuple(rng.choice([0, 16, 48]) for _ in range(n)))
    return req, reps


class _Rng(random.Random):
    """Counts tie-breaks (the policies call rng.choice only to break a tie)."""

    def __init__(self, seed):
        super().__init__(seed)
        self.ties = 0

    def choice(self, seq):
        self.ties += 1
        return super().choice(seq)


def _burst_cases(rng, n=8, bursts=1500, size=4):
    """Bursts of ``size`` requests at one ``now`` (same-burst prefix index in use), quantised loads and cache hits so
    exact ties are frequent, and requests drawn from a few shared prefix families so same-burst matches occur."""
    families = [tuple(range(100 * f, 100 * f + 6)) for f in range(3)]
    for b in range(bursts):
        reps = [ReplicaView(i, rng.randint(0, 3), 16 * rng.randint(0, 2), 0, 0, rng.choice([0.0, 0.25]), 128)
                for i in range(n)]
        for k in range(size):
            hashes = rng.choice(families)[:rng.randint(1, 6)]
            p = 16 * len(hashes) + rng.choice([0, 0, 8])
            hit = lambda: tuple(min(p, rng.choice([0, 0, 16, 32])) for _ in range(n))  # noqa: E731
            yield float(b), RequestView(float(b), p, None, hit(), -1, k, size, hashes, 16, hit(), hit()), reps


def test_defaults_reproduce_the_untuned_policies(tmp_path):
    for fam, base in (("dynamo", "dynamo_kv_cost.py"), ("mooncake", "mooncake_min_ttft.py")):
        template, space = FAMILIES[fam]
        p = tmp_path / f"{fam}.py"
        p.write_text(render(template.read_text(), {k: v[3] for k, v in space.items()}))
        tuned, ref = load_policy(str(p)), load_policy(str(SYSTEMS / base))
        rng = random.Random(1)
        for seed in range(200):  # independent single decisions
            req, reps = _random_case(rng)
            c1, c2 = PolicyContext(random.Random(seed), 4), PolicyContext(random.Random(seed), 4)
            assert tuned.choose(req, reps, c1) == ref.choose(req, reps, c2)
        # one shared context per policy across bursts: exercises the tie-break and the same-burst prefix index
        match, burst_hits = ref._match, [0]

        def counting_match(index, hashes):
            n = match(index, hashes)
            burst_hits[0] += n > 0
            return n

        ref._match = counting_match
        c1, c2 = PolicyContext(_Rng(0), 8), PolicyContext(_Rng(0), 8)
        n = 0
        for now, req, reps in _burst_cases(random.Random(2)):
            c1.now = c2.now = now
            assert tuned.choose(req, reps, c1) == ref.choose(req, reps, c2), (fam, n)
            n += 1
        assert n == 6000 and c1.rng.ties == c2.rng.ties
        assert c2.rng.ties > 300 and burst_hits[0] > 1000, (fam, c2.rng.ties, burst_hits[0])


def test_unit_mapping_round_trips():
    for lo, hi, log in ((0.05, 20.0, True), (0.0, 1.0, False)):
        for u in (0.0, 0.3, 1.0):
            assert abs(to_unit(to_value(u, lo, hi, log), lo, hi, log) - u) < 1e-12


def test_es_finds_the_optimum_of_a_quadratic():
    target = np.array([0.2, 0.7])
    hist = ask_tell(120, 2, lambda pop: [-float(((u - target) ** 2).sum()) for u in pop], seed=0)
    best = max(hist, key=lambda x: x[1])[0]
    assert len(hist) == 120 and np.abs(best - target).max() < 0.05


def test_install_writes_the_frozen_policy(tmp_path, monkeypatch):
    import json
    from evolve import evaluate as ev
    from evolve.analysis import tune_baselines as tb
    monkeypatch.setattr(ev, "TUNED", tmp_path / "tuned")
    run = tmp_path / "run"
    run.mkdir()
    params = {"PREFILL_SCALE": 2.5, "TEMPERATURE": 0.0}
    rec = {"params": params, "fitness": 0.01, "valid": True, "file": "x", "code_version": ev.code_version()}
    (run / "best.json").write_text(json.dumps(rec))
    (run / "history.jsonl").write_text(json.dumps(rec) + "\n")
    out = tb.install("dynamo", run)
    assert out.name == "dynamo_kv_cost_tuned_best.py"
    assert "PREFILL_SCALE = 2.5" in out.read_text()
    load_policy(str(out))


def test_history_keeps_every_record_and_repairs_the_file(tmp_path):
    import json
    from evolve.analysis.tune_baselines import load_history
    log = tmp_path / "history.jsonl"
    rec = lambda a, valid: {"params": {"A": a}, "fitness": 0.1 if valid else -1e9, "valid": valid}
    rows = [rec(0.0, True), rec(1.0, False), rec(2.0, True)]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"params": {"A": 5')
    assert {json.loads(k)["A"] for k in load_history(log)} == {0.0, 1.0, 2.0}  # failures are kept, not retried
    assert log.read_text() == "".join(json.dumps(r) + "\n" for r in rows)  # truncated line removed
    log.write_text(log.read_text().rstrip("\n"))  # complete last record without its newline
    load_history(log)
    with log.open("a") as f:
        f.write(json.dumps(rec(3.0, True)) + "\n")
    assert len(load_history(log)) == 4


def test_resume_replays_without_new_evaluations(tmp_path, monkeypatch):
    import sys
    from evolve import evaluate as ev
    from evolve.analysis import tune_baselines as tb
    calls = []

    def fake_eval(path, scenarios, workers, baselines, objective):
        src = open(path).read()
        calls.append(path)
        x = float(re.search(r"^PREFILL_SCALE = (.*)$", src, re.M).group(1))
        return {"combined_score": -abs(x - 2.0), "valid": True, "code_version": ev.code_version()}

    monkeypatch.setattr(ev, "evaluate_policy", fake_eval)
    monkeypatch.setattr(ev, "ensure_baselines", lambda *a, **k: None)
    argv = ["tb", "--family", "dynamo", "--budget", "13", "--parallel", "2", "--run-dir", str(tmp_path / "r")]
    monkeypatch.setattr(sys, "argv", argv)
    tb.main()
    first = (tmp_path / "r" / "best.json").read_text()
    n = len(calls)
    assert n == 13 and len(set(calls)) == 13
    tb.main()
    assert len(calls) == n and (tmp_path / "r" / "best.json").read_text() == first
    monkeypatch.setattr(sys, "argv", argv[:-2] + ["--seed", "1"] + argv[-2:])
    try:
        tb.main()
        raise AssertionError("resume with another seed must be refused")
    except SystemExit:
        pass


def test_resume_after_a_mid_run_failure_and_a_kill_follows_the_original_trajectory(tmp_path, monkeypatch):
    import json
    import sys
    from evolve import evaluate as ev
    from evolve.analysis import tune_baselines as tb
    seen = set()

    def fake_eval(path, scenarios, workers, baselines, objective):
        x = float(re.search(r"^PREFILL_SCALE = (.*)$", open(path).read(), re.M).group(1))
        first = x not in seen
        seen.add(x)
        if first and len(seen) == 21:  # a transient failure (would succeed on retry), consumed by the ES
            return {"combined_score": -1e9, "valid": False, "failures": {"s": {"reason": "timeout"}}}
        return {"combined_score": -abs(x - 2.0), "valid": True, "code_version": ev.code_version()}

    monkeypatch.setattr(ev, "evaluate_policy", fake_eval)
    monkeypatch.setattr(ev, "ensure_baselines", lambda *a, **k: None)

    def run(d):
        monkeypatch.setattr(sys, "argv", ["tb", "--family", "dynamo", "--budget", "25", "--run-dir", str(d)])
        tb.main()
        return [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]

    full = run(tmp_path / "full")
    for cut in (3, 12, 22, 25):  # killed early, mid-run, just after the failure, after finishing
        seen.clear()
        seen.update(float(r["params"]["PREFILL_SCALE"]) for r in full[:cut])  # recorded: a retry would succeed
        part = tmp_path / f"part{cut}"
        part.mkdir()
        (part / "meta.json").write_text((tmp_path / "full" / "meta.json").read_text())
        lines = (tmp_path / "full" / "history.jsonl").read_text().splitlines()[:cut]
        (part / "history.jsonl").write_text("".join(l + "\n" for l in lines) + lines[-1][:20])  # mid-write kill
        resumed = run(part)
        key = lambda rows: sorted((json.dumps(r["params"], sort_keys=True), r["valid"]) for r in rows)
        assert key(resumed) == key(full)
        pick = lambda d: {k: v for k, v in json.loads((d / "best.json").read_text()).items() if k != "file"}
        assert pick(part) == pick(tmp_path / "full")


def test_a_history_from_another_trajectory_is_refused_within_a_generation(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    from evolve import evaluate as ev
    from evolve.analysis import tune_baselines as tb
    calls = []

    def fake_eval(path, scenarios, workers, baselines, objective):
        calls.append(path)
        x = float(re.search(r"^PREFILL_SCALE = (.*)$", open(path).read(), re.M).group(1))
        return {"combined_score": -abs(x - 2.0), "valid": True, "code_version": ev.code_version()}

    monkeypatch.setattr(ev, "evaluate_policy", fake_eval)
    monkeypatch.setattr(ev, "ensure_baselines", lambda *a, **k: None)
    for seed, d in ((0, "a"), (1, "b")):
        monkeypatch.setattr(sys, "argv", ["tb", "--family", "dynamo", "--budget", "25", "--seed", str(seed),
                                          "--run-dir", str(tmp_path / d)])
        tb.main()
    # seed-1 history under seed-0 settings (as if the history had been copied into the wrong run dir)
    (tmp_path / "a" / "history.jsonl").write_text((tmp_path / "b" / "history.jsonl").read_text())
    (tmp_path / "a" / "best.json").unlink()
    n = len(calls)
    monkeypatch.setattr(sys, "argv", ["tb", "--family", "dynamo", "--budget", "25", "--run-dir", str(tmp_path / "a")])
    with pytest.raises(SystemExit, match="not on the replayed ES trajectory"):
        tb.main()
    assert len(calls) - n <= tb.lam_of(2)  # caught within the first generation that needed new evaluations


def test_meta_mismatch_names_the_differing_keys(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    from evolve.analysis import tune_baselines as tb
    run = tmp_path / "r"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"family": "dynamo", "seed": 7}))
    monkeypatch.setattr(sys, "argv", ["tb", "--family", "dynamo", "--budget", "5", "--run-dir", str(run)])
    with pytest.raises(SystemExit, match=r"'seed': \(7, 0\).*start a new --run-dir"):
        tb.main()
