"""R0 label-reliability probe: zero jitter reproduces the logged rollouts; jitter changes them; analysis runs."""
import json

import pytest

from evolve import evaluate as ev
from evolve.rl_plan import c0
from evolve.rl_plan import headroom as hr
from evolve.rl_plan import r0_probe as r0
from evolve.sim_fork import build_simulator

SC = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1, model="l3_8b")


@pytest.fixture(scope="module", autouse=True)
def base_4873():
    old = hr.POLICY, hr.argv_for.__defaults__
    hr.set_base_policy(c0.BASE)
    yield
    hr.POLICY, hr.argv_for.__defaults__ = old


@pytest.fixture(scope="module")
def c0_log(tmp_path_factory):
    d0 = tmp_path_factory.mktemp("plain")
    h = build_simulator(hr.argv_for(SC, d0))
    h.run()
    h.write_output()
    s = hr.score_output(d0, SC)
    taus = {m: s[f"p99_{m}"] for m in c0.METRICS4}
    root = tmp_path_factory.mktemp("c0")
    d = c0.window_dir(root, SC)
    d.mkdir(parents=True)
    header = {"decision_sha": c0.c0_sha(), "code_version": ev.code_version(), "taus": taus, "cut": s["cut"]}
    c0.log_window(SC, d / "log.jsonl", taus, s["cut"], every=6, workers=4, header=header, max_points=3)
    (d / "summary.json").write_text(json.dumps({"w": 1000.0, "s_tbt": 0.1}))
    return root


def _strip(c):
    return {k: v for k, v in c.items() if k != "cpu_s"}


def test_zero_jitter_reproduces_log_and_jitter_changes_it(c0_log, tmp_path):
    out0 = tmp_path / "zero"
    out0.mkdir()
    r = r0.run_window(SC, out0, every=6, workers=4, timeout=600, c0_dir=c0_log, delta=0.0)
    assert r["decisions"] == r["new"] == 3
    rows = [json.loads(x) for x in (out0 / f"{SC.tag}.jsonl").read_text().splitlines()]
    for row in rows:
        for s in ("1", "2"):
            assert {c: _strip(t) for c, t in row["reps"][s].items()} == \
                   {c: _strip(t) for c, t in row["reps"]["0"].items()}
    out1 = tmp_path / "jit"
    out1.mkdir()
    r0.run_window(SC, out1, every=6, workers=4, timeout=600, c0_dir=c0_log, delta=0.05)
    rows = [json.loads(x) for x in (out1 / f"{SC.tag}.jsonl").read_text().splitlines()]
    changed = 0
    for row in rows:
        n = {t["n"] for s in ("0", "1", "2") for t in row["reps"][s].values()}
        assert len(n) == 1  # the affected set is the same requests under every future
        changed += any(_strip(row["reps"]["1"][c]) != _strip(row["reps"]["0"][c]) for c in row["reps"]["0"])
        assert row["reps"]["1"] != row["reps"]["2"]
    assert changed == len(rows)
    # resume: nothing new
    assert r0.run_window(SC, out1, every=6, workers=4, timeout=600, c0_dir=c0_log, delta=0.05)["new"] == 0
    res = r0.analyze(out1)
    assert res["decisions"] == 3 and set(res["targets"]) == set(r0.TARGETS)
