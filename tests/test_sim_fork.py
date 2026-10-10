"""A forked simulation (evolve.sim_fork) must reproduce the unforked run byte for byte, and a forced decision in a
fork must change only that fork."""

import filecmp
from pathlib import Path

from evolve import evaluate as ev
from evolve.sim_fork import build_simulator, override_next_decision

POLICY = str(ev.REPO / "evolve/finalists/tail4_ni.py")


def _argv(sc, out):
    return ev.sim_command("evolved", POLICY, sc, ev.window_csv(sc), Path(out))[3:]


def _files(d):
    f = next(Path(d).rglob("request_metrics.csv"))
    return f, f.parent / "token_gaps.npz"


def test_fork_reproduces_unforked_run_and_counterfactual_diverges(tmp_path):
    sc = ev.Scenario("Conv", 4, 1.5, 6000, 200, 1)
    a = build_simulator(_argv(sc, tmp_path / "A"))
    a.run()
    a.write_output()

    o = build_simulator(_argv(sc, tmp_path / "C"))
    assert o.run_to_decision(100)
    f = o.fork(str(tmp_path / "B" / "out"))
    g = o.fork(str(tmp_path / "G" / "out"))
    o.run()
    o.write_output()
    f.run()
    f.write_output()
    p = g.fork()
    first = p.sim._scheduler._request_queue[0]
    p.run(lambda h: h.decisions >= 100)
    chosen = p.sim._scheduler._assigned[first.id][0]
    override_next_decision(g, (chosen + 1) % sc.replicas)
    g.run()
    g.write_output()

    A, B, C, G = (_files(tmp_path / x) for x in "ABCG")
    for x in (B, C):
        assert filecmp.cmp(A[0], x[0], shallow=False) and filecmp.cmp(A[1], x[1], shallow=False)
    assert not filecmp.cmp(A[0], G[0], shallow=False)
