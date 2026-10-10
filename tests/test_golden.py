"""Golden-set tool: hashing, manifest comparison and the run plumbing, with a fake simulator (no simulations)."""
import json
import socket
import subprocess
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evolve.evaluate as ev
from evolve.analysis import golden


def test_sha_npz_ignores_container_and_key_order(tmp_path):
    gap, req = np.array([0.1, 0.2, 0.3]), np.array([1, 1, 2])
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, gap=gap, request_id=req)
    np.savez(b, request_id=req, gap=gap)  # different container (uncompressed) and key order
    assert golden.sha_npz(a) == golden.sha_npz(b)
    assert golden.sha_file(a) != golden.sha_file(b)
    c = tmp_path / "c.npz"
    np.savez_compressed(c, gap=gap.astype(np.float32), request_id=req)  # same values, different dtype
    assert golden.sha_npz(c) != golden.sha_npz(a)


def _cell(rm="1" * 64, tg="2" * 64, p99=10.0, ok=True):
    return {"scenario": "s", "policy": "lor", "ok": ok, "request_metrics_sha256": rm, "token_gaps_sha256": tg,
            "summary": {"p99_e2e": p99}}


def test_compare_reports_changed_files_and_metric_moves():
    assert golden.compare([_cell()], [_cell()]) == []
    d = golden.compare([_cell()], [_cell(rm="3" * 64, p99=11.0)])
    assert len(d) == 1 and "request_metrics_sha256" in d[0] and "+10.000 %" in d[0]
    assert "missing" in golden.compare([_cell()], [])[0]
    assert "ok True -> False" in golden.compare([_cell()], [_cell(ok=False)])[0]


def test_committed_manifests_have_schema():
    for p in Path(golden.OUT_DIR).glob("*.json"):
        m = json.loads(p.read_text())
        assert m["schema"] == golden.SCHEMA
        env = m["environment"]
        for k in ("code_version", "host", "python", "numpy", "pandas", "sklearn", "bundles", "traces"):
            assert k in env, (p, k)
        spec = golden.cells_for(m.get("set", "default"), m.get("include_pending", False))
        assert {(c["scenario"], c["policy"]) for c in m["cells"]} == {(sc.tag, n) for sc, n in spec}, p
        assert golden.validate(m["cells"]) == [], p


# --- fake simulator ------------------------------------------------------------------------------------------------

N = 20


@pytest.fixture
def fake_sim(tmp_path, monkeypatch):
    """Replace the simulator subprocess with one that writes outputs whose content depends on the command, and the
    window files with a tiny trace. ``state["offset"]`` perturbs every output (a simulator change)."""
    state = {"offset": 0.0}
    trace = tmp_path / "window.csv"
    pd.DataFrame({"arrived_at": np.arange(N, dtype=float)}).to_csv(trace, index=False)

    def run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        i = cmd.index("--metrics_config_output_dir") + 1
        out = Path(cmd[i]) / "2026-10-10_00-00-00-000000"
        out.mkdir(parents=True, exist_ok=True)
        h = sum(k * ord(ch) for k, ch in enumerate("|".join(cmd[3:i] + cmd[i + 1:]))) % 997 + state["offset"]
        pd.DataFrame({"Request Id": range(N), "request_arrived_at": np.arange(N, dtype=float),
                      "request_e2e_time": h + 1.0, "request_scheduling_delay": 0.0, "prefill_e2e_time": 0.1,
                      "request_num_prefill_tokens": 10, "request_num_decode_tokens": 5,
                      "request_num_prefill_tokens_cached": 0, "request_num_restarts": [1, 2] + [0] * (N - 2)}
                     ).to_csv(out / "request_metrics.csv", index=False, float_format="%.6f")
        np.savez_compressed(out / "token_gaps.npz", gap=np.full(N, h / 1e3), request_id=np.arange(N))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ev, "subprocess", types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(ev, "window_csv", lambda sc: trace)
    monkeypatch.delenv("EVOLVE_KEEP_DIR", raising=False)
    return state


CELLS = [(sc, n) for sc in golden.SCENARIOS[:2] for n in ("lor", "mooncake")] + [(golden.PREEMPT, "feedback")]


def test_run_all_hashes_each_cells_own_output(fake_sim, tmp_path):
    a = golden.run_all(2, tmp_path / "a", 60, CELLS)
    assert all(c["ok"] for c in a) and golden.validate(a) == []
    assert len({c["request_metrics_sha256"] for c in a}) == len({c["token_gaps_sha256"] for c in a}) == len(CELLS)
    assert all(c["restarted_requests"] == 2 and c["total_restarts"] == 3 for c in a)
    assert golden.compare_report(a, golden.run_all(2, tmp_path / "b", 60, CELLS)) == ([], len(CELLS), len(CELLS))
    fake_sim["offset"] = 1e-3  # a simulator change
    d = golden.compare(a, golden.run_all(2, tmp_path / "c", 60, CELLS))
    assert len(d) == len(CELLS) and all("request_metrics_sha256 differ, token_gaps_sha256 differ" in x for x in d)
    assert all("p99_e2e" in x and "mean_tbt" in x and "completed Δ0" in x for x in d)


def test_keep_dir_env_is_refused(fake_sim, tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_KEEP_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(RuntimeError, match="EVOLVE_KEEP_DIR"):
        golden.run_all(2, tmp_path / "a", 60, CELLS)
    assert golden.main(["build", "--out", str(tmp_path / "m.json")]) == 2
    assert golden.main(["check", "--manifest", str(tmp_path / "m.json")]) == 2
    assert not (tmp_path / "m.json").exists()


def test_missing_outputs_are_failures_and_never_identical(tmp_path, monkeypatch):
    """A run_one that succeeds but leaves nothing in the golden keep dir (e.g. outputs written elsewhere)."""
    monkeypatch.setattr(ev, "window_csv", lambda sc: None)
    monkeypatch.setattr(ev, "run_one", lambda *a, **k: {"ok": True, "p99_e2e": 1.0, "wall_s": 0.1})
    r1 = golden.run_all(2, tmp_path / "a", 60, CELLS)
    r2 = golden.run_all(2, tmp_path / "b", 60, CELLS)
    assert not any(c["ok"] for c in r1) and all("found 0 and 0" in c["reason"] for c in r1)
    assert len(golden.validate(r1)) == len(CELLS)
    diffs, compared, identical = golden.compare_report(r1, r2)
    assert len(diffs) == compared == len(CELLS) and identical == 0
    assert all("failed in both runs" in d for d in diffs)
    assert golden.report(r1, r2) == 1
    # a manifest cell marked ok but with no hashes (as EVOLVE_KEEP_DIR produced before the fix)
    none = [_cell(rm=None, tg=None)]
    assert golden.compare(none, none) == ["s / lor: request_metrics_sha256 missing, token_gaps_sha256 missing; "
                                          "p99_e2e Δ0"]
    assert golden.validate(none) == ["s / lor: missing hash"]


def test_ambiguous_outputs_fail(tmp_path, monkeypatch):
    def run_one(sched, sc, pol=None, timeout=0, extra=None):
        for stamp in ("t1", "t2"):  # e.g. a stale output next to the new one
            d = Path(golden._LOCAL.keep) / "out" / stamp
            d.mkdir(parents=True)
            (d / "request_metrics.csv").write_text("x\n")
            np.savez(d / "token_gaps.npz", gap=np.zeros(1), request_id=np.zeros(1))
        return {"ok": True, "wall_s": 0.1}

    monkeypatch.setattr(ev, "window_csv", lambda sc: None)
    monkeypatch.setattr(ev, "run_one", run_one)
    c = golden.run_all(1, tmp_path, 60, CELLS[:1])[0]
    assert not c["ok"] and "found 2 and 2" in c["reason"]


def test_cleanup_failure_fails_the_cell(fake_sim, tmp_path, monkeypatch):
    sc, name = CELLS[0]
    (tmp_path / f"{sc.tag}__{name}" / "old").mkdir(parents=True)

    def rmtree(*a, **k):
        raise PermissionError("busy")

    monkeypatch.setattr(golden.shutil, "rmtree", rmtree)
    c = golden.run_all(1, tmp_path, 60, CELLS[:1])[0]
    assert not c["ok"] and "cleanup" in c["reason"] and "busy" in c["reason"]


def test_compare_reports_one_sided_and_doubly_failed_cells():
    other = dict(_cell(), policy="sticky_lor")
    diffs, compared, identical = golden.compare_report([_cell()], [_cell(), other])
    assert diffs == ["s / sticky_lor: not in the reference"] and (compared, identical) == (1, 1)
    assert golden.compare([], [_cell()]) == ["s / lor: not in the reference"]
    assert golden.report([], []) == 1  # nothing compared is not a pass
    fail = _cell(ok=False)
    assert "failed in both runs" in golden.compare([fail], [fail])[0]
    d = golden.compare([_cell()], [_cell(tg="3" * 64)])[0]
    assert "token_gaps_sha256 differ" in d and "p99_e2e Δ0" in d  # equal metrics are still listed


def _stub_build(monkeypatch, tmp_path, ok=True):
    calls = []

    def run_all(workers, keep, timeout, cells):
        calls.append(len(cells))
        return [dict(_cell(rm="a" * 64, tg="b" * 64, ok=ok), scenario=sc.tag, policy=n, restarted_requests=1)
                for sc, n in cells]

    monkeypatch.setattr(golden, "OUT_DIR", tmp_path)
    monkeypatch.setattr(golden, "environment", lambda cells: {"bundles": {}, "code_version": ev.code_version()})
    monkeypatch.setattr(golden, "run_all", run_all)
    monkeypatch.delenv("EVOLVE_KEEP_DIR", raising=False)
    return calls


def test_build_names_by_data_and_refuses_to_overwrite(tmp_path, monkeypatch):
    calls = _stub_build(monkeypatch, tmp_path)
    host = socket.gethostname().split(".")[0]
    monkeypatch.setattr(ev, "V3_DATA", True)
    assert golden.main(["build"]) == 0 and (tmp_path / f"v3_{host}.json").exists()
    assert golden.main(["build"]) == 2 and calls == [30]  # refused before running anything
    assert golden.main(["build", "--force"]) == 0 and calls == [30, 30]
    ref = tmp_path / f"v3_{host}.json"  # a reference from other code is never overwritten, even with --force
    m = json.loads(ref.read_text())
    m["environment"]["code_version"] = "c6fceae7bfa8"
    ref.write_text(json.dumps(m))
    assert golden.main(["build", "--force"]) == 2 and calls == [30, 30]
    monkeypatch.setattr(ev, "V3_DATA", False)
    assert golden.main(["build"]) == 0 and (tmp_path / f"v4_{host}.json").exists()
    assert golden.main(["build", "--set", "extended"]) == 0
    m = json.loads((tmp_path / f"v4ext_{host}.json").read_text())
    assert m["set"] == "extended" and len(m["cells"]) == len(golden.cells_for("extended")) == calls[-1]
    # check re-runs the manifest's own set
    assert golden.main(["check", "--manifest", str(tmp_path / f"v4ext_{host}.json")]) == 0
    assert calls[-1] == len(golden.cells_for("extended"))


def test_failed_build_is_not_written_as_reference(tmp_path, monkeypatch):
    _stub_build(monkeypatch, tmp_path, ok=False)
    assert golden.main(["build", "--out", str(tmp_path / "ref.json")]) == 1
    assert not (tmp_path / "ref.json").exists() and (tmp_path / "ref.failed.json").exists()
