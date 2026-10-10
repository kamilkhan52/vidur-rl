"""Golden-set helpers: hashing and manifest comparison (no simulations)."""
import json
from pathlib import Path

import numpy as np

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


def _cell(rm="x", tg="y", p99=10.0, ok=True):
    return {"scenario": "s", "policy": "lor", "ok": ok, "request_metrics_sha256": rm, "token_gaps_sha256": tg,
            "summary": {"p99_e2e": p99}}


def test_compare_reports_changed_files_and_metric_moves():
    assert golden.compare([_cell()], [_cell()]) == []
    d = golden.compare([_cell()], [_cell(rm="z", p99=11.0)])
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
        assert len(m["cells"]) == len(golden.SCENARIOS) * len(golden.POLICIES)
        for c in m["cells"]:
            assert c["ok"] and len(c["request_metrics_sha256"]) == 64 and len(c["token_gaps_sha256"]) == 64
