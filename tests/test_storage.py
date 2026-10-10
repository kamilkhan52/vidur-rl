import os

import pytest

from evolve import storage


def _repo(tmp_path, target="/Volumes/FakeSSD/MacOffload/x/cache"):
    (tmp_path / "evolve").mkdir()
    os.symlink(target, tmp_path / "cache")
    return tmp_path


def test_no_links_is_a_no_op(tmp_path):
    assert storage.check(tmp_path) == {}


def test_unmounted_volume_is_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(os.path, "ismount", lambda p: False)
    with pytest.raises(storage.StorageError, match="not mounted"):
        storage.check(repo)


def test_unwritable_target_is_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(os.path, "ismount", lambda p: str(p) == "/Volumes/FakeSSD")
    monkeypatch.setattr(os, "access", lambda p, m: False)
    with pytest.raises(storage.StorageError, match="not writable"):
        storage.check(repo)


def test_low_internal_space_is_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(os.path, "ismount", lambda p: True)
    monkeypatch.setattr(os, "access", lambda p, m: True)
    with pytest.raises(storage.StorageError, match="GB free"):
        storage.check(repo, min_free_gb=10**9)
    assert storage.check(repo, min_free_gb=0, set_tmp=False) == {str(repo / "cache"): "/Volumes/FakeSSD/MacOffload/x/cache"}


def test_internal_targets_are_not_offload_links(tmp_path):
    (tmp_path / "evolve").mkdir()
    os.symlink(str(tmp_path / "elsewhere"), tmp_path / "cache")
    assert storage.offload_links(tmp_path) == {}
