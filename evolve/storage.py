"""Keep bulky outputs off the Mac mini's internal disk (2026-10-10: the internal disk filled up and stalled every job).

On the Mac, bulky directories of this checkout (``cache/``, ``evolve/runs/``, ``simulator_output/``, ...) are symlinks
into the external SSD, and temporary simulation outputs go to an SSD scratch directory. Two failure modes are guarded:

- The SSD is not mounted. A write through such a link then fails, or a ``mkdir -p`` recreates the path under
  ``/Volumes`` on the internal disk and fills it silently. ``check()`` refuses to run instead.
- The internal disk is nearly full anyway. ``check()`` refuses below ``EVOLVE_MIN_FREE_GB`` (default 5 GB).

``check()`` runs when ``evolve`` is imported. It does nothing where the checkout has no links into ``/Volumes`` (Alpine,
CI). Set ``EVOLVE_STORAGE_CHECK=0`` to skip it (e.g. read-only analysis while the SSD is away).
"""
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Directories that live on the SSD when it is configured (see scripts/offload_to_ssd.sh).
OFFLOADED = ("cache", "evolve/runs", "evolve/rl_plan/out", "simulator_output", "results",
             "rl_models", "scratch")  # untracked only (outputs/, graduation/ hold tracked files)
SCRATCH = REPO / "scratch"  # temporary simulation outputs (tempfile.tempdir / TMPDIR for this process and children)


class StorageError(RuntimeError):
    pass


def volume_root(target: str):
    """The /Volumes/<name> mount point a path lives under, or None for paths on the internal disk."""
    parts = Path(target).parts
    return Path(*parts[:3]) if len(parts) >= 3 and parts[1] == "Volumes" else None


def offload_links(repo: Path = REPO) -> dict:
    """{link path: target} for this checkout's links into /Volumes."""
    out = {}
    for rel in OFFLOADED:
        p = repo / rel
        if p.is_symlink():
            t = os.readlink(p)
            if volume_root(t) is not None:
                out[p] = t
    return out


def check(repo: Path = REPO, min_free_gb: float = None, set_tmp: bool = True) -> dict:
    links = offload_links(repo)
    if not links:
        return {}
    for link, target in links.items():
        root = volume_root(target)
        if not os.path.ismount(root):
            raise StorageError(
                f"{link} points to {target}, but {root} is not mounted: writing there would fail or fill the internal "
                f"disk. Mount the SSD (sudo diskutil mount <disk>) or set EVOLVE_STORAGE_CHECK=0 for read-only work.")
        if not os.access(target, os.W_OK):
            raise StorageError(f"{target} (via {link}) is not writable: grant this app access to removable volumes "
                               "(System Settings > Privacy & Security > Files and Folders) and check the directory.")
    min_free = float(os.environ.get("EVOLVE_MIN_FREE_GB", 5)) if min_free_gb is None else min_free_gb
    free = shutil.disk_usage(Path.home()).free / 2**30
    if free < min_free:
        raise StorageError(f"internal disk has {free:.1f} GB free (< {min_free} GB): move outputs to the SSD before "
                           "running more simulations (EVOLVE_MIN_FREE_GB overrides the threshold).")
    if set_tmp and SCRATCH.is_symlink() and SCRATCH in links:
        tmp = SCRATCH / "tmp"
        tmp.mkdir(exist_ok=True)
        tempfile.tempdir = str(tmp)  # this process
        os.environ["TMPDIR"] = str(tmp)  # simulator subprocesses
    return {str(k): v for k, v in links.items()}


if os.environ.get("EVOLVE_STORAGE_CHECK", "1") != "0":
    check()
