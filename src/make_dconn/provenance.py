"""Provenance: code version, environment, input fingerprints, parameter hash."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from make_dconn import __version__
from make_dconn.config import DconnConfig, to_dict

REPO_DIR = Path(__file__).resolve().parents[2]


class ProvenanceError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        raise ProvenanceError(f"cannot read git state of {REPO_DIR}: {err}") from err
    return {"commit": commit, "dirty": dirty}


def require_clean_commit(expected: str | None = None) -> dict:
    """Fail unless the code is committed (and, if given, at the expected commit)."""
    state = git_state()
    if state["dirty"]:
        raise ProvenanceError(
            f"uncommitted changes in {REPO_DIR}; commit first (or pass --allow-dirty for a test run)"
        )
    if expected is not None and state["commit"] != expected:
        raise ProvenanceError(
            f"code is at {state['commit']} but this run was submitted at {expected}; "
            "the code changed after submission"
        )
    return state


def environment() -> dict:
    return {
        "make_dconn": __version__,
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in ("numpy", "nibabel", "pyyaml")},
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "wb_command": shutil.which("wb_command"),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path: Path, hash_contents: bool) -> dict:
    """Size + mtime for every file; full sha256 only for small files (motion, boundaries)."""
    st = path.stat()
    info = {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }
    if hash_contents:
        info["sha256"] = sha256(path)
    return info


def params_hash(config: DconnConfig) -> str:
    """Hash of every config value that can change a dconn's contents.

    Paths, subjects, conditions and Slurm resources are excluded: they choose
    which dconns are made and where, not what is in them.
    """
    relevant = {
        "session": config.session,
        "inputs": to_dict(config.inputs),
        "censoring": to_dict(config.censoring),
        "smoothing": to_dict(config.smoothing),
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def write_json(path: Path, data: dict) -> None:
    """Write via a temporary file so a sidecar is never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())
