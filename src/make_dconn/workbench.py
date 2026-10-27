"""Thin, logged wrappers around the wb_command calls this package uses."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class WorkbenchError(RuntimeError):
    pass


def find_wb_command() -> Path:
    found = shutil.which("wb_command")
    if found is None:
        raise WorkbenchError("wb_command not on PATH (load the workbench module first)")
    return Path(found)


def wb_version(wb: Path) -> str:
    out = subprocess.run([str(wb), "-version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"Version:\s*(\S+)", out)
    if match is None:
        raise WorkbenchError(f"could not parse wb_command version from:\n{out}")
    return match.group(1)


def run(cmd: list[str]) -> None:
    log.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        log.info("wb_command stdout:\n%s", result.stdout.strip())
    if result.returncode != 0:
        raise WorkbenchError(
            f"wb_command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    if result.stderr.strip():
        log.warning("wb_command stderr:\n%s", result.stderr.strip())


def smooth_command(
    wb: Path,
    dtseries_in: Path,
    dtseries_out: Path,
    sigma_mm: float,
    surface_left: Path,
    surface_right: Path,
    merged_volume: bool,
) -> list[str]:
    """Same call as pypfm (xcpd_inputs.smooth_cifti): sigma in mm, surface and volume."""
    cmd = [
        str(wb),
        "-cifti-smoothing",
        str(dtseries_in),
        str(sigma_mm),
        str(sigma_mm),
        "COLUMN",
        str(dtseries_out),
        "-left-surface",
        str(surface_left),
        "-right-surface",
        str(surface_right),
    ]
    if merged_volume:
        cmd.append("-merged-volume")
    return cmd


def correlation_command(wb: Path, dtseries_in: Path, dconn_out: Path, mem_limit_gb: float) -> list[str]:
    """Pearson r between all grayordinates (no Fisher z)."""
    return [
        str(wb),
        "-cifti-correlation",
        str(dtseries_in),
        str(dconn_out),
        "-mem-limit",
        f"{mem_limit_gb:g}",
    ]

