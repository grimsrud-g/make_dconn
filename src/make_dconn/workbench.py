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
    fix_zeros_surface: bool = False,
) -> list[str]:
    """Same call as pypfm (xcpd_inputs.smooth_cifti): sigma in mm, surface and volume.

    ``fix_zeros_surface`` makes Workbench treat surface zeros as missing data
    (left out of every average, and filled by extrapolation); used for the
    vertices XCP-D leaves without data, which are set to 0 beforehand.
    """
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
    if fix_zeros_surface:
        cmd.append("-fix-zeros-surface")
    return cmd


def dilate_command(
    wb: Path,
    dtseries_in: Path,
    dtseries_out: Path,
    surface_mm: float,
    surface_left: Path,
    surface_right: Path,
    bad_roi: Path,
) -> list[str]:
    """Replace only the vertices in ``bad_roi`` with a distance-weighted average of nearby good ones."""
    return [
        str(wb),
        "-cifti-dilate",
        str(dtseries_in),
        "COLUMN",
        f"{surface_mm:g}",
        "0",
        str(dtseries_out),
        "-left-surface",
        str(surface_left),
        "-right-surface",
        str(surface_right),
        "-bad-brainordinate-roi",
        str(bad_roi),
    ]


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

