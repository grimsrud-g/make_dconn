"""Frame censoring: FD threshold, optional BOLD-SD outliers, short-segment removal.

Order of operations (each step only considers frames that survived the
previous one):

1. FD: keep frames with FD <= threshold.
2. Optional outlier frames: among FD survivors, drop frames whose SD across
   grayordinates is more than ``n_mads`` scaled MADs from the median.
3. Contiguity: within each run, drop surviving stretches shorter than
   ``min_contiguous_frames``. Runs are taken from the XCP-D boundaries file so
   that the last frames of one run and the first frames of the next are never
   treated as one stretch.

Step 3 runs last so the final mask never contains a short stretch.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from make_dconn.config import Censoring

# MATLAB isoutlier's MAD scale factor, -1 / (sqrt(2) * erfcinv(3/2)).
MAD_SCALE = 1.482602218505602


class CensoringError(RuntimeError):
    """Motion/boundary inputs are inconsistent."""


@dataclass(frozen=True)
class Run:
    start: int  # 0-based, inclusive
    stop: int  # 0-based, exclusive
    session: str
    task: str
    run: str

    @property
    def n_frames(self) -> int:
        return self.stop - self.start


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_fd(motion_tsv: Path, column: str) -> np.ndarray:
    rows = _read_tsv(motion_tsv)
    if not rows:
        raise CensoringError(f"{motion_tsv} has no rows")
    if column not in rows[0]:
        raise CensoringError(f"column {column!r} not in {motion_tsv}; columns: {list(rows[0])}")
    values = [r[column].strip() for r in rows]
    return np.array([np.nan if v in ("", "n/a", "NaN", "nan") else float(v) for v in values])


def load_runs(boundaries_tsv: Path, n_frames: int) -> list[Run]:
    """Read run boundaries and check they tile [0, n_frames) exactly."""
    rows = _read_tsv(boundaries_tsv)
    runs = [
        Run(
            start=int(r["start_volume_0based"]),
            stop=int(r["stop_volume_0based_exclusive"]),
            session=r.get("session", ""),
            task=r.get("task", ""),
            run=r.get("run", ""),
        )
        for r in rows
    ]
    if not runs:
        raise CensoringError(f"{boundaries_tsv} lists no runs")
    expected_start = 0
    for run in runs:
        if run.start != expected_start or run.stop <= run.start:
            raise CensoringError(
                f"{boundaries_tsv}: runs must be contiguous and non-empty; "
                f"expected a run starting at {expected_start}, got [{run.start}, {run.stop})"
            )
        expected_start = run.stop
    if expected_start != n_frames:
        raise CensoringError(
            f"{boundaries_tsv}: runs cover {expected_start} frames but the data has {n_frames}"
        )
    return runs


def fd_pass_mask(fd: np.ndarray, threshold: float, runs: list[Run]) -> np.ndarray:
    """True where FD <= threshold.

    FD is undefined for the first frame of a run (no previous frame); a
    missing value there counts as 0, which is what XCP-D writes. A missing
    value anywhere else is an error.
    """
    fd = fd.copy()
    run_starts = {r.start for r in runs}
    missing = np.flatnonzero(np.isnan(fd))
    bad = [int(i) for i in missing if int(i) not in run_starts]
    if bad:
        raise CensoringError(f"FD is missing at frames that are not run starts: {bad[:10]}")
    fd[missing] = 0.0
    return fd <= threshold


def outlier_frames(frame_sd: np.ndarray, candidates: np.ndarray, n_mads: float) -> np.ndarray:
    """True where a candidate frame's SD is a MAD outlier relative to the other candidates."""
    flagged = np.zeros_like(candidates, dtype=bool)
    values = frame_sd[candidates]
    if values.size == 0:
        return flagged
    median = np.median(values)
    scaled_mad = MAD_SCALE * np.median(np.abs(values - median))
    flagged[candidates] = np.abs(values - median) > n_mads * scaled_mad
    return flagged


def short_segment_frames(keep: np.ndarray, runs: list[Run], min_length: int) -> np.ndarray:
    """True for kept frames that lie in a within-run stretch shorter than ``min_length``."""
    short = np.zeros_like(keep, dtype=bool)
    for run in runs:
        segment = keep[run.start : run.stop].astype(np.int8)
        # +1 marks a stretch start, -1 marks one past its end.
        edges = np.diff(np.concatenate(([0], segment, [0])))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for a, b in zip(starts, stops, strict=True):
            if b - a < min_length:
                short[run.start + a : run.start + b] = True
    return short


@dataclass(frozen=True)
class CensoringResult:
    fd: np.ndarray
    fd_pass: np.ndarray
    outlier: np.ndarray | None  # None when outlier removal is disabled
    short_segment: np.ndarray
    keep: np.ndarray
    runs: list[Run]

    def summary(self) -> dict:
        per_run = [
            {
                "session": r.session,
                "task": r.task,
                "run": r.run,
                "n_frames": r.n_frames,
                "n_kept": int(self.keep[r.start : r.stop].sum()),
            }
            for r in self.runs
        ]
        n = int(self.keep.size)
        return {
            "n_frames_total": n,
            "n_frames_fd_pass": int(self.fd_pass.sum()),
            "n_frames_outlier": None if self.outlier is None else int(self.outlier.sum()),
            "n_frames_short_segment": int(self.short_segment.sum()),
            "n_frames_kept": int(self.keep.sum()),
            "fraction_kept": float(self.keep.sum() / n),
            "per_run": per_run,
        }

    def write_tsv(self, path: Path, fd_column: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        run_of = np.empty(self.keep.size, dtype=int)
        for i, r in enumerate(self.runs):
            run_of[r.start : r.stop] = i
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(
                ["frame", "session", "task", "run", fd_column, "fd_pass", "outlier", "short_segment", "keep"]
            )
            for i in range(self.keep.size):
                r = self.runs[run_of[i]]
                w.writerow(
                    [
                        i,
                        r.session,
                        r.task,
                        r.run,
                        "n/a" if np.isnan(self.fd[i]) else repr(float(self.fd[i])),
                        int(self.fd_pass[i]),
                        "n/a" if self.outlier is None else int(self.outlier[i]),
                        int(self.short_segment[i]),
                        int(self.keep[i]),
                    ]
                )


def censor(
    fd: np.ndarray,
    runs: list[Run],
    params: Censoring,
    frame_sd: np.ndarray | None = None,
) -> CensoringResult:
    """Build the frame-keep mask. ``frame_sd`` is required when outlier removal is on."""
    n = runs[-1].stop
    if fd.size != n:
        raise CensoringError(f"motion file has {fd.size} frames but boundaries cover {n}")
    fd_pass = fd_pass_mask(fd, params.fd_threshold, runs)
    keep = fd_pass.copy()
    outlier = None
    if params.outlier_frames.enabled:
        if frame_sd is None or frame_sd.size != n:
            raise CensoringError("outlier removal needs one SD value per frame")
        outlier = outlier_frames(frame_sd, keep, params.outlier_frames.n_mads)
        keep &= ~outlier
    short = short_segment_frames(keep, runs, params.min_contiguous_frames)
    keep &= ~short
    return CensoringResult(fd=fd, fd_pass=fd_pass, outlier=outlier, short_segment=short, keep=keep, runs=runs)
