from __future__ import annotations

import numpy as np
import pytest

from make_dconn.censor import (
    MAD_SCALE,
    CensoringError,
    Run,
    censor,
    fd_pass_mask,
    load_fd,
    load_runs,
    outlier_frames,
    short_segment_frames,
)
from make_dconn.config import Censoring, OutlierFrames


def _runs(*lengths: int) -> list[Run]:
    runs, start = [], 0
    for i, n in enumerate(lengths):
        runs.append(Run(start=start, stop=start + n, session="01", task="t", run=str(i + 1)))
        start += n
    return runs


def _params(min_len: int = 5, outliers: bool = False) -> Censoring:
    return Censoring(
        fd_column="framewise_displacement_filtered",
        fd_threshold=0.2,
        min_contiguous_frames=min_len,
        outlier_frames=OutlierFrames(enabled=outliers, n_mads=3.0),
    )


def test_threshold_is_inclusive():
    fd = np.array([0.0, 0.2, 0.2000001, 0.1])
    assert fd_pass_mask(fd, 0.2, _runs(4)).tolist() == [True, True, False, True]


def test_missing_fd_allowed_only_at_run_start():
    runs = _runs(3, 3)
    fd = np.array([np.nan, 0.1, 0.1, np.nan, 0.1, 0.1])
    assert fd_pass_mask(fd, 0.2, runs).all()
    fd[1] = np.nan
    with pytest.raises(CensoringError, match="not run starts"):
        fd_pass_mask(fd, 0.2, runs)


def test_short_segments_counted_within_runs():
    # Two runs of 6 frames. 3 kept frames end run 1 and 3 start run 2: 6 in a
    # row overall, but each within-run stretch is only 3 long, so all are dropped.
    keep = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    short = short_segment_frames(keep, _runs(6, 6), 5)
    assert short.tolist() == keep.tolist()


def test_segments_at_or_above_minimum_survive():
    keep = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    short = short_segment_frames(keep, _runs(12), 5)
    assert np.flatnonzero(short).tolist() == [6, 7, 8, 9, 11]


def test_censor_end_to_end():
    fd = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = censor(fd, _runs(9, 6), _params())
    # run 1: frames 0-5 kept (6 long), frame 6 high FD, frames 7-8 too short
    # run 2: frames 9-14 kept (6 long)
    assert np.flatnonzero(result.keep).tolist() == [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]
    s = result.summary()
    assert s["n_frames_fd_pass"] == 14
    assert s["n_frames_short_segment"] == 2
    assert [r["n_kept"] for r in s["per_run"]] == [6, 6]
    assert s["n_frames_outlier"] is None


def test_outliers_use_matlab_scaled_mad():
    sd = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 5.0, 1.0])
    candidates = np.ones(8, dtype=bool)
    candidates[7] = False  # not a candidate, never flagged
    flagged = outlier_frames(sd, candidates, 3.0)
    assert flagged.tolist() == [False] * 6 + [True, False]
    assert MAD_SCALE == pytest.approx(1.4826, abs=1e-4)


def test_outlier_step_requires_frame_sd():
    with pytest.raises(CensoringError, match="SD value per frame"):
        censor(np.zeros(10), _runs(10), _params(outliers=True))


def test_load_fd_and_runs(tmp_path):
    motion = tmp_path / "motion.tsv"
    motion.write_text("framewise_displacement\tframewise_displacement_filtered\n0.0\tn/a\n0.3\t0.15\n")
    np.testing.assert_array_equal(load_fd(motion, "framewise_displacement_filtered"), [np.nan, 0.15])
    with pytest.raises(CensoringError, match="not in"):
        load_fd(motion, "fd")

    bounds = tmp_path / "boundaries.tsv"
    bounds.write_text(
        "session\ttask\trun\tstart_volume_0based\tstop_volume_0based_exclusive\n"
        "01\tnBack\t1\t0\t4\n02\tnBack\t1\t4\t10\n"
    )
    runs = load_runs(bounds, 10)
    assert [(r.start, r.stop, r.session) for r in runs] == [(0, 4, "01"), (4, 10, "02")]
    with pytest.raises(CensoringError, match="cover 10 frames"):
        load_runs(bounds, 11)


def test_runs_must_be_contiguous(tmp_path):
    bounds = tmp_path / "boundaries.tsv"
    bounds.write_text(
        "start_volume_0based\tstop_volume_0based_exclusive\n0\t4\n5\t10\n"
    )
    with pytest.raises(CensoringError, match="contiguous"):
        load_runs(bounds, 10)


def test_censoring_tsv(tmp_path):
    fd = np.array([0.0] * 6 + [0.9])
    result = censor(fd, _runs(7), _params())
    out = tmp_path / "c.tsv"
    result.write_tsv(out, "framewise_displacement_filtered")
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == [
        "frame", "session", "task", "run", "framewise_displacement_filtered",
        "fd_pass", "outlier", "short_segment", "keep",
    ]
    assert len(lines) == 8
    assert lines[-1].split("\t")[-4:] == ["0", "n/a", "0", "0"]
