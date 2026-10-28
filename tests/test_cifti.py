from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from make_dconn import cifti


def _brain(n_left=4, n_right=3, vol_shape=(2, 2, 1)) -> nib.cifti2.BrainModelAxis:
    left = nib.cifti2.BrainModelAxis.from_mask(np.ones(n_left, bool), name="CortexLeft")
    right = nib.cifti2.BrainModelAxis.from_mask(np.ones(n_right, bool), name="CortexRight")
    vol = nib.cifti2.BrainModelAxis.from_mask(np.ones(vol_shape, bool), name="ThalamusLeft", affine=np.eye(4))
    return left + right + vol


def _reference_zscore(dconn: np.ndarray, sections) -> np.ndarray:
    """The original zscore_dconn maths: scipy zscore(axis=None, nan_policy='omit') per block."""
    out = np.zeros_like(dconn, dtype=np.float64)
    for r0, r1 in sections:
        for c0, c1 in sections:
            block = dconn[r0:r1, c0:c1].astype(np.float64)
            out[r0:r1, c0:c1] = (block - np.nanmean(block)) / np.nanstd(block)
    return out


def test_sections_from_header():
    assert cifti.zscore_sections(_brain()) == [(0, 4), (4, 7), (7, 11)]


def test_sections_reject_unexpected_order():
    right = nib.cifti2.BrainModelAxis.from_mask(np.ones(3, bool), name="CortexRight")
    left = nib.cifti2.BrainModelAxis.from_mask(np.ones(4, bool), name="CortexLeft")
    with pytest.raises(cifti.CiftiError, match="left then right"):
        cifti.zscore_sections(right + left)


def test_zscore_matches_reference_with_nans():
    rng = np.random.default_rng(0)
    n = 11
    a = rng.normal(size=(n, n))
    dconn = np.corrcoef(a).astype(np.float32)
    dconn[2, 9] = dconn[9, 2] = np.nan
    sections = cifti.zscore_sections(_brain())
    expected = _reference_zscore(dconn, sections)
    got = dconn.copy()
    stats = cifti.zscore_in_place(got, sections, chunk_rows=2)  # chunking must not matter
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert np.isnan(got[2, 9]) and np.isnan(got[9, 2])
    assert len(stats) == 9
    for (r0, r1), (c0, c1) in [((r0, r1), (c0, c1)) for r0, r1 in sections for c0, c1 in sections]:
        block = got[r0:r1, c0:c1].astype(np.float64)
        assert np.nanmean(block) == pytest.approx(0, abs=1e-6)
        assert np.nanstd(block) == pytest.approx(1, abs=1e-6)


def test_zscore_file_round_trip(tmp_path):
    brain = _brain()
    rng = np.random.default_rng(1)
    dconn = np.corrcoef(rng.normal(size=(brain.size, 30))).astype(np.float32)
    src = tmp_path / "in.dconn.nii"
    nib.Cifti2Image(dconn, header=(brain, brain)).to_filename(src)
    dst = tmp_path / "out.dconn.nii"
    cifti.zscore_dconn_file(src, dst)
    img = nib.load(dst)
    assert img.get_data_dtype() == np.float32
    assert img.header.get_axis(0) == brain and img.header.get_axis(1) == brain
    expected = _reference_zscore(dconn, cifti.zscore_sections(brain))
    np.testing.assert_allclose(np.asarray(img.dataobj), expected, rtol=1e-6, atol=1e-6)


def test_write_censored_and_check(tmp_path):
    brain = _brain()
    series = nib.cifti2.SeriesAxis(start=0, step=1.49, size=20, unit="SECOND")
    rng = np.random.default_rng(2)
    data = rng.normal(size=(20, brain.size)).astype(np.float32)
    src = tmp_path / "in.dtseries.nii"
    nib.Cifti2Image(data, header=(series, brain)).to_filename(src)

    img, loaded = cifti.load_dtseries(src)
    np.testing.assert_array_equal(loaded, data)
    keep = np.zeros(20, bool)
    keep[[0, 1, 2, 5, 6, 7, 8, 19]] = True
    out = tmp_path / "censored.dtseries.nii"
    cifti.write_censored(img, loaded[keep], out)
    out_img = nib.load(out)
    np.testing.assert_array_equal(np.asarray(out_img.dataobj), data[keep])
    assert out_img.header.get_axis(0).step == pytest.approx(1.49)
    assert out_img.header.get_axis(1) == brain
    assert cifti.n_frames(out) == 8

    # a dconn computed from it passes the checks; a corrupted one does not
    r = np.corrcoef(data[keep].T).astype(np.float32)
    good = tmp_path / "good.dconn.nii"
    nib.Cifti2Image(r, header=(brain, brain)).to_filename(good)
    checks = cifti.check_dconn(good, out, n_probe_rows=brain.size)
    assert checks["probe_diagonal"] == pytest.approx([1.0] * brain.size)
    bad = r.copy()
    bad[3, 3] = 0.5
    bad_path = tmp_path / "bad.dconn.nii"
    nib.Cifti2Image(bad, header=(brain, brain)).to_filename(bad_path)
    with pytest.raises(cifti.CiftiError, match="diagonal"):
        cifti.check_dconn(bad_path, out, n_probe_rows=brain.size)


def test_find_missing_whole_columns_only():
    brain = _brain()
    data = np.random.default_rng(3).normal(size=(10, brain.size)).astype(np.float32)
    data[:, [1, 5]] = np.nan
    missing, n_zeros = cifti.find_missing(data, brain, chunk_cols=3)  # chunking must not matter
    assert missing.tolist() == [1, 5] and n_zeros == 0
    data[0, 2] = 0.0
    assert cifti.find_missing(data, brain)[1] == 1
    data[4, 3] = np.nan  # partly missing: a different problem, never filled
    with pytest.raises(cifti.CiftiError, match="only some frames"):
        cifti.find_missing(data, brain)


def test_find_missing_rejects_subcortical():
    brain = _brain()
    data = np.ones((4, brain.size), dtype=np.float32)
    data[:, brain.size - 1] = np.nan  # last grayordinate is volume
    with pytest.raises(cifti.CiftiError, match="subcortical"):
        cifti.find_missing(data, brain)


def test_frame_sd_skips_missing_columns():
    data = np.random.default_rng(4).normal(size=(7, 11)).astype(np.float32)
    ref = np.delete(data, [2, 8], axis=1).std(axis=1, dtype=np.float64)
    data[:, [2, 8]] = np.nan
    np.testing.assert_allclose(cifti.frame_sd(data, exclude=np.array([2, 8]), chunk_rows=3), ref)


def test_missing_tsv_and_require_finite(tmp_path):
    brain = _brain()
    out = tmp_path / "filled.tsv"
    cifti.write_missing_tsv(brain, np.array([1, 5]), out, {1: "smoothing", 5: "dilation"})
    assert out.read_text().splitlines() == [
        "grayordinate\tstructure\tvertex\tfill_method",
        "1\tCIFTI_STRUCTURE_CORTEX_LEFT\t1\tsmoothing",
        "5\tCIFTI_STRUCTURE_CORTEX_RIGHT\t1\tdilation",
    ]
    series = nib.cifti2.SeriesAxis(start=0, step=1, size=6, unit="SECOND")
    data = np.ones((6, brain.size), dtype=np.float32)
    ok = tmp_path / "ok.dtseries.nii"
    nib.Cifti2Image(data, header=(series, brain)).to_filename(ok)
    cifti.require_finite(ok, chunk_rows=4)
    data[5, 3] = np.nan
    bad = tmp_path / "bad.dtseries.nii"
    nib.Cifti2Image(data, header=(series, brain)).to_filename(bad)
    with pytest.raises(cifti.CiftiError, match="non-finite"):
        cifti.require_finite(bad, chunk_rows=4)


def test_constant_columns_and_roi(tmp_path):
    brain = _brain()
    series = nib.cifti2.SeriesAxis(start=0, step=1, size=9, unit="SECOND")
    data = np.random.default_rng(5).normal(size=(9, brain.size)).astype(np.float32)
    data[:, 2] = 0.0
    data[:, 7] = 3.5
    data[8, 4] = data[0, 4] = 1.0  # varies: not constant
    f = tmp_path / "x.dtseries.nii"
    nib.Cifti2Image(data, header=(series, brain)).to_filename(f)
    assert cifti.constant_columns(f, chunk_rows=4).tolist() == [2, 7]

    roi = tmp_path / "roi.dscalar.nii"
    cifti.write_roi(brain, np.array([2, 7]), roi)
    img = nib.load(roi)
    assert img.header.get_axis(1) == brain
    assert np.flatnonzero(np.asarray(img.dataobj)[0]).tolist() == [2, 7]
