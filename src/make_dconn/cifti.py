"""CIFTI reading/writing with nibabel: censoring, output checks, sectioned z-score."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

# Section bounds hardcoded in temp_match_pfm zscore_dconn_v1.0.0.py (91k fsLR).
REPROTM_91K_SECTIONS = [(0, 29696), (29696, 59412), (59412, 91282)]


class CiftiError(RuntimeError):
    pass


def load_dtseries(path: Path) -> tuple[nib.Cifti2Image, np.ndarray]:
    """Return the image and its (frames x grayordinates) float32 data in memory."""
    img = nib.load(path, mmap=False)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 2:
        raise CiftiError(f"{path}: expected 2-D dtseries, got shape {data.shape}")
    series = img.header.get_axis(0)
    if not isinstance(series, nib.cifti2.SeriesAxis):
        raise CiftiError(f"{path}: first axis is {type(series).__name__}, not a series")
    return img, data


def find_missing(
    data: np.ndarray, brain: nib.cifti2.BrainModelAxis, chunk_cols: int = 8192
) -> tuple[np.ndarray, int]:
    """Grayordinates with no data, and the number of exact zeros in the real data.

    XCP-D leaves some cortical vertices NaN in every frame (no BOLD coverage).
    Returns their indices. Fails if a grayordinate is non-finite in only some
    frames, because that is a different problem and must not be filled.
    """
    missing, partial, n_zeros = [], [], 0
    for c0 in range(0, data.shape[1], chunk_cols):
        block = data[:, c0 : c0 + chunk_cols]
        finite = np.isfinite(block)
        none = ~finite.any(axis=0)
        some = ~finite.all(axis=0) & ~none
        missing.extend((c0 + np.flatnonzero(none)).tolist())
        partial.extend((c0 + np.flatnonzero(some)).tolist())
        n_zeros += int(np.count_nonzero(block[finite] == 0))
    if partial:
        raise CiftiError(
            f"{len(partial)} grayordinates are non-finite in only some frames (first: {partial[:10]})"
        )
    missing_idx = np.array(missing, dtype=int)
    if missing_idx.size and brain.volume_mask[missing_idx].any():
        raise CiftiError(
            f"{int(brain.volume_mask[missing_idx].sum())} missing grayordinates are subcortical; "
            "only surface vertices can be filled"
        )
    return missing_idx, n_zeros


def write_missing_tsv(brain: nib.cifti2.BrainModelAxis, missing: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("grayordinate\tstructure\tvertex\n")
        for g in missing:
            f.write(f"{int(g)}\t{brain.name[g]}\t{int(brain.vertex[g])}\n")


def frame_sd(data: np.ndarray, exclude: np.ndarray | None = None, chunk_rows: int = 512) -> np.ndarray:
    """SD across grayordinates for each frame (float64), skipping ``exclude`` columns."""
    cols = np.ones(data.shape[1], dtype=bool)
    if exclude is not None:
        cols[exclude] = False
    out = np.empty(data.shape[0])
    for r0 in range(0, data.shape[0], chunk_rows):
        out[r0 : r0 + chunk_rows] = data[r0 : r0 + chunk_rows][:, cols].std(axis=1, dtype=np.float64)
    return out


def require_finite(path: Path, chunk_rows: int = 512) -> None:
    """Fail if a dtseries contains any non-finite value (read in frame chunks)."""
    img = nib.load(path)
    for r0 in range(0, img.shape[0], chunk_rows):
        block = np.asarray(img.dataobj[r0 : r0 + chunk_rows])
        if not np.isfinite(block).all():
            cols = np.flatnonzero(~np.isfinite(block).all(axis=0))
            raise CiftiError(f"{path}: non-finite values at {cols.size} grayordinates (first: {cols[:10].tolist()})")


def write_censored(img: nib.Cifti2Image, kept: np.ndarray, out: Path) -> None:
    """Write the kept frames (already selected). The series step stays the original TR."""
    series = img.header.get_axis(0)
    brain = img.header.get_axis(1)
    new_series = nib.cifti2.SeriesAxis(start=0.0, step=series.step, size=kept.shape[0], unit=series.unit)
    out_img = nib.Cifti2Image(np.ascontiguousarray(kept, dtype=np.float32), header=(new_series, brain))
    out_img.set_data_dtype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_img.to_filename(out)


def n_frames(path: Path) -> int:
    return int(nib.load(path).shape[0])


def check_dconn(
    dconn: Path,
    reference: Path,
    n_probe_rows: int = 12,
    extra_rows: np.ndarray | None = None,
    require_all_finite: bool = False,
) -> dict:
    """Header and spot-value checks on a finished dconn without loading all of it.

    Fails if the matrix is not square, its grayordinates differ from the
    input dtseries, any probe-row value lies outside [-1, 1], or a finite
    diagonal value is not 1. NaNs are counted; with ``require_all_finite``
    any NaN in a probe row is a failure. ``extra_rows`` (e.g. filled
    vertices) are probed in addition to the evenly spaced rows.
    """
    img = nib.load(dconn)
    ref_brain = nib.load(reference).header.get_axis(1)
    n = ref_brain.size
    if img.shape != (n, n):
        raise CiftiError(f"{dconn}: shape {img.shape}, expected ({n}, {n})")
    for axis in (0, 1):
        if img.header.get_axis(axis) != ref_brain:
            raise CiftiError(f"{dconn}: axis {axis} grayordinates differ from {reference}")
    rows = np.linspace(0, n - 1, n_probe_rows).astype(int)
    if extra_rows is not None and len(extra_rows):
        rows = np.unique(np.concatenate([rows, np.asarray(extra_rows, dtype=int)]))
    diag, n_nan = [], 0
    for i in rows:
        row = np.asarray(img.dataobj[i, :], dtype=np.float64)
        finite = row[np.isfinite(row)]
        n_nan += row.size - finite.size
        if require_all_finite and finite.size != row.size:
            raise CiftiError(f"{dconn}: row {i} has {row.size - finite.size} non-finite values")
        if finite.size and (finite.min() < -1 - 1e-5 or finite.max() > 1 + 1e-5):
            raise CiftiError(f"{dconn}: row {i} outside [-1, 1] (min {finite.min()}, max {finite.max()})")
        diag.append(float(row[i]))
    finite_diag = [d for d in diag if np.isfinite(d)]
    if not np.allclose(finite_diag, 1.0, atol=1e-4):
        raise CiftiError(f"{dconn}: diagonal values at probe rows are not 1: {diag}")
    return {
        "shape": list(img.shape),
        "dtype": str(img.get_data_dtype()),
        "probe_rows": rows.tolist(),
        "probe_diagonal": diag,
        "probe_nonfinite_values": int(n_nan),
    }


def zscore_sections(brain: nib.cifti2.BrainModelAxis) -> list[tuple[int, int]]:
    """Left cortex, right cortex, and all volume (subcortical) grayordinates.

    Read from the header rather than assumed, then checked against the bounds
    hardcoded in the original zscore_dconn script for 91k files.
    """
    structures = list(brain.iter_structures())
    names = [name for name, _, _ in structures]
    if names[:2] != ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]:
        raise CiftiError(f"expected left then right cortex first, got {names[:2]}")
    left, right = structures[0][1], structures[1][1]
    if left.start != 0 or right.start != left.stop:
        raise CiftiError("cortex grayordinates are not contiguous from index 0")
    if not np.all(brain.volume_mask[right.stop :]) or np.any(brain.volume_mask[: right.stop]):
        raise CiftiError("expected all grayordinates after right cortex to be volume (subcortical)")
    sections = [(0, left.stop), (left.stop, right.stop), (right.stop, brain.size)]
    if brain.size == 91282 and sections != REPROTM_91K_SECTIONS:
        raise CiftiError(f"91k sections {sections} differ from zscore_dconn's {REPROTM_91K_SECTIONS}")
    return sections


def _nan_mean_std(block: np.ndarray, chunk_rows: int) -> tuple[float, float, int]:
    """Population (ddof=0) mean/SD over all non-NaN elements, float64, two passes."""
    total, count = 0.0, 0
    for i in range(0, block.shape[0], chunk_rows):
        chunk = block[i : i + chunk_rows].astype(np.float64)
        finite = ~np.isnan(chunk)
        total += chunk[finite].sum()
        count += int(finite.sum())
    if count == 0:
        raise CiftiError("section has no non-NaN values")
    mean = total / count
    ss = 0.0
    for i in range(0, block.shape[0], chunk_rows):
        chunk = block[i : i + chunk_rows].astype(np.float64)
        ss += np.nansum((chunk - mean) ** 2)
    return mean, float(np.sqrt(ss / count)), count


def zscore_in_place(mat: np.ndarray, sections: list[tuple[int, int]], chunk_rows: int = 2048) -> list[dict]:
    """Z-score each (row section x column section) block over all its elements.

    Reproduces zscore_dconn_v1.0.0.py, i.e. scipy.stats.zscore(block,
    axis=None, nan_policy='omit'): population SD, NaNs ignored and kept as
    NaN, diagonal included. Works in place on a float32 matrix so peak
    memory is one copy of the dconn plus one float64 chunk.
    """
    stats = []
    for r0, r1 in sections:
        for c0, c1 in sections:
            block = mat[r0:r1, c0:c1]
            mean, sd, count = _nan_mean_std(block, chunk_rows)
            if not sd > 0:
                raise CiftiError(f"block rows {r0}:{r1} cols {c0}:{c1} has zero SD")
            for i in range(0, block.shape[0], chunk_rows):
                chunk = block[i : i + chunk_rows].astype(np.float64)
                block[i : i + chunk_rows] = (chunk - mean) / sd
            stats.append({"rows": [r0, r1], "cols": [c0, c1], "mean": mean, "sd": sd, "n": count})
            log.info("z-scored block rows %d:%d cols %d:%d (mean %.6f, sd %.6f)", r0, r1, c0, c1, mean, sd)
    return stats


def zscore_dconn_file(dconn_in: Path, dconn_out: Path) -> list[dict]:
    img = nib.load(dconn_in, mmap=False)
    mat = np.asarray(img.dataobj, dtype=np.float32)
    if not mat.flags.writeable:
        mat = mat.copy()
    brain = img.header.get_axis(1)
    if img.header.get_axis(0) != brain:
        raise CiftiError(f"{dconn_in}: row and column grayordinates differ")
    stats = zscore_in_place(mat, zscore_sections(brain))
    out_img = nib.Cifti2Image(mat, header=img.header, nifti_header=img.nifti_header)
    out_img.set_data_dtype(np.float32)
    dconn_out.parent.mkdir(parents=True, exist_ok=True)
    out_img.to_filename(dconn_out)
    return stats
