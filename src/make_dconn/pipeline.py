"""Run one unit: censor -> smooth -> correlate (stage ``dconn``), or z-score (stage ``zscore``).

Every output is built in a private work directory (node-local disk by
default), copied to the output directory under a ``.partial`` name and
renamed, and its JSON sidecar is written last. A sidecar with
``status: complete`` and a matching parameter hash is the only thing that
marks a unit as done, so an interrupted job never looks finished.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from make_dconn import cifti, provenance, workbench
from make_dconn.censor import censor, load_fd, load_runs
from make_dconn.config import DconnConfig, dump_config
from make_dconn.inputs import Unit, processing_tag, resolve_inputs, resolve_outputs

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


def work_root(config: DconnConfig) -> Path:
    if config.paths.work_dir is not None:
        return config.paths.work_dir
    for var in ("L_SCRATCH_JOB", "L_SCRATCH", "TMPDIR"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path(tempfile.gettempdir())


def _require_space(directory: Path, needed_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed_bytes:
        raise PipelineError(
            f"{directory} has {free / 1e9:.1f} GB free, need about {needed_bytes / 1e9:.1f} GB"
        )


def _publish(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    log.info("copying %s -> %s", src, dest)
    shutil.copyfile(src, partial)
    os.replace(partial, dest)


def stage_status(config: DconnConfig, unit: Unit, stage: str) -> str:
    """'complete', 'stale' (made with different parameters), or 'missing'."""
    out = resolve_outputs(config, unit)
    data, sidecar = (out.dconn, out.sidecar) if stage == "dconn" else (out.zscored, out.zscored_sidecar)
    meta = provenance.read_json(sidecar)
    if meta is None or meta.get("status") != "complete" or not data.is_file():
        return "missing"
    if meta.get("params_hash") != provenance.params_hash(config):
        return "stale"
    return "complete"


def _check_existing(config: DconnConfig, unit: Unit, stage: str, overwrite: bool) -> bool:
    """True if the stage is already done and should be skipped."""
    status = stage_status(config, unit, stage)
    if status == "complete" and not overwrite:
        log.info("%s %s already complete; skipping", unit.label, stage)
        return True
    if status == "stale" and not overwrite:
        raise PipelineError(
            f"{unit.label} {stage} exists but was made with different parameters; "
            "pass --overwrite to replace it"
        )
    return False


def _code_state(allow_dirty: bool, expected_commit: str | None) -> dict:
    if allow_dirty:
        state = provenance.git_state()
        if state["dirty"]:
            log.warning("running with uncommitted code changes (--allow-dirty)")
        return state
    return provenance.require_clean_commit(expected_commit)


def run_dconn(
    config: DconnConfig,
    unit: Unit,
    *,
    overwrite: bool = False,
    allow_dirty: bool = False,
    expected_commit: str | None = None,
) -> None:
    if _check_existing(config, unit, "dconn", overwrite):
        return
    git = _code_state(allow_dirty, expected_commit)
    started = provenance.now()
    inputs = resolve_inputs(config, unit)
    out = resolve_outputs(config, unit)
    wb = workbench.find_wb_command()
    wb_ver = workbench.wb_version(wb)
    log.info("%s: wb_command %s at %s", unit.label, wb_ver, wb)

    img, data = cifti.load_dtseries(inputs.dtseries)
    n_frames, n_gray = data.shape
    brain = img.header.get_axis(1)
    missing, n_exact_zeros = cifti.find_missing(data, brain)
    fill = config.smoothing.fill_missing and missing.size > 0
    log.info("%s: %d grayordinates have no data in any frame", unit.label, missing.size)
    if fill and n_exact_zeros:
        # -fix-zeros-surface would treat these real values as missing too
        raise PipelineError(f"{unit.label}: {n_exact_zeros} exact zeros in the data; cannot fill safely")
    if missing.size and not config.smoothing.fill_missing:
        log.warning("%s: leaving %d missing grayordinates as NaN (NaN rows in the dconn)", unit.label, missing.size)
    fd = load_fd(inputs.motion, config.censoring.fd_column)
    runs = load_runs(inputs.boundaries, n_frames)
    sd = cifti.frame_sd(data, exclude=missing) if config.censoring.outlier_frames.enabled else None
    result = censor(fd, runs, config.censoring, frame_sd=sd)
    summary = result.summary()
    log.info(
        "%s: keeping %d of %d frames (%.1f%%)",
        unit.label,
        summary["n_frames_kept"],
        n_frames,
        100 * summary["fraction_kept"],
    )
    if summary["n_frames_kept"] < 3:
        raise PipelineError(f"{unit.label}: only {summary['n_frames_kept']} frames survive censoring")

    kept = data[result.keep]
    del data
    if fill:
        kept[:, missing] = 0.0  # -fix-zeros-surface reads zero as "no data"
    present = np.ones(n_gray, dtype=bool)
    present[missing] = False
    n_zero_var = int(np.count_nonzero(kept.std(axis=0)[present] == 0))
    if n_zero_var:
        log.warning("%s: %d grayordinates have zero variance after censoring", unit.label, n_zero_var)

    root = work_root(config)
    # censored + smoothed dtseries and the dconn, with 10% headroom
    _require_space(root, int(1.1 * (2 * kept.nbytes + 4 * n_gray * n_gray)))
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(dir=root, prefix="make_dconn_") as tmp_name:
        tmp = Path(tmp_name)
        censored = tmp / "censored.dtseries.nii"
        cifti.write_censored(img, kept, censored)
        del kept, img
        corr_input = censored
        if config.smoothing.sigma_mm is not None:
            smoothed = tmp / "smoothed.dtseries.nii"
            cmd = workbench.smooth_command(
                wb,
                censored,
                smoothed,
                config.smoothing.sigma_mm,
                inputs.surface_left,
                inputs.surface_right,
                config.smoothing.merged_volume,
                fix_zeros_surface=fill,
            )
            workbench.run(cmd)
            commands.append(cmd)
            corr_input = smoothed
            if fill:
                cifti.require_finite(smoothed)
        if cifti.n_frames(corr_input) != summary["n_frames_kept"]:
            raise PipelineError(f"{corr_input} does not have the expected number of frames")

        dconn_tmp = tmp / out.dconn.name
        cmd = workbench.correlation_command(wb, corr_input, dconn_tmp, config.correlation.mem_limit_gb)
        workbench.run(cmd)
        commands.append(cmd)
        checks = cifti.check_dconn(
            dconn_tmp,
            corr_input,
            extra_rows=missing[np.linspace(0, missing.size - 1, min(4, missing.size)).astype(int)],
            require_all_finite=fill,
        )

        _publish(dconn_tmp, out.dconn)
        if config.keep_intermediates:
            stem = out.dconn.name.removesuffix("_stat-pearsoncorrelation_boldmap.dconn.nii")
            _publish(corr_input, out.dconn.with_name(f"{stem}_bold.dtseries.nii"))

    result.write_tsv(out.censoring, config.censoring.fd_column)
    cifti.write_missing_tsv(brain, missing, out.filled)
    provenance.write_json(
        out.sidecar,
        {
            "status": "complete",
            "stage": "dconn",
            "measure": "Pearson correlation (wb_command -cifti-correlation, no Fisher z)",
            "processing_tag": processing_tag(config),
            "params_hash": provenance.params_hash(config),
            "unit": {"subject": unit.subject, "task": unit.task, "split": unit.split},
            "started": started,
            "finished": provenance.now(),
            "git": git,
            "environment": provenance.environment(),
            "workbench_version": wb_ver,
            "inputs": {
                "dtseries": provenance.fingerprint(inputs.dtseries, hash_contents=False),
                "motion": provenance.fingerprint(inputs.motion, hash_contents=True),
                "boundaries": provenance.fingerprint(inputs.boundaries, hash_contents=True),
                "surface_left": provenance.fingerprint(inputs.surface_left, hash_contents=True),
                "surface_right": provenance.fingerprint(inputs.surface_right, hash_contents=True),
            },
            "censoring": summary,
            "censoring_file": out.censoring.name,
            "missing_grayordinates": {
                "n": int(missing.size),
                "filled": bool(fill),
                "method": (
                    "set to 0, smoothed with -fix-zeros-surface (values extrapolated from neighbours)"
                    if fill
                    else "left as NaN"
                ),
                "file": out.filled.name,
            },
            "zero_variance_grayordinates": n_zero_var,
            "commands": commands,
            "output_checks": checks,
            "config": dump_config(config),
        },
    )
    log.info("%s: wrote %s", unit.label, out.dconn)


def run_zscore(
    config: DconnConfig,
    unit: Unit,
    *,
    overwrite: bool = False,
    allow_dirty: bool = False,
    expected_commit: str | None = None,
) -> None:
    if _check_existing(config, unit, "zscore", overwrite):
        return
    if stage_status(config, unit, "dconn") != "complete":
        raise PipelineError(f"{unit.label}: the dconn is not complete; run the dconn stage first")
    git = _code_state(allow_dirty, expected_commit)
    started = provenance.now()
    out = resolve_outputs(config, unit)
    root = work_root(config)
    _require_space(root, int(1.1 * out.dconn.stat().st_size))
    with tempfile.TemporaryDirectory(dir=root, prefix="make_dconn_z_") as tmp_name:
        z_tmp = Path(tmp_name) / out.zscored.name
        blocks = cifti.zscore_dconn_file(out.dconn, z_tmp)
        _publish(z_tmp, out.zscored)
    provenance.write_json(
        out.zscored_sidecar,
        {
            "status": "complete",
            "stage": "zscore",
            "method": (
                "Each of the 9 blocks (left cortex, right cortex, subcortex x same) of the Pearson r "
                "dconn z-scored over all its elements: population SD, NaNs ignored, diagonal included. "
                "Reproduces temp_match_pfm zscore_dconn_v1.0.0.py (after Hermosillo et al. 2024)."
            ),
            "processing_tag": processing_tag(config),
            "params_hash": provenance.params_hash(config),
            "unit": {"subject": unit.subject, "task": unit.task, "split": unit.split},
            "source_dconn": provenance.fingerprint(out.dconn, hash_contents=False),
            "source_dconn_sidecar": out.sidecar.name,
            "started": started,
            "finished": provenance.now(),
            "git": git,
            "environment": provenance.environment(),
            "blocks": blocks,
        },
    )
    log.info("%s: wrote %s", unit.label, out.zscored)
