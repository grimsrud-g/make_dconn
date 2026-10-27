"""Freeze a run (config + manifest + code commit) and submit it as a Slurm job array.

Layout of a run directory, ``<output_dir>/_runs/<name>_<stage>_<timestamp>/``:

    config.yaml      fully resolved config the jobs read (later edits to the
                     source config cannot change a submitted run)
    manifest.tsv     one row per array index: subject, task, split
    submission.json  code commit, sbatch commands, job ids
    logs/            Slurm stdout/stderr, one file per array task
"""

from __future__ import annotations

import csv
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from make_dconn import provenance
from make_dconn.config import DconnConfig, SlurmResources, dump_config, load_config
from make_dconn.inputs import Unit, resolve_inputs, units_from_config
from make_dconn.pipeline import PipelineError, run_dconn, run_zscore, stage_status

log = logging.getLogger(__name__)

ARRAY_SCRIPT = provenance.REPO_DIR / "slurm" / "array.sbatch"
VENV_PYTHON = provenance.REPO_DIR / ".venv" / "bin" / "python"


def select_units(config: DconnConfig, units: list[Unit], stage: str, overwrite: bool) -> list[Unit]:
    """Units that still need ``stage``; stale outputs are an error unless overwriting."""
    selected = []
    for unit in units:
        status = stage_status(config, unit, stage)
        if status == "stale" and not overwrite:
            raise PipelineError(
                f"{unit.label} {stage} was made with different parameters; use --overwrite to redo it"
            )
        if status != "complete" or overwrite:
            selected.append(unit)
    return selected


def write_manifest(path: Path, units: list[Unit]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["index", "subject", "task", "split"])
        for i, u in enumerate(units):
            w.writerow([i, u.subject, u.task, "" if u.split is None else u.split])


def read_manifest_row(path: Path, index: int) -> Unit:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    row = rows[index]
    if int(row["index"]) != index:
        raise PipelineError(f"{path}: row {index} has index {row['index']}")
    return Unit(subject=row["subject"], task=row["task"], split=int(row["split"]) if row["split"] else None)


def _sbatch_command(
    config: DconnConfig,
    res: SlurmResources,
    stage: str,
    run_dir: Path,
    n_units: int,
    dependency: str | None,
) -> list[str]:
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name=md_{config.name}_{stage}",
        f"--array=0-{n_units - 1}",
        f"--partition={res.partition}",
        f"--time={res.time}",
        f"--cpus-per-task={res.cpus_per_task}",
        f"--mem={res.mem}",
        f"--output={run_dir / 'logs' / '%A_%a.out'}",
        f"--export=ALL,MAKE_DCONN_MODULES={' '.join(config.slurm.modules)}",
    ]
    if res.mail_user:
        cmd += [f"--mail-user={res.mail_user}", f"--mail-type={res.mail_type or 'FAIL'}"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd += [str(ARRAY_SCRIPT), str(run_dir), str(VENV_PYTHON)]
    return cmd


def _make_run_dir(config: DconnConfig, stage: str, units: list[Unit], extra: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.paths.output_dir / "_runs" / f"{config.name}_{stage}_{stamp}"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(dump_config(config))
    write_manifest(run_dir / "manifest.tsv", units)
    provenance.write_json(run_dir / "submission.json", {"stage": stage, "created": provenance.now(), **extra})
    return run_dir


def submit(
    config_path: Path,
    stage: str,
    *,
    subjects: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Submit ``stage`` ('dconn', 'zscore' or 'both') for every unit that still needs it."""
    config = load_config(config_path)
    git = provenance.require_clean_commit()
    if not VENV_PYTHON.is_file():
        raise PipelineError(f"{VENV_PYTHON} not found; run `uv sync --frozen` in {provenance.REPO_DIR}")

    units = units_from_config(config)
    if subjects:
        wanted = {s.removeprefix("sub-") for s in subjects}
        unknown = wanted - set(config.subjects)
        if unknown:
            raise PipelineError(f"subjects not in config: {sorted(unknown)}")
        units = [u for u in units if u.subject in wanted]

    first = "zscore" if stage == "zscore" else "dconn"
    todo = select_units(config, units, first, overwrite)
    if first == "dconn":
        for unit in todo:
            resolve_inputs(config, unit)  # fail now, not inside a queued job
    else:
        not_ready = [u.label for u in todo if stage_status(config, u, "dconn") != "complete"]
        if not_ready:
            raise PipelineError(f"dconn not complete for: {not_ready}; use --stage both")
    if not todo:
        log.info("nothing to do: every selected unit is already complete")
        return

    log.info("%d unit(s) to run:", len(todo))
    for u in todo:
        log.info("  %s", u.label)

    stages = ["dconn", "zscore"] if stage == "both" else [stage]
    dependency = None
    for st in stages:
        res = getattr(config.slurm, st)
        extra = {"git": git, "overwrite": overwrite, "config_source": str(Path(config_path).resolve())}
        if dry_run:
            cmd = _sbatch_command(config, res, st, Path("<run_dir>"), len(todo), dependency)
            log.info("dry run, would submit: %s", " ".join(cmd))
            dependency = "aftercorr:<job_id>"
            continue
        run_dir = _make_run_dir(config, st, todo, extra)
        cmd = _sbatch_command(config, res, st, run_dir, len(todo), dependency)
        job_id = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split(";")[0]
        provenance.write_json(run_dir / "submission.json", {**extra, "stage": st, "sbatch": cmd, "job_id": job_id})
        log.info("submitted %s stage as job %s (run dir %s)", st, job_id, run_dir)
        # Each z-score task waits only for its own dconn task.
        dependency = f"aftercorr:{job_id}"


def run_array_task(run_dir: Path, index: int) -> None:
    """Entry point inside a Slurm array task."""
    meta = provenance.read_json(run_dir / "submission.json")
    if meta is None:
        raise PipelineError(f"{run_dir}/submission.json missing")
    config = load_config(run_dir / "config.yaml")
    unit = read_manifest_row(run_dir / "manifest.tsv", index)
    log.info("array task %d: %s, stage %s", index, unit.label, meta["stage"])
    runner = run_dconn if meta["stage"] == "dconn" else run_zscore
    runner(config, unit, overwrite=meta["overwrite"], expected_commit=meta["git"]["commit"])
