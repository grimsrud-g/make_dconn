"""Freeze a run (config + manifest + code commit) and submit it as a Slurm job array.

Layout of a run directory, ``<output_dir>/_runs/<name>_<stage>_<timestamp>/``:

    config.yaml      fully resolved config the jobs read (later edits to the
                     source config cannot change a submitted run)
    manifest.tsv     one row per array index: subject, task, split
    code/src/        the package as committed at submission (``git archive``);
                     jobs import this copy, so later edits or commits in the
                     repo cannot change what a queued job runs
    submission.json  code commit, snapshot digest, sbatch commands, job ids
    logs/            Slurm stdout/stderr, one file per array task
"""

from __future__ import annotations

import csv
import logging
import subprocess
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from make_dconn import provenance
from make_dconn.config import DconnConfig, SlurmResources, dump_config, load_config
from make_dconn.inputs import Unit, resolve_inputs, resolve_outputs, units_from_config
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


def filter_units(
    config: DconnConfig, units: list[Unit], subjects: list[str] | None, tasks: list[str] | None
) -> list[Unit]:
    """Restrict to the given subjects and/or tasks; names not in the config are an error."""
    if subjects:
        wanted = {s.removeprefix("sub-") for s in subjects}
        unknown = wanted - set(config.subjects)
        if unknown:
            raise PipelineError(f"subjects not in config: {sorted(unknown)}")
        units = [u for u in units if u.subject in wanted]
    if tasks:
        wanted = set(tasks)
        unknown = wanted - {c.task for c in config.conditions}
        if unknown:
            raise PipelineError(f"tasks not in config: {sorted(unknown)}")
        units = [u for u in units if u.task in wanted]
    return units


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
    """Create the run directory; adds ``code_snapshot`` to ``extra``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.paths.output_dir / "_runs" / f"{config.name}_{stage}_{stamp}"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(dump_config(config))
    write_manifest(run_dir / "manifest.tsv", units)
    digest = provenance.snapshot_code(extra["git"]["commit"], run_dir / "code")
    extra["code_snapshot"] = {"path": "code/src", "sha256": digest}
    provenance.write_json(run_dir / "submission.json", {"stage": stage, "created": provenance.now(), **extra})
    return run_dir


def parse_active_tasks(squeue_ids: Iterable[str], job_id: str) -> list[int]:
    """Array indices of ``job_id`` in ``squeue -r -o %i`` output (one id per array task)."""
    prefix = f"{job_id}_"
    return sorted(int(i[len(prefix):]) for i in squeue_ids if i.startswith(prefix) and i[len(prefix):].isdigit())


def _my_queued_task_ids() -> list[str]:
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-r", "-o", "%i"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        # Without the queue we cannot rule out duplicates, so do not submit.
        raise PipelineError(f"cannot read the Slurm queue to check for duplicates: {err}") from err
    return out.stdout.split()


def active_outputs(config: DconnConfig, stage: str, queued: list[str] | None = None) -> dict[Path, str]:
    """Outputs of ``stage`` that a queued or running task from an earlier run will write.

    Maps output path -> Slurm task id, so ``submit`` never queues a second
    job for the same output.
    """
    queued = _my_queued_task_ids() if queued is None else queued
    active: dict[Path, str] = {}
    for meta_path in sorted((config.paths.output_dir / "_runs").glob("*/submission.json")):
        meta = provenance.read_json(meta_path)
        if not meta or meta.get("stage") != stage or "job_id" not in meta:
            continue
        indices = parse_active_tasks(queued, meta["job_id"])
        if not indices:
            continue
        run_config = load_config(meta_path.parent / "config.yaml")
        for index in indices:
            out = resolve_outputs(run_config, read_manifest_row(meta_path.parent / "manifest.tsv", index))
            active[out.dconn if stage == "dconn" else out.zscored] = f"{meta['job_id']}_{index}"
    return active


def submit(
    config_path: Path,
    stage: str,
    *,
    subjects: list[str] | None = None,
    tasks: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Submit ``stage`` ('dconn', 'zscore' or 'both') for every unit that still needs it."""
    config = load_config(config_path)
    git = provenance.require_clean_commit()
    if not VENV_PYTHON.is_file():
        raise PipelineError(f"{VENV_PYTHON} not found; run `uv sync --frozen --offline` in {provenance.REPO_DIR}")

    units = filter_units(config, units_from_config(config), subjects, tasks)

    first = "zscore" if stage == "zscore" else "dconn"
    todo = select_units(config, units, first, overwrite)
    if first == "dconn":
        for unit in todo:
            resolve_inputs(config, unit)  # fail now, not inside a queued job
    else:
        not_ready = [u.label for u in todo if stage_status(config, u, "dconn") != "complete"]
        if not_ready:
            raise PipelineError(f"dconn not complete for: {not_ready}; use --stage both")
    active = active_outputs(config, first)
    if active:
        kept = []
        for unit in todo:
            out = resolve_outputs(config, unit)
            task = active.get(out.dconn if first == "dconn" else out.zscored)
            if task:
                log.info("skipping %s: already queued or running as %s", unit.label, task)
            else:
                kept.append(unit)
        todo = kept
    if not todo:
        log.info("nothing to do: every selected unit is complete, queued or running")
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
    snapshot = run_dir / "code" / "src"
    import make_dconn

    running = Path(make_dconn.__file__).resolve().parent
    if running != (snapshot / "make_dconn").resolve():
        raise PipelineError(f"job imported make_dconn from {running}, not the run's snapshot {snapshot}")
    digest = provenance.tree_digest(snapshot)
    if digest != meta["code_snapshot"]["sha256"]:
        raise PipelineError(f"code snapshot in {snapshot} was modified after submission")
    code_state = {
        "commit": meta["git"]["commit"],
        "dirty": False,
        "source": "run snapshot (git archive of the commit at submission)",
        "snapshot_sha256": digest,
    }
    config = load_config(run_dir / "config.yaml")
    unit = read_manifest_row(run_dir / "manifest.tsv", index)
    log.info("array task %d: %s, stage %s, code %s", index, unit.label, meta["stage"], code_state["commit"][:7])
    runner = run_dconn if meta["stage"] == "dconn" else run_zscore
    runner(config, unit, overwrite=meta["overwrite"], code_state=code_state)
