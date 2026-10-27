"""Command-line interface: ``make-dconn {status,run,submit,run-array}``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from make_dconn.config import load_config
from make_dconn.inputs import InputError, Unit, resolve_inputs, units_from_config
from make_dconn.pipeline import run_dconn, run_zscore, stage_status


def _status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"{'unit':<45} {'inputs':<8} {'dconn':<9} {'zscore':<9}")
    for unit in units_from_config(config):
        try:
            resolve_inputs(config, unit)
            inputs = "ok"
        except InputError:
            inputs = "MISSING"
        print(
            f"{unit.label:<45} {inputs:<8} "
            f"{stage_status(config, unit, 'dconn'):<9} {stage_status(config, unit, 'zscore'):<9}"
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    unit = Unit(subject=args.subject.removeprefix("sub-"), task=args.task, split=args.split)
    runner = run_dconn if args.stage == "dconn" else run_zscore
    runner(config, unit, overwrite=args.overwrite, allow_dirty=args.allow_dirty)
    return 0


def _submit(args: argparse.Namespace) -> int:
    from make_dconn.submit import submit

    submit(args.config, args.stage, subjects=args.subjects, overwrite=args.overwrite, dry_run=args.dry_run)
    return 0


def _run_array(args: argparse.Namespace) -> int:
    from make_dconn.submit import run_array_task

    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    run_array_task(args.run_dir, index)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="make-dconn", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="show input availability and stage status for every unit in a config")
    s.add_argument("config", type=Path)
    s.set_defaults(func=_status)

    r = sub.add_parser("run", help="run one unit in the current shell/allocation")
    r.add_argument("config", type=Path)
    r.add_argument("--subject", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--split", type=int, default=None)
    r.add_argument("--stage", choices=["dconn", "zscore"], default="dconn")
    r.add_argument("--overwrite", action="store_true", help="replace complete or stale outputs")
    r.add_argument("--allow-dirty", action="store_true", help="allow uncommitted code (test runs only)")
    r.set_defaults(func=_run)

    b = sub.add_parser("submit", help="submit every unit that still needs a stage as a Slurm array")
    b.add_argument("config", type=Path)
    b.add_argument("--stage", choices=["dconn", "zscore", "both"], required=True)
    b.add_argument("--subjects", nargs="+", default=None, help="restrict to these subjects")
    b.add_argument("--overwrite", action="store_true", help="redo complete or stale outputs")
    b.add_argument("--dry-run", action="store_true", help="show what would be submitted")
    b.set_defaults(func=_submit)

    a = sub.add_parser("run-array", help="(internal) run one array task of a submitted run")
    a.add_argument("run_dir", type=Path)
    a.add_argument("--index", type=int, default=None)
    a.set_defaults(func=_run_array)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
