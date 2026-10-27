"""Locate inputs and name outputs for one processing unit (subject x condition)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from make_dconn.config import DconnConfig


class InputError(RuntimeError):
    """A required input is missing or ambiguous."""


@dataclass(frozen=True)
class Unit:
    subject: str
    task: str
    split: int | None = None

    @property
    def label(self) -> str:
        split = f" split-{self.split}" if self.split is not None else ""
        return f"sub-{self.subject} task-{self.task}{split}"


@dataclass(frozen=True)
class UnitInputs:
    dtseries: Path
    motion: Path
    boundaries: Path
    surface_left: Path
    surface_right: Path

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class UnitOutputs:
    dconn: Path
    sidecar: Path
    censoring: Path
    zscored: Path
    zscored_sidecar: Path


def units_from_config(config: DconnConfig) -> list[Unit]:
    return [
        Unit(subject=s, task=c.task, split=c.split)
        for s in config.subjects
        for c in config.conditions
    ]


def _prefix(config: DconnConfig, unit: Unit) -> str:
    split = f"_split-{unit.split}" if unit.split is not None else ""
    return f"sub-{unit.subject}_ses-{config.session}_task-{unit.task}{split}"


def _exactly_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise InputError(
            f"expected exactly one match for {pattern!r} in {directory}, found {len(matches)}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _require(path: Path) -> Path:
    if not path.is_file():
        raise InputError(f"missing input: {path}")
    return path


def resolve_inputs(config: DconnConfig, unit: Unit) -> UnitInputs:
    inp = config.inputs
    sub_dir = config.paths.xcpd_dir / f"sub-{unit.subject}"
    func_dir = sub_dir / f"ses-{config.session}" / inp.func_subdir
    prefix = _prefix(config, unit)
    anat_dir = sub_dir / "anat"
    return UnitInputs(
        dtseries=_require(
            func_dir / f"{prefix}_space-{inp.space}_den-{inp.den}_desc-{inp.desc}_bold.dtseries.nii"
        ),
        motion=_require(func_dir / f"{prefix}_motion.tsv"),
        boundaries=_require(func_dir / f"{prefix}_boundaries.tsv"),
        surface_left=_exactly_one(anat_dir, inp.surface_glob.format(subject=unit.subject, hemi="L")),
        surface_right=_exactly_one(anat_dir, inp.surface_glob.format(subject=unit.subject, hemi="R")),
    )


def _num_tag(value: float) -> str:
    """0.2 -> '0p2', 2.55 -> '2p55', 5 -> '5'."""
    text = f"{value:g}"
    if "e" in text:
        raise ValueError(f"cannot encode {value!r} in a filename tag")
    return text.replace(".", "p")


def processing_tag(config: DconnConfig) -> str:
    """BIDS ``desc`` value encoding every parameter that changes the dconn.

    Example: fd0p2c5sm2p55 = FD <= 0.2, drop stretches < 5 frames, sigma 2.55.
    A ``mad3`` suffix is added when outlier-frame removal is on.
    """
    c = config.censoring
    sigma = config.smoothing.sigma_mm
    tag = f"fd{_num_tag(c.fd_threshold)}c{c.min_contiguous_frames}sm{_num_tag(sigma) if sigma else '0'}"
    if c.outlier_frames.enabled:
        tag += f"mad{_num_tag(c.outlier_frames.n_mads)}"
    return tag


def resolve_outputs(config: DconnConfig, unit: Unit) -> UnitOutputs:
    inp = config.inputs
    out_dir = config.paths.output_dir / f"sub-{unit.subject}" / f"ses-{config.session}" / "func"
    stem = f"{_prefix(config, unit)}_space-{inp.space}_den-{inp.den}_desc-{processing_tag(config)}"
    return UnitOutputs(
        dconn=out_dir / f"{stem}_stat-pearsoncorrelation_boldmap.dconn.nii",
        sidecar=out_dir / f"{stem}_stat-pearsoncorrelation_boldmap.json",
        censoring=out_dir / f"{stem}_censoring.tsv",
        zscored=out_dir / f"{stem}_stat-zscored_boldmap.dconn.nii",
        zscored_sidecar=out_dir / f"{stem}_stat-zscored_boldmap.json",
    )
