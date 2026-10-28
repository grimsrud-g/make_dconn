"""Load and validate YAML job configs.

A config may name a parent with ``extends: <path>`` (resolved relative to the
config file). Parents are deep-merged: mappings merge key by key, every other
value (including lists) is replaced by the child. Unknown keys, missing keys
and wrong types are errors, so a typo can never silently fall back to a
default.
"""

from __future__ import annotations

import dataclasses
import os
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_UNRESOLVED_VAR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


class ConfigError(ValueError):
    pass


def _expand_path(value: str, where: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    leftover = _UNRESOLVED_VAR.search(expanded)
    if leftover:
        raise ConfigError(f"{where}: environment variable {leftover.group(0)} in {value!r} is not set")
    return Path(expanded)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class Paths:
    xcpd_dir: Path
    output_dir: Path
    # Where intermediates (censored/smoothed dtseries, dconn before copy-out)
    # are written. None = $L_SCRATCH_JOB, else $L_SCRATCH, else $TMPDIR.
    work_dir: Path | None = None


@dataclass(frozen=True)
class Inputs:
    """BIDS entities/patterns used to locate files inside ``paths.xcpd_dir``.

    ``func_subdir`` is the folder under ``sub-<label>/ses-<session>/`` holding
    the dtseries (empty for the XCP-D ses-combined layout). ``surface_glob``
    is matched inside ``sub-<label>/anat/``; ``{subject}`` and ``{hemi}``
    (L/R) are filled in and exactly one file must match per hemisphere.
    """

    func_subdir: str
    space: str
    den: str
    desc: str
    surface_glob: str


@dataclass(frozen=True)
class OutlierFrames:
    """Optional removal of frames whose across-grayordinate SD is an outlier.

    Follows Hermosillo et al. 2024: median-absolute-deviation rule applied to
    the frames that survived FD censoring (MATLAB ``isoutlier`` default:
    more than ``n_mads`` scaled MADs from the median).
    """

    enabled: bool
    n_mads: float = 3.0

    def __post_init__(self) -> None:
        _check(self.n_mads > 0, "censoring.outlier_frames.n_mads must be > 0")


@dataclass(frozen=True)
class Censoring:
    fd_column: str
    # Frames with FD <= fd_threshold are kept.
    fd_threshold: float
    # Surviving stretches shorter than this (counted within each run) are dropped.
    min_contiguous_frames: int
    outlier_frames: OutlierFrames

    def __post_init__(self) -> None:
        _check(self.fd_threshold > 0, "censoring.fd_threshold must be > 0")
        _check(self.min_contiguous_frames >= 1, "censoring.min_contiguous_frames must be >= 1")


@dataclass(frozen=True)
class Smoothing:
    # Gaussian sigma in mm for both surface and volume; null disables smoothing.
    # Required on purpose: every config must state whether it smooths.
    sigma_mm: float | None
    merged_volume: bool
    # Fill surface vertices that have no data in any frame (XCP-D leaves some
    # NaN): set to 0 and smooth with -fix-zeros-surface, keeping Workbench's
    # extrapolated values, so the dconn has no NaN rows. false = leave NaN.
    fill_missing: bool
    # Missing vertices too far inside a gap for the smoothing kernel to reach
    # stay 0; they are then filled with -cifti-dilate over this distance.
    fill_dilate_mm: float

    def __post_init__(self) -> None:
        _check(self.sigma_mm is None or self.sigma_mm > 0, "smoothing.sigma_mm must be > 0 or null")
        _check(self.fill_dilate_mm > 0, "smoothing.fill_dilate_mm must be > 0")
        _check(
            not (self.fill_missing and self.sigma_mm is None),
            "smoothing.fill_missing needs smoothing (sigma_mm) to extrapolate values",
        )


@dataclass(frozen=True)
class Correlation:
    mem_limit_gb: float

    def __post_init__(self) -> None:
        _check(self.mem_limit_gb > 0, "correlation.mem_limit_gb must be > 0")


@dataclass(frozen=True)
class SlurmResources:
    partition: str
    time: str
    cpus_per_task: int
    mem: str
    mail_user: str | None = None
    mail_type: str | None = None

    def __post_init__(self) -> None:
        _check(self.cpus_per_task >= 1, "slurm cpus_per_task must be >= 1")


@dataclass(frozen=True)
class Slurm:
    modules: list[str]
    dconn: SlurmResources
    zscore: SlurmResources


@dataclass(frozen=True)
class Condition:
    task: str
    split: int | None = None


@dataclass(frozen=True)
class DconnConfig:
    name: str
    paths: Paths
    inputs: Inputs
    session: str
    subjects: list[str]
    conditions: list[Condition]
    censoring: Censoring
    smoothing: Smoothing
    correlation: Correlation
    slurm: Slurm
    keep_intermediates: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", [s.removeprefix("sub-") for s in self.subjects])
        _check(len(self.subjects) > 0, "subjects must not be empty")
        _check(len(self.conditions) > 0, "conditions must not be empty")
        _check(len(self.subjects) == len(set(self.subjects)), f"duplicate subjects: {self.subjects}")
        keys = [(c.task, c.split) for c in self.conditions]
        _check(len(keys) == len(set(keys)), f"duplicate conditions: {keys}")


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(inner, value, where)
    if origin is list:
        _check(isinstance(value, list), f"{where}: expected a list, got {value!r}")
        return [_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if tp is Path:
        _check(isinstance(value, str), f"{where}: expected a path string, got {value!r}")
        return _expand_path(value, where)
    if tp is bool:
        _check(isinstance(value, bool), f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        _check(isinstance(value, int) and not isinstance(value, bool), f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        _check(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{where}: expected a number, got {value!r}",
        )
        return float(value)
    if tp is str:
        _check(isinstance(value, str), f"{where}: expected a string, got {value!r}")
        return value
    raise TypeError(f"unsupported config field type {tp!r} at {where}")


def _build(cls: type, data: Any, where: str) -> Any:
    _check(isinstance(data, dict), f"{where or 'config'}: expected a mapping, got {data!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - set(fields))
    _check(not unknown, f"{where or 'config'}: unknown key(s) {unknown}")
    kwargs = {}
    for name, f in fields.items():
        key = f"{where}.{name}" if where else name
        if name not in data:
            has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            _check(has_default, f"{key}: required key is missing")
            continue
        kwargs[name] = _coerce(hints[name], data[name], key)
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Read a config and its ``extends`` chain into one merged mapping."""
    path = Path(path).resolve()
    _check(path not in _seen, f"circular 'extends' chain: {[str(p) for p in (*_seen, path)]}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    _check(isinstance(data, dict), f"{path} must contain a YAML mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _deep_merge(load_raw(path.parent / parent, (*_seen, path)), data)


def config_from_dict(data: dict) -> DconnConfig:
    return _build(DconnConfig, data, "")


def load_config(path: Path) -> DconnConfig:
    return config_from_dict(load_raw(path))


def to_dict(obj: Any) -> Any:
    """Plain YAML/JSON-safe structure (paths as strings)."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_config(config: DconnConfig) -> str:
    """Fully resolved config as YAML (no ``extends``), for freezing alongside a run."""
    return yaml.safe_dump(to_dict(config), sort_keys=False)
