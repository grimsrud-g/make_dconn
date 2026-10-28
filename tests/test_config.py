from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from make_dconn.config import ConfigError, config_from_dict, dump_config, load_config, load_raw
from make_dconn.inputs import processing_tag, units_from_config
from make_dconn.provenance import params_hash

from conftest import CONFIG_DIR


@pytest.mark.parametrize(
    "name, n_conditions",
    [("task_concat", 1), ("rest", 1), ("task_individual", 8), ("split_half", 4)],
)
def test_repo_configs_load(monkeypatch, name, n_conditions):
    monkeypatch.setenv("GROUP_SCRATCH", "/scratch/groups/test")
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    assert config.name == name
    assert len(config.conditions) == n_conditions
    assert len(units_from_config(config)) == 5 * n_conditions
    assert str(config.paths.output_dir).startswith("/scratch/groups/test/")
    assert processing_tag(config) == "fd0p2c5sm2p55fill"


def test_unset_env_var_is_an_error(monkeypatch):
    monkeypatch.delenv("GROUP_SCRATCH", raising=False)
    with pytest.raises(ConfigError, match="GROUP_SCRATCH"):
        load_config(CONFIG_DIR / "rest.yaml")


def test_unknown_key_rejected(base_raw):
    base_raw["censoring"]["fd_treshold"] = 0.3
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict(base_raw)


def test_missing_key_rejected(base_raw):
    del base_raw["smoothing"]["sigma_mm"]
    with pytest.raises(ConfigError, match="sigma_mm: required"):
        config_from_dict(base_raw)


def test_wrong_type_rejected(base_raw):
    base_raw["censoring"]["min_contiguous_frames"] = 5.0
    with pytest.raises(ConfigError, match="expected an integer"):
        config_from_dict(base_raw)
    base_raw["censoring"]["min_contiguous_frames"] = 5
    base_raw["keep_intermediates"] = "no"
    with pytest.raises(ConfigError, match="true/false"):
        config_from_dict(base_raw)


def test_range_checks(base_raw):
    base_raw["censoring"]["fd_threshold"] = 0
    with pytest.raises(ConfigError, match="fd_threshold"):
        config_from_dict(base_raw)


def test_duplicate_conditions_rejected(base_raw):
    base_raw["conditions"] = [{"task": "nBack"}, {"task": "nBack"}]
    with pytest.raises(ConfigError, match="duplicate conditions"):
        config_from_dict(base_raw)


def test_sub_prefix_stripped(base_raw):
    base_raw["subjects"] = ["sub-s03", "s10"]
    assert config_from_dict(base_raw).subjects == ["s03", "s10"]


def test_null_smoothing_tag(base_raw):
    base_raw["smoothing"]["sigma_mm"] = None
    base_raw["smoothing"]["fill_missing"] = False
    assert processing_tag(config_from_dict(base_raw)) == "fd0p2c5sm0"


def test_fill_requires_smoothing(base_raw):
    base_raw["smoothing"]["sigma_mm"] = None
    with pytest.raises(ConfigError, match="fill_missing needs smoothing"):
        config_from_dict(base_raw)


def test_outlier_tag(base_raw):
    base_raw["censoring"]["outlier_frames"]["enabled"] = True
    assert processing_tag(config_from_dict(base_raw)) == "fd0p2c5sm2p55fillmad3"
    base_raw["smoothing"]["fill_missing"] = False
    assert processing_tag(config_from_dict(base_raw)) == "fd0p2c5sm2p55mad3"


def test_extends_deep_merges(tmp_path):
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump({"a": {"x": 1, "y": 2}, "l": [1, 2]}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"extends": "parent.yaml", "a": {"y": 3}, "l": [9]}))
    assert load_raw(tmp_path / "child.yaml") == {"a": {"x": 1, "y": 3}, "l": [9]}


def test_circular_extends(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_raw(tmp_path / "a.yaml")


def test_dump_round_trips(base_raw):
    config = config_from_dict(base_raw)
    assert config_from_dict(yaml.safe_load(dump_config(config))) == config


def test_params_hash_tracks_content_not_resources(base_raw):
    h = params_hash(config_from_dict(base_raw))
    other = copy.deepcopy(base_raw)
    other["slurm"]["dconn"]["mem"] = "256G"
    other["subjects"] = ["s03"]
    assert params_hash(config_from_dict(other)) == h
    other["smoothing"]["merged_volume"] = False
    assert params_hash(config_from_dict(other)) != h
    filled = copy.deepcopy(base_raw)
    filled["smoothing"]["fill_missing"] = False
    assert params_hash(config_from_dict(filled)) != h


def test_git_state_works_with_system_git(monkeypatch):
    """Jobs run after `ml purge`, where only the old system git is on PATH."""
    from make_dconn import provenance

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    state = provenance.git_state()
    assert len(state["commit"]) == 40


def test_smooth_command_flags(tmp_path):
    from make_dconn.workbench import smooth_command

    base = smooth_command(Path("wb"), tmp_path / "i", tmp_path / "o", 2.55, Path("L"), Path("R"), True)
    assert base[-1] == "-merged-volume" and "-fix-zeros-surface" not in base
    fixed = smooth_command(Path("wb"), tmp_path / "i", tmp_path / "o", 2.55, Path("L"), Path("R"), True, fix_zeros_surface=True)
    assert fixed == base + ["-fix-zeros-surface"]
