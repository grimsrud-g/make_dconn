from __future__ import annotations

import pytest

from make_dconn.config import config_from_dict
from make_dconn.inputs import InputError, Unit, resolve_inputs, resolve_outputs


def _make_tree(root, subject="s03", prefix="sub-s03_ses-combined_task-nBack"):
    func = root / f"sub-{subject}" / "ses-combined"
    anat = root / f"sub-{subject}" / "anat"
    func.mkdir(parents=True)
    anat.mkdir(parents=True)
    for name in (
        f"{prefix}_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii",
        f"{prefix}_motion.tsv",
        f"{prefix}_boundaries.tsv",
    ):
        (func / name).touch()
    for hemi in "LR":
        (anat / f"sub-{subject}_acq-X_run-1_hemi-{hemi}_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii").touch()
    return func, anat


def test_resolve_inputs_and_outputs(tmp_path, base_raw):
    base_raw["paths"]["xcpd_dir"] = str(tmp_path)
    config = config_from_dict(base_raw)
    func, anat = _make_tree(tmp_path)
    inputs = resolve_inputs(config, Unit("s03", "nBack"))
    assert inputs.dtseries.parent == func
    assert inputs.surface_left.name.startswith("sub-s03_acq-X_run-1_hemi-L")
    out = resolve_outputs(config, Unit("s03", "nBack"))
    assert out.dconn.name == (
        "sub-s03_ses-combined_task-nBack_space-fsLR_den-91k_desc-fd0p2c5sm2p55"
        "_stat-pearsoncorrelation_boldmap.dconn.nii"
    )
    assert out.zscored.name.endswith("_desc-fd0p2c5sm2p55_stat-zscored_boldmap.dconn.nii")
    assert out.dconn.parent == config.paths.output_dir / "sub-s03" / "ses-combined" / "func"


def test_split_in_name(tmp_path, base_raw):
    base_raw["paths"]["xcpd_dir"] = str(tmp_path)
    config = config_from_dict(base_raw)
    _make_tree(tmp_path, prefix="sub-s03_ses-combined_task-rest_split-2")
    unit = Unit("s03", "rest", 2)
    assert resolve_inputs(config, unit).motion.name == "sub-s03_ses-combined_task-rest_split-2_motion.tsv"
    assert "_task-rest_split-2_space-fsLR" in resolve_outputs(config, unit).dconn.name


def test_missing_input(tmp_path, base_raw):
    base_raw["paths"]["xcpd_dir"] = str(tmp_path)
    config = config_from_dict(base_raw)
    func, _ = _make_tree(tmp_path)
    (func / "sub-s03_ses-combined_task-nBack_motion.tsv").unlink()
    with pytest.raises(InputError, match="missing input"):
        resolve_inputs(config, Unit("s03", "nBack"))


def test_ambiguous_surface(tmp_path, base_raw):
    base_raw["paths"]["xcpd_dir"] = str(tmp_path)
    config = config_from_dict(base_raw)
    _, anat = _make_tree(tmp_path)
    (anat / "sub-s03_acq-Y_run-1_hemi-L_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii").touch()
    with pytest.raises(InputError, match="exactly one"):
        resolve_inputs(config, Unit("s03", "nBack"))


def test_filter_units(base_raw):
    from make_dconn.inputs import units_from_config
    from make_dconn.pipeline import PipelineError
    from make_dconn.submit import filter_units

    base_raw["conditions"] = [{"task": "nBack"}, {"task": "flanker"}]
    config = config_from_dict(base_raw)
    units = units_from_config(config)
    picked = filter_units(config, units, ["sub-s03"], ["nBack"])
    assert picked == [Unit("s03", "nBack")]
    assert len(filter_units(config, units, None, ["flanker"])) == 5
    with pytest.raises(PipelineError, match="tasks not in config"):
        filter_units(config, units, None, ["rest"])
    with pytest.raises(PipelineError, match="subjects not in config"):
        filter_units(config, units, ["s99"], None)
