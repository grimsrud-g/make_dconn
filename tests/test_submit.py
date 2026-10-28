from __future__ import annotations

import os
import subprocess
import sys

import pytest

from make_dconn import provenance
from make_dconn.config import config_from_dict, dump_config
from make_dconn.inputs import Unit, resolve_outputs
from make_dconn.submit import active_outputs, parse_active_tasks, write_manifest


def test_snapshot_is_the_committed_package(tmp_path):
    commit = provenance.git_state()["commit"]
    digest = provenance.snapshot_code(commit, tmp_path)
    pkg = tmp_path / "src" / "make_dconn"
    assert (pkg / "pipeline.py").is_file() and (pkg / "cli.py").is_file()
    assert provenance.tree_digest(tmp_path / "src") == digest
    # __pycache__ written when a job imports the copy must not change the digest
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert provenance.tree_digest(tmp_path / "src") == digest
    # any edit to the copy does
    (pkg / "cli.py").write_text((pkg / "cli.py").read_text() + "\n# edited\n")
    assert provenance.tree_digest(tmp_path / "src") != digest


def test_pythonpath_snapshot_wins_over_editable_install(tmp_path):
    """Array jobs put the snapshot on PYTHONPATH; it must shadow the repo's editable install."""
    provenance.snapshot_code(provenance.git_state()["commit"], tmp_path)
    out = subprocess.run(
        [sys.executable, "-c", "import make_dconn; print(make_dconn.__file__)"],
        env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out.startswith(str(tmp_path / "src"))


def test_parse_active_tasks():
    queued = ["49885825_1", "49885827_6", "49885827_19", "49885831", "4988582_3", "49885827_[7-9]"]
    assert parse_active_tasks(queued, "49885827") == [6, 19]
    assert parse_active_tasks(queued, "49885825") == [1]
    assert parse_active_tasks(queued, "49885831") == []


def test_active_outputs_maps_queued_tasks_to_outputs(tmp_path, base_raw):
    base_raw["paths"]["output_dir"] = str(tmp_path)
    base_raw["conditions"] = [{"task": "rest", "split": 1}, {"task": "rest", "split": 2}]
    config = config_from_dict(base_raw)
    units = [Unit("s03", "rest", 1), Unit("s03", "rest", 2), Unit("s10", "rest", 1)]

    run = tmp_path / "_runs" / "split_half_dconn_x"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(dump_config(config))
    write_manifest(run / "manifest.tsv", units)
    provenance.write_json(run / "submission.json", {"stage": "dconn", "job_id": "123"})
    zrun = tmp_path / "_runs" / "split_half_zscore_x"
    zrun.mkdir()
    (zrun / "config.yaml").write_text(dump_config(config))
    write_manifest(zrun / "manifest.tsv", units)
    provenance.write_json(zrun / "submission.json", {"stage": "zscore", "job_id": "124"})

    active = active_outputs(config, "dconn", queued=["123_2", "123_0", "124_1", "999_0"])
    assert active == {
        resolve_outputs(config, units[0]).dconn: "123_0",
        resolve_outputs(config, units[2]).dconn: "123_2",
    }
    assert active_outputs(config, "zscore", queued=["124_1"]) == {resolve_outputs(config, units[1]).zscored: "124_1"}
    assert active_outputs(config, "dconn", queued=[]) == {}


def test_active_outputs_without_runs(tmp_path, base_raw):
    base_raw["paths"]["output_dir"] = str(tmp_path)
    assert active_outputs(config_from_dict(base_raw), "dconn", queued=["1_0"]) == {}


@pytest.mark.parametrize("stage", ["dconn", "zscore"])
def test_code_state_passthrough(stage):
    from make_dconn.pipeline import _code_state

    state = {"commit": "abc", "dirty": False, "source": "run snapshot"}
    assert _code_state(False, state) is state
