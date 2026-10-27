from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def base_raw() -> dict:
    """The repo's base.yaml as a dict, with a condition added, independent of $GROUP_SCRATCH."""
    with open(CONFIG_DIR / "base.yaml") as f:
        raw = yaml.safe_load(f)
    raw["paths"]["output_dir"] = "/tmp/make_dconn_test_out"
    raw["name"] = "test"
    raw["conditions"] = [{"task": "nBack"}]
    return raw
