"""Unit tests: normalization.json loading and resolve_data_config."""

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tomo.data.datamodule import TomographyDataModule
from tomo.data.normalization_metadata import (
    load_normalization_bounds,
    resolve_data_config,
)


def test_load_normalization_bounds_roundtrip(tmp_path: Path) -> None:
    manifest = {
        "version": 1,
        "min_sos": 1.4,
        "max_sos": 1.65,
        "min_tof": 0.01,
        "max_tof": 99.0,
        "min_tof_diff": -0.5,
        "max_tof_diff": 0.5,
    }
    p = tmp_path / "normalization.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    b = load_normalization_bounds(str(tmp_path))
    assert b["min_sos"] == 1.4
    assert b["max_tof"] == 99.0
    assert b["min_tof_diff"] == -0.5
    assert b["max_tof_diff"] == 0.5


def test_resolve_auto_uses_manifest(tmp_path: Path) -> None:
    (tmp_path / "normalization.json").write_text(
        json.dumps(
            {
                "min_sos": 1.0,
                "max_sos": 2.0,
                "min_tof": 0.0,
                "max_tof": 50.0,
                "min_tof_diff": -1.0,
                "max_tof_diff": 2.0,
            }
        ),
        encoding="utf-8",
    )
    c = resolve_data_config(
        {
            "data_root": str(tmp_path),
            "normalization_source": "auto",
            "min_sos": 0.1,
            "max_sos": 2.1,
            "min_tof": 0.0,
            "max_tof": 1000.0,
            "min_tof_diff": 0.0,
            "max_tof_diff": 1.0,
        }
    )
    assert c["min_sos"] == 1.0
    assert c["max_tof"] == 50.0
    assert c["min_tof_diff"] == -1.0
    assert c["max_tof_diff"] == 2.0


def test_resolve_config_ignores_manifest(tmp_path: Path) -> None:
    (tmp_path / "normalization.json").write_text(
        json.dumps(
            {
                "min_sos": 9.0,
                "max_sos": 10.0,
                "min_tof": 0.0,
                "max_tof": 1.0,
                "min_tof_diff": -0.1,
                "max_tof_diff": 0.1,
            }
        ),
        encoding="utf-8",
    )
    c = resolve_data_config(
        {
            "data_root": str(tmp_path),
            "normalization_source": "config",
            "min_sos": 0.2,
            "max_sos": 2.2,
            "min_tof": 1.0,
            "max_tof": 2.0,
            "min_tof_diff": 0.0,
            "max_tof_diff": 1.0,
        }
    )
    assert c["min_sos"] == 0.2
    assert c["max_tof"] == 2.0


def test_resolve_dataset_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_data_config(
            {
                "data_root": str(tmp_path),
                "normalization_source": "dataset",
                "min_sos": 0.1,
                "max_sos": 2.1,
                "min_tof": 0.0,
                "max_tof": 1000.0,
                "min_tof_diff": -1.0,
                "max_tof_diff": 1.0,
            }
        )


def test_tomography_datamodule_dataset_mode_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TomographyDataModule(
            {
                "data_root": str(tmp_path),
                "normalization_source": "dataset",
                "min_sos": 0.1,
                "max_sos": 2.1,
                "min_tof": 0.0,
                "max_tof": 1000.0,
                "min_tof_diff": -1.0,
                "max_tof_diff": 1.0,
                "batch_size": 1,
                "num_workers": 0,
            }
        )
