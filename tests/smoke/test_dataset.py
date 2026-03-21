"""Smoke tests: TofDataset shapes and TomographyDataModule batching.

Skipped on this machine; run with real `inputData` symlink to a `dataset_*` tree.
"""

from pathlib import Path

import pytest

from tomo.data.datamodule import TomographyDataModule
from tomo.data.dataset import TofDataset

pytestmark = pytest.mark.skip(reason="run with MATLAB-generated data under inputData; not available on this computer")


def _default_data_config() -> dict:
    """Config pointing at repo-root inputData (train/validate/test/mat)."""
    root = Path(__file__).resolve().parents[2]
    return {
        "data_root": str(root / "inputData"),
        "normalization_source": "auto",
        "min_sos": 0.1,
        "max_sos": 2.1,
        "min_tof": 0.0,
        "max_tof": 1000.0,
        "min_tof_diff": -1.0,
        "max_tof_diff": 1.0,
        "batch_size": 2,
        "num_workers": 0,
    }


def test_tof_dataset_shapes() -> None:
    """If non-empty, sample has MATLAB-aligned keys + derived tensors."""
    cfg = _default_data_config()
    dataset = TofDataset(
        modes=["train"],
        data_root=cfg["data_root"],
        min_sos=cfg["min_sos"],
        max_sos=cfg["max_sos"],
        min_tof=cfg["min_tof"],
        max_tof=cfg["max_tof"],
        min_tof_diff=cfg["min_tof_diff"],
        max_tof_diff=cfg["max_tof_diff"],
    )
    if len(dataset) == 0:
        pytest.skip("no MAT data found (missing or empty data dir)")
    sample = dataset[0]
    assert sample["tof_tumor_normalized_grid"].shape[0] == 1
    assert sample["tof_diff_normalized"].shape[0] == 1
    assert sample["sos_map_normalized"].shape[0] == 1
    for key in TofDataset.SAMPLE_KEYS:
        assert key in sample, f"missing key {key}"


def test_dataloader_batch() -> None:
    """TomographyDataModule: stacked batch keys include tof_tumor_raw and sos_map_normalized."""
    cfg = _default_data_config()
    dm = TomographyDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    if len(train_loader.dataset) < 2:
        pytest.skip("need at least 2 samples for batch size 2")
    batch = next(iter(train_loader))
    assert batch["tof_tumor_raw"].shape[0] == 2
    assert batch["sos_map_normalized"].shape[0] == 2
    assert batch["x_s"].shape[0] == 2
    assert batch["x_r"].shape[0] == 2
