"""Smoke tests: TofDataset shapes and TomographyDataModule batching.

Skipped on this machine; run on AWS instance with data.
"""

from pathlib import Path

import pytest

from tomo.data.datamodule import TomographyDataModule
from tomo.data.dataset import TofDataset

pytestmark = pytest.mark.skip(reason="run on AWS instance with data; cannot run on this computer")


def _default_data_config() -> dict:
    """Config with default paths (may point to missing dirs)."""
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "inputData"
    return {
        "train_path": str(data_dir / "ForLearning"),
        "validation_path": str(data_dir / "ForValidation"),
        "test_path": str(data_dir / "ForTest"),
        "tof_path": str(data_dir / "TimeOfFlightData"),
        "min_sos": 0.1,
        "max_sos": 2.1,
        "min_tof": 0.0,
        "max_tof": 1000.0,
        "batch_size": 2,
        "num_workers": 0,
    }


def test_tof_dataset_shapes() -> None:
    """Instantiate TofDataset; if non-empty, check sample[0] has tof [1,128,128], anatomy [1,128,128], and 8 keys."""
    cfg = _default_data_config()
    dataset = TofDataset(
        modes=["train"],
        train_path=cfg["train_path"],
        validation_path=cfg["validation_path"],
        test_path=cfg["test_path"],
        tof_path=cfg["tof_path"],
        min_sos=cfg["min_sos"],
        max_sos=cfg["max_sos"],
        min_tof=cfg["min_tof"],
        max_tof=cfg["max_tof"],
    )
    if len(dataset) == 0:
        pytest.skip("no MAT data found (missing or empty data dir)")
    sample = dataset[0]
    assert sample["tof"].shape == (1, 128, 128), sample["tof"].shape
    assert sample["anatomy"].shape == (1, 128, 128), sample["anatomy"].shape
    for key in TofDataset.SAMPLE_KEYS:
        assert key in sample, f"missing key {key}"


def test_dataloader_batch() -> None:
    """TomographyDataModule with batch_size=2; one batch has tof/anatomy [2,1,128,128], list keys len 2."""
    cfg = _default_data_config()
    dm = TomographyDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    if len(train_loader.dataset) < 2:
        pytest.skip("need at least 2 samples for batch size 2")
    batch = next(iter(train_loader))
    assert batch["tof"].shape == (2, 1, 128, 128), batch["tof"].shape
    assert batch["anatomy"].shape == (2, 1, 128, 128), batch["anatomy"].shape
    assert len(batch["x_s"]) == 2
    assert len(batch["x_r"]) == 2
    assert len(batch["expanded_tof"]) == 2
    assert len(batch["tof_maps"]) == 2
