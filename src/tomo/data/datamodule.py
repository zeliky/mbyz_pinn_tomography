"""DataModule for training: creates train/val/test TofDatasets and dataloaders from config."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from tomo.data.dataset import TofDataset


def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size fields; keep variable-size fields as lists."""
    stacked = {}
    list_fields = ("x_s", "x_r", "expanded_tof", "tof_maps")
    stack_fields = ("anatomy", "tof", "raw_sos", "raw_tof")
    for key in stack_fields:
        stacked[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in list_fields:
        stacked[key] = [b[key] for b in batch]
    return stacked


class TomographyDataModule:
    """Creates train/validation/test TofDatasets and dataloaders from a config dict.

    Config should provide at least:
      - train_path, validation_path, test_path, tof_path
      - min_sos, max_sos, min_tof, max_tof
      - batch_size, num_workers
    Optional: sources_amount, receivers_amount, anatomy_width, anatomy_height (defaults 32, 32, 128, 128).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)
        self._train_dataset: TofDataset | None = None
        self._val_dataset: TofDataset | None = None
        self._test_dataset: TofDataset | None = None

    def _dataset_kwargs(self) -> dict[str, Any]:
        c = self._config
        return {
            "train_path": c["train_path"],
            "validation_path": c["validation_path"],
            "test_path": c["test_path"],
            "tof_path": c["tof_path"],
            "min_sos": c["min_sos"],
            "max_sos": c["max_sos"],
            "min_tof": c["min_tof"],
            "max_tof": c["max_tof"],
            "sources_amount": c.get("sources_amount", 32),
            "receivers_amount": c.get("receivers_amount", 32),
            "anatomy_width": c.get("anatomy_width", 128),
            "anatomy_height": c.get("anatomy_height", 128),
        }

    def setup(self) -> None:
        """Build train, validation, and test TofDatasets from config."""
        kwargs = self._dataset_kwargs()
        self._train_dataset = TofDataset(modes=["train"], **kwargs)
        self._val_dataset = TofDataset(modes=["validation"], **kwargs)
        self._test_dataset = TofDataset(modes=["test"], **kwargs)

    def _ensure_setup(self) -> None:
        if self._train_dataset is None:
            self.setup()

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        """DataLoader for training (shuffle=True)."""
        self._ensure_setup()
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self._config.get("batch_size", 4),
            shuffle=True,
            num_workers=self._config.get("num_workers", 0),
            collate_fn=_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        """DataLoader for validation (shuffle=False)."""
        self._ensure_setup()
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self._config.get("batch_size", 4),
            shuffle=False,
            num_workers=self._config.get("num_workers", 0),
            collate_fn=_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        """DataLoader for test (shuffle=False)."""
        self._ensure_setup()
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self._config.get("batch_size", 4),
            shuffle=False,
            num_workers=self._config.get("num_workers", 0),
            collate_fn=_collate_fn,
        )
