"""Build Stage2Sample list from Stage 1 outputs and TomographyDataModule validation set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from tomo.data.datamodule import TomographyDataModule
from tomo.stage2.stage2_env import Stage2Sample


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_grid_coords(
    x_s: np.ndarray,
    x_r: np.ndarray,
    grid_shape: tuple[int, int],
    coord_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = grid_shape
    if coord_range is None:
        return x_s, x_r
    lo, hi = coord_range
    scale = (W - 1) / (hi - lo) if hi != lo else 1.0
    x_s = (x_s - lo) * scale
    x_r = (x_r - lo) * scale
    x_s = np.clip(x_s, 0, W - 1)
    x_r = np.clip(x_r, 0, W - 1)
    return x_s, x_r


def load_stage2_samples(
    stage1_dir: Path,
    dm: TomographyDataModule,
    c_base: float = 1.5,
    c_min: float = 1.45,
    c_max: float = 1.8,
    scale_factor: float = 1.0,
    coord_range: tuple[float, float] | None = None,
    max_samples: int | None = None,
) -> list[Stage2Sample]:
    """Load Stage 1 outputs and build Stage2Sample list (uses val split / dataset indexing)."""
    loader: DataLoader[dict[str, Any]] = dm.val_dataloader()
    dataset = loader.dataset
    samples: list[Stage2Sample] = []

    for sample_idx in range(len(dataset)):
        if max_samples is not None and sample_idx >= max_samples:
            break

        s1_dir = stage1_dir / f"sample_{sample_idx}"
        if not s1_dir.exists():
            continue

        best_c_map = np.load(s1_dir / "best_c_map.npy")
        tof_pred = np.load(s1_dir / "tof_pred.npy")

        batch = dataset[sample_idx]
        raw_tof = _to_numpy(batch["tof_tumor_raw"]).squeeze()
        x_s = _to_numpy(batch["x_s"]).squeeze()
        x_r = _to_numpy(batch["x_r"]).squeeze()

        H, W = best_c_map.shape
        x_s, x_r = _ensure_grid_coords(x_s, x_r, (H, W), coord_range)

        if x_s.ndim == 1:
            x_s = x_s.reshape(1, -1)
        if x_r.ndim == 1:
            x_r = x_r.reshape(1, -1)

        sample = Stage2Sample(
            c_stage1=best_c_map.astype(np.float64),
            raw_tof=raw_tof.astype(np.float64),
            x_s=x_s.astype(np.float64),
            x_r=x_r.astype(np.float64),
            tof_pred_stage1=tof_pred.astype(np.float64),
            c_base=c_base,
            c_min=c_min,
            c_max=c_max,
            scale_factor=scale_factor,
        )
        samples.append(sample)

    return samples
