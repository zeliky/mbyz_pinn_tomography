"""Stage 0: train initializer only (raw_tof -> 128x128 scaled residual delta_c0).

Stage 0 trains in scaled space. c_base_phys is the source of truth (config);
c_base_scaled is derived at runtime via tomo.utils.units.to_scaled_sos(c_base_phys, min_sos, max_sos).
The model predicts delta_c0_scaled; the training target is built consistently in scaled space.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tomo.data.datamodule import TomographyDataModule
from tomo.initializers.unet_initializer import (
    c0_normalized_from_delta,
    delta_target_from_anatomy,
    normalized_c_base_from_config,
)


def _ensure_raw_tof_4d(raw_tof: torch.Tensor) -> torch.Tensor:
    """Ensure raw_tof is [B, 1, 32, 32] for the SR model."""
    if raw_tof.dim() == 3:
        return raw_tof.unsqueeze(1)
    return raw_tof


def _get_criterion(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {loss_name}. Use 'l1' or 'smooth_l1'.")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalized_c_base: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mse_recon = 0.0
    n = 0
    for batch in loader:
        raw_tof = _ensure_raw_tof_4d(batch["raw_tof"]).to(device)
        anatomy = batch["anatomy"].to(device)
        delta_target = delta_target_from_anatomy(anatomy, normalized_c_base)
        optimizer.zero_grad()
        delta_pred = model(raw_tof)
        loss = criterion(delta_pred, delta_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            c0_pred_norm = c0_normalized_from_delta(normalized_c_base, delta_pred)
            total_mse_recon += nn.functional.mse_loss(c0_pred_norm, anatomy).item()
        n += 1
    return total_loss / max(n, 1), total_mse_recon / max(n, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    normalized_c_base: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mse_recon = 0.0
    n = 0
    for batch in loader:
        raw_tof = _ensure_raw_tof_4d(batch["raw_tof"]).to(device)
        anatomy = batch["anatomy"].to(device)
        delta_target = delta_target_from_anatomy(anatomy, normalized_c_base)
        delta_pred = model(raw_tof)
        total_loss += criterion(delta_pred, delta_target).item()
        c0_pred_norm = c0_normalized_from_delta(normalized_c_base, delta_pred)
        total_mse_recon += nn.functional.mse_loss(c0_pred_norm, anatomy).item()
        n += 1
    return total_loss / max(n, 1), total_mse_recon / max(n, 1)


def run_stage0(
    model: nn.Module,
    data_config: dict[str, Any],
    training_config: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> str | None:
    """
    Run Stage 0 training: train SR initializer on raw_tof -> delta_c0 (residual).
    Target: delta_target = clamp(anatomy - normalized_c_base, 0, 1). Loss on residual.
    Returns path to best checkpoint, or None if no checkpoint saved.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    min_sos = data_config["min_sos"]
    max_sos = data_config["max_sos"]
    c_base_phys = data_config.get("c_base_phys", data_config.get("c_base", 1.5))
    normalized_c_base = normalized_c_base_from_config(min_sos, max_sos, c_base_phys)

    dm = TomographyDataModule(data_config)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if len(train_loader.dataset) == 0:
        raise ValueError("Stage 0: train dataset is empty. Check data paths.")

    epochs = training_config.get("epochs", 50)
    lr = training_config.get("lr", 0.001)
    loss_name = training_config.get("loss", "smooth_l1")
    checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints/stage0")
    save_best = training_config.get("save_best", True)
    save_last = training_config.get("save_last", True)
    val_every = training_config.get("val_every", 1)

    criterion = _get_criterion(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path = None
    last_val_loss: float | None = None
    last_val_mse: float | None = None

    for epoch in range(epochs):
        train_loss, train_mse_recon = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            normalized_c_base,
        )
        if (epoch + 1) % val_every == 0 or epoch == 0:
            val_loss, val_mse_recon = _validate(
                model, val_loader, criterion, device, normalized_c_base
            )
            last_val_loss, last_val_mse = val_loss, val_mse_recon
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  train_mse_recon={train_mse_recon:.6f}  "
                f"val_loss={val_loss:.4f}  val_mse_recon={val_mse_recon:.6f}"
            )
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_mse_recon": val_mse_recon,
                    },
                    best_path,
                )
        else:
            print(
                f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  train_mse_recon={train_mse_recon:.6f}"
            )

    if save_last:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epochs,
                "val_loss": last_val_loss,
                "val_mse_recon": last_val_mse,
            },
            last_path,
        )

    return best_path
