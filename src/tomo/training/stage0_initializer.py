"""Stage 0: train initializer only (normalized diff ToF -> 128x128 scaled residual delta_c0).

Input is `tof_diff_normalized` from the dataset (global min/max over `tof_diff_raw`).
Stage 0 trains in scaled space. c_base_phys is the source of truth (config);
c_base_scaled is derived at runtime via tomo.utils.units.to_scaled_sos(c_base_phys, min_sos, max_sos).
The model predicts delta_c0_scaled; the training target is built consistently in scaled space.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tomo.data.datamodule import TomographyDataModule
from tomo.data.normalization_metadata import resolve_data_config
from tomo.initializers.unet_initializer import (
    c0_normalized_from_delta,
    delta_target_from_anatomy,
    normalized_c_base_from_config,
)
from tomo.utils.logging import EpochMetricsLogger, HtmlTrainingReport, setup_training_logging


def _ensure_tof_tumor_4d(tof_map: torch.Tensor) -> torch.Tensor:
    """Ensure ToF map batch is [B, 1, H, W] for the SR model (raw tumor or normalized diff)."""
    if tof_map.dim() == 2:
        return tof_map.unsqueeze(0).unsqueeze(0)
    if tof_map.dim() == 3:
        return tof_map.unsqueeze(1)
    return tof_map


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
        tof_in = _ensure_tof_tumor_4d(batch["tof_diff_normalized"]).to(device)
        sos_norm = batch["sos_map_normalized"].to(device)
        delta_target = delta_target_from_anatomy(sos_norm, normalized_c_base)
        optimizer.zero_grad()
        delta_pred = model(tof_in)
        loss = criterion(delta_pred, delta_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            c0_pred_norm = c0_normalized_from_delta(normalized_c_base, delta_pred)
            total_mse_recon += nn.functional.mse_loss(c0_pred_norm, sos_norm).item()
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
        tof_in = _ensure_tof_tumor_4d(batch["tof_diff_normalized"]).to(device)
        sos_norm = batch["sos_map_normalized"].to(device)
        delta_target = delta_target_from_anatomy(sos_norm, normalized_c_base)
        delta_pred = model(tof_in)
        total_loss += criterion(delta_pred, delta_target).item()
        c0_pred_norm = c0_normalized_from_delta(normalized_c_base, delta_pred)
        total_mse_recon += nn.functional.mse_loss(c0_pred_norm, sos_norm).item()
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
    Run Stage 0 training: train SR initializer on tof_diff_normalized -> delta_c0 (residual).
    Target: delta_target = clamp(anatomy - normalized_c_base, 0, 1). Loss on residual.
    Returns path to best checkpoint, or None if no checkpoint saved.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_config = resolve_data_config(dict(data_config))
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
    enable_html_report = training_config.get("enable_html_report", True)
    log_dir_override = training_config.get("log_dir")

    criterion = _get_criterion(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_root = (
        Path(log_dir_override) / run_id
        if log_dir_override
        else Path(checkpoint_dir) / "runs" / run_id
    )
    train_logger, _ = setup_training_logging(log_root, run_id=run_id)
    report: HtmlTrainingReport | None = None
    if enable_html_report:
        report = HtmlTrainingReport(log_root / f"output___{run_id}.html")
        report.add_text(f"Stage 0  log_dir={log_root}")
        report.add_text(f"terminal log: terminal_{run_id}.txt")

    metrics = EpochMetricsLogger(train_logger, report, total_epochs=epochs)
    train_logger.info(
        "Stage 0 start: epochs=%d lr=%g loss=%s checkpoint_dir=%s",
        epochs,
        lr,
        loss_name,
        checkpoint_dir,
    )

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
            metrics.on_epoch_end(
                epoch + 1,
                train_loss,
                train_mse_recon,
                val_loss=val_loss,
                val_mse_recon=val_mse_recon,
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
            metrics.on_epoch_end(epoch + 1, train_loss, train_mse_recon)

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

    train_logger.info("Stage 0 finished. best_checkpoint=%s", best_path)
    return best_path
