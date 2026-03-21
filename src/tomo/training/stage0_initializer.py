"""Stage 0: train initializer (tof_diff_normalized -> scaled residual delta_c0).

Supports SR grid [B,1,H_tof,W_tof] or measurement grid [B,1,S,R] with backprojection+U-Net.
Input is `tof_diff_normalized` from the dataset. Target: delta above normalized_c_base.
Spatial sizes come from config (tof_grid_height/width, anatomy_height/width).
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _ensure_tof_tumor_4d(tof_map: torch.Tensor) -> torch.Tensor:
    """Ensure ToF map batch is [B, 1, H, W] for the SR model (raw tumor or normalized diff)."""
    if tof_map.dim() == 2:
        return tof_map.unsqueeze(0).unsqueeze(0)
    if tof_map.dim() == 3:
        return tof_map.unsqueeze(1)
    return tof_map


def ensure_tof_measurement_grid(tof_map: torch.Tensor) -> torch.Tensor:
    """Normalize diff ToF to [B, 1, S, R] (no spatial resize)."""
    if tof_map.dim() == 2:
        return tof_map.unsqueeze(0).unsqueeze(0)
    if tof_map.dim() == 3:
        return tof_map.unsqueeze(1)
    if tof_map.dim() == 4:
        if tof_map.shape[1] != 1:
            raise ValueError(f"Expected [B,1,S,R] ToF, got shape {tuple(tof_map.shape)}")
        return tof_map
    raise ValueError(f"ToF tensor must be 2D–4D, got dim {tof_map.dim()}")


def _model_type(model: nn.Module) -> str:
    mt = getattr(model, "model_type", "sr_initializer_net")
    if not isinstance(mt, str):
        return "sr_initializer_net"
    return mt


def _forward_stage0(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    mt = _model_type(model)
    if mt == "stage0_backproj_unet":
        tof_in = ensure_tof_measurement_grid(batch["tof_diff_normalized"]).to(device)
        x_s = batch["x_s"].to(device)
        x_r = batch["x_r"].to(device)
        return model(tof_in, x_s=x_s, x_r=x_r)
    tof_in = _ensure_tof_tumor_4d(batch["tof_diff_normalized"]).to(device)
    return model(tof_in)


def _get_criterion(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {loss_name}. Use 'l1' or 'smooth_l1'.")


def _log_tensor_stats(label: str, t: torch.Tensor) -> None:
    x = t.detach().float()
    logger.info(
        "%s shape=%s min=%.6g max=%.6g mean=%.6g",
        label,
        tuple(x.shape),
        float(x.min().item()),
        float(x.max().item()),
        float(x.mean().item()),
    )


def _maybe_log_first_batch_stage0(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    delta_pred: torch.Tensor,
    delta_target: torch.Tensor,
) -> None:
    if _model_type(model) != "stage0_backproj_unet":
        return
    if not hasattr(model, "backprojection"):
        return
    tof_in = ensure_tof_measurement_grid(batch["tof_diff_normalized"]).to(device)
    x_s = batch["x_s"].to(device)
    x_r = batch["x_r"].to(device)
    with torch.no_grad():
        bp = model.backprojection(tof_in, x_s=x_s, x_r=x_r)
        cov = model.backprojection.coverage_map_image()
    _log_tensor_stats("stage0 tof_diff_normalized (measurement grid)", tof_in)
    _log_tensor_stats("stage0 backprojection prior", bp)
    _log_tensor_stats("stage0 coverage map (1,1,H,W)", cov)
    _log_tensor_stats("stage0 delta_pred", delta_pred)
    _log_tensor_stats("stage0 delta_target", delta_target)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalized_c_base: float,
    *,
    log_first_batch_tensors: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mse_recon = 0.0
    n = 0
    for batch in loader:
        sos_norm = batch["sos_map_normalized"].to(device)
        delta_target = delta_target_from_anatomy(sos_norm, normalized_c_base)
        optimizer.zero_grad()
        delta_pred = _forward_stage0(model, batch, device)
        loss = criterion(delta_pred, delta_target)
        loss.backward()
        optimizer.step()
        if log_first_batch_tensors and n == 0:
            _maybe_log_first_batch_stage0(model, batch, device, delta_pred, delta_target)
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
        sos_norm = batch["sos_map_normalized"].to(device)
        delta_target = delta_target_from_anatomy(sos_norm, normalized_c_base)
        delta_pred = _forward_stage0(model, batch, device)
        total_loss += criterion(delta_pred, delta_target).item()
        c0_pred_norm = c0_normalized_from_delta(normalized_c_base, delta_pred)
        total_mse_recon += nn.functional.mse_loss(c0_pred_norm, sos_norm).item()
        n += 1
    return total_loss / max(n, 1), total_mse_recon / max(n, 1)


def _checkpoint_payload(
    model: nn.Module,
    epoch: int,
    val_loss: float | None,
    val_mse_recon: float | None,
) -> dict[str, Any]:
    mt = _model_type(model)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_mse_recon": val_mse_recon,
        "model_type": mt,
    }
    if hasattr(model, "checkpoint_metadata"):
        payload["model_config"] = model.checkpoint_metadata()
    return payload


def run_stage0(
    model: nn.Module,
    data_config: dict[str, Any],
    training_config: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> str | None:
    """
    Run Stage 0: train initializer on tof_diff_normalized -> delta_c0 (residual).
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
    log_first_batch_tensors = training_config.get("log_first_batch_tensors", True)

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
        "Stage 0 start: epochs=%d lr=%g loss=%s checkpoint_dir=%s model_type=%s",
        epochs,
        lr,
        loss_name,
        checkpoint_dir,
        _model_type(model),
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
            log_first_batch_tensors=log_first_batch_tensors and epoch == 0,
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
                    _checkpoint_payload(model, epoch + 1, val_loss, val_mse_recon),
                    best_path,
                )
        else:
            metrics.on_epoch_end(epoch + 1, train_loss, train_mse_recon)

    if save_last:
        last_path = os.path.join(checkpoint_dir, "last.pt")
        torch.save(
            _checkpoint_payload(model, epochs, last_val_loss, last_val_mse),
            last_path,
        )

    train_logger.info("Stage 0 finished. best_checkpoint=%s", best_path)
    return best_path
