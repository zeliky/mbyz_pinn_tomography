"""Stage 1: physical calibration around frozen Stage 0 initializer.

Uses FMM as forward-only oracle and derivative-free optimization.
Stage 1A: optimize alpha only. Stage 1B: alpha + smooth correction.
Builds c_map in scaled space and converts to physical before the solver.

Frozen initializer inference uses ``tof_diff_normalized`` (same input contract as Stage 0).
Physics loss, ``forward_tof``, residuals, and preflight shape/scale checks use
``tof_tumor_raw`` only.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tomo.data.datamodule import TomographyDataModule
from tomo.data.normalization_metadata import resolve_data_config
from tomo.initializers.unet_initializer import UNetInitializer
from tomo.training.stage0_initializer import (
    _ensure_tof_tumor_4d,
    ensure_tof_measurement_grid,
)
from tomo.operators.matlab_fmm_wrapper import forward_tof
from tomo.state.observation import Observation
from tomo.training.stage1_parametrization import params_to_c_map_scaled
from tomo.training.stage1_search import run_stage1a_search, run_stage1b_search
from tomo.utils.units import physical_delta_to_scaled, to_physical_sos, to_scaled_sos

logger = logging.getLogger(__name__)


def _initializer_model_type(net: nn.Module) -> str:
    mt = getattr(net, "model_type", "sr_initializer_net")
    if not isinstance(mt, str):
        return "sr_initializer_net"
    return mt


def _initializer_tof_batch(batch: dict[str, Any], net: nn.Module) -> torch.Tensor:
    mt = _initializer_model_type(net)
    if mt == "stage0_backproj_unet":
        return ensure_tof_measurement_grid(batch["tof_diff_normalized"])
    return _ensure_tof_tumor_4d(batch["tof_diff_normalized"])


def _initializer_observation(
    batch: dict[str, Any],
    sample_idx: int,
    initializer_tof_bchw: torch.Tensor,
    net: nn.Module,
) -> Observation:
    tof_1 = initializer_tof_bchw[sample_idx : sample_idx + 1].squeeze(0)
    mt = _initializer_model_type(net)
    if mt == "stage0_backproj_unet":
        return Observation(
            tof_observed=tof_1,
            layout_metadata={
                "x_s": batch["x_s"][sample_idx : sample_idx + 1],
                "x_r": batch["x_r"][sample_idx : sample_idx + 1],
            },
        )
    return Observation(tof_observed=tof_1)


def _log_tof_stats(label: str, x: Any) -> None:
    """Log min/max/mean/std for a tensor or array (finite values only)."""
    if hasattr(x, "detach"):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        logger.info("Stage 1 ToF stats [%s]: no finite values", label)
        return
    logger.info(
        "Stage 1 ToF stats [%s]: min=%s max=%s mean=%s std=%s (n=%d)",
        label,
        float(np.min(flat)),
        float(np.max(flat)),
        float(np.mean(flat)),
        float(np.std(flat)),
        int(flat.size),
    )


def _run_stage1_preflight(
    loader: Any,
    initializer: UNetInitializer,
    c_base_phys: float,
    min_sos: float,
    max_sos: float,
    c_min_phys: float,
    c_max_phys: float,
    tof_output_scale: float,
    coord_range: tuple[float, float] | None,
    *,
    preflight_tof_ratio_min: float = 0.01,
    preflight_tof_ratio_max: float = 100.0,
) -> None:
    """Run preflight checks before Stage 1A/1B. Exits with clear message on failure."""
    # 1. FMM wrapper can run
    try:
        H, W = 128, 128
        sos_phys = np.full((H, W), c_base_phys, dtype=np.float64)
        x_s = np.array([[0.0, 0.0]], dtype=np.float64)
        x_r = np.array([[W - 1.0, H - 1.0]], dtype=np.float64)
        _ = forward_tof(sos_phys, x_s, x_r, scale_factor=tof_output_scale)
    except Exception as e:
        print(f"Stage 1 preflight failed: FMM solver unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. One batch loads
    batch = next(iter(loader), None)
    if batch is None:
        print("Stage 1 preflight failed: loader is empty (no batch).", file=sys.stderr)
        sys.exit(1)
    for key in ("tof_diff_normalized", "tof_tumor_raw", "x_s", "x_r"):
        if key not in batch:
            print(f"Stage 1 preflight failed: batch missing key '{key}'.", file=sys.stderr)
            sys.exit(1)

    _log_tof_stats("initializer_input_tof_diff_normalized", batch["tof_diff_normalized"])
    _log_tof_stats("physical_observation_tof_tumor_raw", batch["tof_tumor_raw"])

    # 3. delta_c0 shape and finite (initializer: same input as Stage 0)
    initializer_tof_4d = _initializer_tof_batch(batch, initializer._net).to(initializer._device)
    if initializer_tof_4d.dim() == 3:
        initializer_tof_4d = initializer_tof_4d.unsqueeze(1)
    obs = _initializer_observation(batch, 0, initializer_tof_4d, initializer._net)
    with torch.no_grad():
        state = initializer(obs)
    c0_phys = (
        state.c_map_2d.cpu().numpy()
        if state.c_map_2d is not None
        else state.c_values.reshape(128, 128).cpu().numpy()
    )
    delta_c0_phys = c0_phys - c_base_phys
    if delta_c0_phys.shape != (128, 128):
        print(
            f"Stage 1 preflight failed: delta_c0 shape {delta_c0_phys.shape} != (128, 128).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not np.isfinite(delta_c0_phys).all():
        print("Stage 1 preflight failed: delta_c0 contains NaN or Inf.", file=sys.stderr)
        sys.exit(1)

    c_base_scaled = to_scaled_sos(c_base_phys, min_sos, max_sos)
    delta_c0_scaled = physical_delta_to_scaled(delta_c0_phys, min_sos, max_sos)
    if delta_c0_scaled.shape != (128, 128):
        print(
            f"Stage 1 preflight failed: delta_c0_scaled shape {delta_c0_scaled.shape} != (128, 128).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not np.isfinite(delta_c0_scaled).all():
        print("Stage 1 preflight failed: delta_c0_scaled contains NaN or Inf.", file=sys.stderr)
        sys.exit(1)

    # 4. c_map_scaled stats in [0, 1]
    c_map_scaled = params_to_c_map_scaled(
        np.array([1.0]), c_base_scaled, delta_c0_scaled, "stage1a", target_h=128, target_w=128
    )
    c_min_s, c_max_s = float(np.min(c_map_scaled)), float(np.max(c_map_scaled))
    if c_min_s < -0.01 or c_max_s > 1.01:
        print(
            f"Stage 1 preflight failed: c_map_scaled out of [0,1]: min={c_min_s}, max={c_max_s}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 5. c_map_phys stats in [min_sos, max_sos]
    c_map_phys = to_physical_sos(c_map_scaled, min_sos, max_sos)
    c_min_p, c_max_p = float(np.min(c_map_phys)), float(np.max(c_map_phys))
    if c_min_p < min_sos - 1e-6 or c_max_p > max_sos + 1e-6:
        print(
            f"Stage 1 preflight failed: c_map_phys out of [{min_sos},{max_sos}]: "
            f"min={c_min_p}, max={c_max_p}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 6. forward_tof shape matches physical ToF (tof_tumor_raw)
    x_s = batch["x_s"][0]
    x_r = batch["x_r"][0]
    if hasattr(x_s, "detach"):
        x_s = x_s.detach().cpu().numpy()
    if hasattr(x_r, "detach"):
        x_r = x_r.detach().cpu().numpy()
    x_s = np.asarray(x_s, dtype=np.float64)
    x_r = np.asarray(x_r, dtype=np.float64)
    if x_s.ndim == 1:
        x_s = x_s.reshape(1, -1)
    if x_r.ndim == 1:
        x_r = x_r.reshape(1, -1)
    physical_tof_np = batch["tof_tumor_raw"]
    if hasattr(physical_tof_np, "detach"):
        physical_tof_np = physical_tof_np.detach().cpu().numpy()
    physical_tof_np = np.asarray(physical_tof_np)
    if physical_tof_np.ndim == 4:
        physical_tof_np = physical_tof_np[0].squeeze(0)
    elif physical_tof_np.ndim == 3:
        physical_tof_np = physical_tof_np[0]
    tof_pred = forward_tof(c_map_phys, x_s, x_r, scale_factor=tof_output_scale)
    if not np.isfinite(tof_pred).all():
        print("Stage 1 preflight failed: forward_tof returned non-finite values.", file=sys.stderr)
        sys.exit(1)
    if tof_pred.shape != physical_tof_np.shape:
        print(
            f"Stage 1 preflight failed: forward_tof shape {tof_pred.shape} != "
            f"tof_tumor_raw shape {physical_tof_np.shape}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 7. scale_factor validation: ToF order-of-magnitude check (vs physical observation)
    pred_median = float(np.median(tof_pred))
    obs_median = float(np.median(physical_tof_np))
    if obs_median <= 0:
        obs_median = np.finfo(np.float64).tiny
    ratio = pred_median / obs_median
    if ratio < preflight_tof_ratio_min or ratio > preflight_tof_ratio_max:
        print(
            "Stage 1 preflight failed: predicted ToF and observed ToF differ by more than two orders of magnitude; "
            f"likely scale_factor mismatch. Predicted median: {pred_median}, observed median: {obs_median}. "
            f"Check scale_factor (currently {tof_output_scale}) and SoS units.",
            file=sys.stderr,
        )
        sys.exit(1)


def run_stage1_operator_baseline(
    full_config: dict[str, Any],
    *,
    device: torch.device | None = None,
    max_samples: int | None = None,
    run_stage1b: bool = True,
) -> dict[str, Any]:
    """Run Stage 1A and optionally Stage 1B calibration.

    The frozen Stage 0 initializer is run on ``tof_diff_normalized``. Stage 1A/1B
    physics objectives compare ``forward_tof`` predictions to ``tof_tumor_raw`` only.

    Args:
        full_config: Hydra-composed config with data, training (stage1a/stage1b),
            initializer, checkpoint path, etc.
        device: Device for initializer inference.
        max_samples: Max number of samples to process (None = all).
        run_stage1b: Whether to run Stage 1B after Stage 1A.

    Returns:
        Aggregated results dict with per-sample and summary metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = resolve_data_config(dict(full_config.get("data", full_config)))
    training_config = full_config.get("training", full_config)
    stage = training_config.get("stage", "stage1a")

    c_base_phys = training_config.get("c_base_phys", training_config.get("c_base", 1.5))
    c_min_phys = training_config.get("c_min_phys", training_config.get("c_min", 1.45))
    c_max_phys = training_config.get("c_max_phys", training_config.get("c_max", 1.8))
    min_sos = data_config["min_sos"]
    max_sos = data_config["max_sos"]
    # Single source of truth: c_base_phys; derive scaled at runtime
    c_base_scaled = to_scaled_sos(c_base_phys, min_sos, max_sos)

    checkpoint_path = training_config.get("checkpoint_path", "checkpoints/stage0/best.pt")
    if not Path(checkpoint_path).exists():
        print(
            f"Stage 1 preflight failed: checkpoint not found: {checkpoint_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints/stage1a")
    if run_stage1b and stage == "stage1a":
        checkpoint_dir_1b = training_config.get(
            "checkpoint_dir_1b", "checkpoints/stage1b"
        )
    else:
        checkpoint_dir_1b = training_config.get("checkpoint_dir_1b", "checkpoints/stage1b")

    dm = TomographyDataModule(data_config)
    dm.setup()
    loader_split = training_config.get("loader_split", "validation")
    if loader_split == "train":
        loader = dm.train_dataloader()
    elif loader_split == "test":
        loader = dm.test_dataloader()
    else:
        loader = dm.val_dataloader()
    init_cfg = full_config.get("initializer")
    init_net: nn.Module | None = None
    expected_model_type: str | None = None
    if init_cfg is not None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(init_cfg) if isinstance(init_cfg, dict) else init_cfg
        init_net = instantiate(cfg)
        expected_model_type = str(getattr(init_net, "model_type", "")) or None

    initializer = UNetInitializer(
        num_nodes=128 * 128,
        c0_fallback=float(c_base_phys),
        min_sos=min_sos,
        max_sos=max_sos,
        c_base=float(c_base_phys),
        device=device,
        net=init_net,
        expected_model_type=expected_model_type,
    )
    initializer.load_checkpoint(checkpoint_path, device=device)
    initializer._net.eval()
    for p in initializer._net.parameters():
        p.requires_grad = False

    tof_output_scale = training_config.get("scale_factor", training_config.get("tof_output_scale", 1.0))
    coord_range = training_config.get("coord_range")
    if coord_range is not None:
        coord_range = tuple(coord_range)
    preflight_tof_ratio_min = training_config.get("preflight_tof_ratio_min", 0.01)
    preflight_tof_ratio_max = training_config.get("preflight_tof_ratio_max", 100.0)

    _run_stage1_preflight(
        loader,
        initializer,
        c_base_phys,
        min_sos,
        max_sos,
        c_min_phys,
        c_max_phys,
        tof_output_scale,
        coord_range,
        preflight_tof_ratio_min=preflight_tof_ratio_min,
        preflight_tof_ratio_max=preflight_tof_ratio_max,
    )

    results: list[dict[str, Any]] = []
    n_processed = 0

    for batch_idx, batch in enumerate(loader):
        if max_samples is not None and n_processed >= max_samples:
            break

        _log_tof_stats(
            f"batch_{batch_idx}_initializer_input_tof_diff_normalized",
            batch["tof_diff_normalized"],
        )
        _log_tof_stats(
            f"batch_{batch_idx}_physical_observation_tof_tumor_raw",
            batch["tof_tumor_raw"],
        )

        initializer_tof_bchw = _initializer_tof_batch(batch, initializer._net).to(device)
        physical_tof_bchw_shape = _ensure_tof_tumor_4d(batch["tof_tumor_raw"]).shape[0]
        batch_size = physical_tof_bchw_shape
        if initializer_tof_bchw.shape[0] != batch_size:
            print(
                "Stage 1 failed: batch size mismatch between tof_diff_normalized and tof_tumor_raw.",
                file=sys.stderr,
            )
            sys.exit(1)

        for sample_idx in range(batch_size):
            if max_samples is not None and n_processed >= max_samples:
                break

            obs = _initializer_observation(batch, sample_idx, initializer_tof_bchw, initializer._net)
            with torch.no_grad():
                state = initializer(obs)
            c0_phys = (
                state.c_map_2d.cpu().numpy()
                if state.c_map_2d is not None
                else state.c_values.reshape(128, 128).cpu().numpy()
            )
            delta_c0_phys = c0_phys - c_base_phys
            delta_c0_scaled = physical_delta_to_scaled(delta_c0_phys, min_sos, max_sos)

            # Physics search / FMM objective: only tof_tumor_raw (no fallback from diff).
            physical_tof_slice = batch["tof_tumor_raw"]
            if physical_tof_slice.dim() == 2:
                physical_tof_slice = physical_tof_slice.unsqueeze(0)
            physical_tof_slice = physical_tof_slice[sample_idx : sample_idx + 1]
            sample_batch = {
                "tof_tumor_raw": physical_tof_slice,
                "x_s": batch["x_s"][sample_idx : sample_idx + 1],
                "x_r": batch["x_r"][sample_idx : sample_idx + 1],
            }

            stage1a_result = run_stage1a_search(
                sample_batch,
                delta_c0_scaled,
                c0_phys,
                c_base_scaled,
                min_sos,
                max_sos,
                c_min_phys,
                c_max_phys,
                training_config,
                sample_idx=0,
                tof_output_scale=tof_output_scale,
                coord_range=coord_range,
            )

            sample_result: dict[str, Any] = {
                "sample_idx": n_processed,
                "batch_idx": batch_idx,
                "stage1a": stage1a_result,
            }

            if run_stage1b:
                stage1b_result = run_stage1b_search(
                    sample_batch,
                    delta_c0_scaled,
                    c0_phys,
                    c_base_scaled,
                    min_sos,
                    max_sos,
                    c_min_phys,
                    c_max_phys,
                    training_config,
                    stage1a_result=stage1a_result,
                    sample_idx=0,
                    tof_output_scale=tof_output_scale,
                    coord_range=coord_range,
                )
                sample_result["stage1b"] = stage1b_result

            results.append(sample_result)
            n_processed += 1

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if run_stage1b:
        Path(checkpoint_dir_1b).mkdir(parents=True, exist_ok=True)

    for i, res in enumerate(results):
        out_dir = Path(checkpoint_dir) / f"sample_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s1a = res["stage1a"]
        np.save(out_dir / "best_alpha.npy", np.array(s1a["best_alpha"]))
        np.save(out_dir / "best_c_map.npy", s1a["best_c_map"])
        np.save(out_dir / "tof_pred.npy", s1a["tof_pred"])
        metrics_1a: dict[str, Any] = {
            "best_alpha": s1a["best_alpha"],
            "best_loss": s1a["best_loss"],
        }
        if "diagnostics" in s1a:
            metrics_1a["diagnostics"] = s1a["diagnostics"]
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_1a, f, indent=2)
        if run_stage1b and "stage1b" in res:
            s1b = res["stage1b"]
            out_dir_1b = Path(checkpoint_dir_1b) / f"sample_{i}"
            out_dir_1b.mkdir(parents=True, exist_ok=True)
            np.save(out_dir_1b / "best_alpha.npy", np.array(s1b["best_alpha"]))
            np.save(out_dir_1b / "best_z.npy", s1b.get("best_z", np.array([])))
            np.save(out_dir_1b / "best_c_map.npy", s1b["best_c_map"])
            np.save(out_dir_1b / "tof_pred.npy", s1b["tof_pred"])
            metrics_1b: dict[str, Any] = {
                "best_alpha": s1b["best_alpha"],
                "best_loss": s1b["best_loss"],
                "optimization_success": s1b.get("optimization_success", False),
            }
            if "diagnostics" in s1b:
                metrics_1b["diagnostics"] = s1b["diagnostics"]
            with open(out_dir_1b / "metrics.json", "w") as f:
                json.dump(metrics_1b, f, indent=2)

    avg_loss_1a = np.mean([r["stage1a"]["best_loss"] for r in results])
    summary: dict[str, Any] = {
        "num_samples": len(results),
        "stage1a_avg_loss": float(avg_loss_1a),
        "results": results,
    }
    if run_stage1b:
        avg_loss_1b = np.mean(
            [r["stage1b"]["best_loss"] for r in results if "stage1b" in r]
        )
        summary["stage1b_avg_loss"] = float(avg_loss_1b)

    return summary
