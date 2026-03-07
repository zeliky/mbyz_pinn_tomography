"""Stage 1: physical calibration around frozen Stage 0 initializer.

Uses FMM as forward-only oracle and derivative-free optimization.
Stage 1A: optimize alpha only. Stage 1B: alpha + smooth correction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tomo.data.datamodule import TomographyDataModule
from tomo.initializers.unet_initializer import UNetInitializer
from tomo.state.observation import Observation
from tomo.training.stage1_search import run_stage1a_search, run_stage1b_search


def _ensure_raw_tof_4d(raw_tof: torch.Tensor) -> torch.Tensor:
    """Ensure raw_tof is [B, 1, 32, 32] for the SR model."""
    if raw_tof.dim() == 2:
        return raw_tof.unsqueeze(0).unsqueeze(0)
    if raw_tof.dim() == 3:
        return raw_tof.unsqueeze(1)
    return raw_tof


def run_stage1_operator_baseline(
    full_config: dict[str, Any],
    *,
    device: torch.device | None = None,
    max_samples: int | None = None,
    run_stage1b: bool = True,
) -> dict[str, Any]:
    """Run Stage 1A and optionally Stage 1B calibration.

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
    data_config = full_config.get("data", full_config)
    training_config = full_config.get("training", full_config)
    stage = training_config.get("stage", "stage1a")

    c_base = training_config.get("c_base_phys", training_config.get("c_base", 1.5))
    c_min = training_config.get("c_min_phys", training_config.get("c_min", 1.45))
    c_max = training_config.get("c_max_phys", training_config.get("c_max", 1.8))

    checkpoint_path = training_config.get("checkpoint_path", "checkpoints/stage0/best.pt")
    checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints/stage1a")
    if run_stage1b and stage == "stage1a":
        checkpoint_dir_1b = training_config.get(
            "checkpoint_dir_1b", "checkpoints/stage1b"
        )
    else:
        checkpoint_dir_1b = training_config.get("checkpoint_dir_1b", "checkpoints/stage1b")

    dm = TomographyDataModule(data_config)
    dm.setup()
    loader = dm.val_dataloader()
    dataset = loader.dataset

    min_sos = data_config.get("min_sos", 0.1)
    max_sos = data_config.get("max_sos", 2.1)
    initializer = UNetInitializer(
        num_nodes=128 * 128,
        c0_fallback=float(c_base),
        min_sos=min_sos,
        max_sos=max_sos,
        c_base=float(c_base),
        device=device,
        net=None,
    )
    initializer.load_checkpoint(checkpoint_path, device=device)
    initializer._net.eval()
    for p in initializer._net.parameters():
        p.requires_grad = False

    scale_factor = training_config.get("scale_factor", 1.0)
    coord_range = training_config.get("coord_range")
    if coord_range is not None:
        coord_range = tuple(coord_range)

    results: list[dict[str, Any]] = []
    n_processed = 0

    for batch_idx, batch in enumerate(loader):
        if max_samples is not None and n_processed >= max_samples:
            break

        raw_tof_batch = _ensure_raw_tof_4d(batch["raw_tof"]).to(device)
        batch_size = raw_tof_batch.shape[0]

        for sample_idx in range(batch_size):
            if max_samples is not None and n_processed >= max_samples:
                break

            raw_tof = raw_tof_batch[sample_idx : sample_idx + 1]
            obs = Observation(tof_observed=raw_tof.squeeze(0))
            with torch.no_grad():
                state = initializer(obs)
            c0_flat = state.c_values
            c0 = c0_flat.reshape(128, 128).cpu().numpy()
            delta_c0 = c0 - c_base

            raw = batch["raw_tof"]
            if raw.dim() == 2:
                raw = raw.unsqueeze(0)
            raw = raw[sample_idx : sample_idx + 1]
            sample_batch = {
                "raw_tof": raw,
                "x_s": [batch["x_s"][sample_idx]],
                "x_r": [batch["x_r"][sample_idx]],
            }

            stage1a_result = run_stage1a_search(
                sample_batch,
                delta_c0,
                c0,
                c_base,
                c_min,
                c_max,
                training_config,
                sample_idx=0,
                scale_factor=scale_factor,
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
                    delta_c0,
                    c0,
                    c_base,
                    c_min,
                    c_max,
                    training_config,
                    stage1a_result=stage1a_result,
                    sample_idx=0,
                    scale_factor=scale_factor,
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
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "best_alpha": s1a["best_alpha"],
                    "best_loss": s1a["best_loss"],
                },
                f,
                indent=2,
            )
        if run_stage1b and "stage1b" in res:
            s1b = res["stage1b"]
            out_dir_1b = Path(checkpoint_dir_1b) / f"sample_{i}"
            out_dir_1b.mkdir(parents=True, exist_ok=True)
            np.save(out_dir_1b / "best_alpha.npy", np.array(s1b["best_alpha"]))
            np.save(out_dir_1b / "best_z.npy", s1b.get("best_z", np.array([])))
            np.save(out_dir_1b / "best_c_map.npy", s1b["best_c_map"])
            np.save(out_dir_1b / "tof_pred.npy", s1b["tof_pred"])
            with open(out_dir_1b / "metrics.json", "w") as f:
                json.dump(
                    {
                        "best_alpha": s1b["best_alpha"],
                        "best_loss": s1b["best_loss"],
                        "optimization_success": s1b.get("optimization_success", False),
                    },
                    f,
                    indent=2,
                )

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
