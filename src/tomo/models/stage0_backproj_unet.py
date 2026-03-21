"""Stage 0: fixed backprojection + MONAI U-Net -> delta_c0 (residual)."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from tomo.models.backprojection_layer import BackProjectionLayer


def _apply_output_activation(
    x: torch.Tensor, mode: str, *, tanh_scale: float
) -> torch.Tensor:
    if mode in ("none", ""):
        return x
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "tanh_scaled":
        return float(tanh_scale) * torch.tanh(x)
    raise ValueError(f"Unknown output_activation: {mode!r} (use none|sigmoid|tanh_scaled)")


class Stage0BackprojUNet(nn.Module):
    """tof_diff [B,1,S,R] (+ x_s, x_r for geometry check) -> delta [B,1,H,W]."""

    model_type: ClassVar[str] = "stage0_backproj_unet"

    def __init__(
        self,
        operator_path: str,
        *,
        add_coverage_channel: bool = False,
        normalize_coverage_channel_for_unet: bool = False,
        output_activation: str = "none",
        tanh_scale: float = 1.0,
        coverage_norm: bool = True,
        coverage_power: float = 1.0,
        coverage_eps: float = 1e-8,
        validate_geometry: bool = True,
        geometry_rtol: float = 1e-5,
        geometry_atol: float = 1e-6,
        unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        unet_strides: tuple[int, ...] = (2, 2, 2, 2),
        unet_num_res_units: int = 0,
    ) -> None:
        super().__init__()
        uc = tuple(int(x) for x in unet_channels)
        us = tuple(int(x) for x in unet_strides)
        self.add_coverage_channel = bool(add_coverage_channel)
        self.normalize_coverage_channel_for_unet = bool(normalize_coverage_channel_for_unet)
        self.output_activation = str(output_activation)
        self.tanh_scale = float(tanh_scale)

        self.backprojection = BackProjectionLayer(
            operator_path,
            coverage_norm=coverage_norm,
            coverage_power=coverage_power,
            coverage_eps=coverage_eps,
            validate_geometry=validate_geometry,
            geometry_rtol=geometry_rtol,
            geometry_atol=geometry_atol,
        )

        in_ch = 2 if self.add_coverage_channel else 1
        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=1,
            channels=uc,
            strides=us,
            num_res_units=int(unet_num_res_units),
        )

    def forward(
        self,
        tof_diff: torch.Tensor,
        *,
        x_s: torch.Tensor | None = None,
        x_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bp = self.backprojection(tof_diff, x_s=x_s, x_r=x_r)
        if self.add_coverage_channel:
            cov = self.backprojection.coverage_map_image().to(
                device=bp.device, dtype=bp.dtype
            )
            if self.normalize_coverage_channel_for_unet:
                cov = cov / (cov.amax() + 1e-8)
            cov = cov.expand(bp.shape[0], -1, -1, -1)
            x = torch.cat([bp, cov], dim=1)
        else:
            x = bp
        y = self.unet(x)
        return _apply_output_activation(y, self.output_activation, tanh_scale=self.tanh_scale)

    def checkpoint_metadata(self) -> dict[str, Any]:
        m = self.backprojection.metadata_dict()
        m.update(
            {
                "model_type": self.model_type,
                "add_coverage_channel": self.add_coverage_channel,
                "normalize_coverage_channel_for_unet": self.normalize_coverage_channel_for_unet,
                "output_activation": self.output_activation,
                "tanh_scale": self.tanh_scale,
                "coverage_norm": self.backprojection.coverage_norm,
                "coverage_power": self.backprojection.coverage_power,
                "coverage_eps": self.backprojection.coverage_eps,
                "validate_geometry": self.backprojection.validate_geometry,
            }
        )
        return m
