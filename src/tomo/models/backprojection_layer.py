"""Fixed backprojection: sparse scatter-add from ToF grid [B,S,R] to image [B,1,H,W].

Measurement flattening: m = s * R + r
Pixel flattening: p = row * W + col
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tomo.geometry.build_backprojection_operator import (
    COORD_CONVENTION,
    MEASUREMENT_INDEX_FORMULA,
    PIXEL_INDEX_FORMULA,
    load_operator_arrays,
)


class BackProjectionLayer(nn.Module):
    """Load precomputed COO operator; forward is scatter-add + optional coverage normalization."""

    def __init__(
        self,
        operator_path: str,
        *,
        coverage_norm: bool = True,
        coverage_power: float = 1.0,
        coverage_eps: float = 1e-8,
        validate_geometry: bool = True,
        geometry_rtol: float = 1e-5,
        geometry_atol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.coverage_norm = coverage_norm
        self.coverage_power = float(coverage_power)
        self.coverage_eps = float(coverage_eps)
        self.validate_geometry = validate_geometry
        self.geometry_rtol = geometry_rtol
        self.geometry_atol = geometry_atol

        data = load_operator_arrays(operator_path)
        exp_m = MEASUREMENT_INDEX_FORMULA
        exp_p = PIXEL_INDEX_FORMULA
        if data["_measurement_formula"] != exp_m or data["_pixel_formula"] != exp_p:
            raise ValueError(
                "Operator .npz flatten formulas do not match this code version: "
                f"got measurement={data['_measurement_formula']!r} pixel={data['_pixel_formula']!r}"
            )
        if data["_coord_convention"] != COORD_CONVENTION:
            raise ValueError(
                f"Operator coord_convention {data['_coord_convention']!r} != {COORD_CONVENTION!r}"
            )

        self.H = data["_H"]
        self.W = data["_W"]
        self.S = data["_S"]
        self.R = data["_R"]
        self.M = self.S * self.R
        self.P = self.H * self.W

        rows = torch.as_tensor(data["rows"], dtype=torch.long)
        cols = torch.as_tensor(data["cols"], dtype=torch.long)
        vals = torch.as_tensor(data["vals"], dtype=torch.float32)
        coverage = torch.as_tensor(data["coverage"], dtype=torch.float32)

        x_s_ref = torch.as_tensor(data["x_s_ref"], dtype=torch.float64)
        x_r_ref = torch.as_tensor(data["x_r_ref"], dtype=torch.float64)

        self.register_buffer("rows", rows, persistent=True)
        self.register_buffer("cols", cols, persistent=True)
        self.register_buffer("vals", vals, persistent=True)
        self.register_buffer("coverage", coverage, persistent=True)
        self.register_buffer("x_s_ref", x_s_ref, persistent=True)
        self.register_buffer("x_r_ref", x_r_ref, persistent=True)

        self._coord_range_applied = data["_coord_range_applied"]
        self._coord_range_lo = data["_coord_range_lo"]
        self._coord_range_hi = data["_coord_range_hi"]
        self._geometry_digest = data["_geometry_digest"]
        self._operator_path = operator_path

    def extra_repr(self) -> str:
        return (
            f"S={self.S}, R={self.R}, H={self.H}, W={self.W}, "
            f"coverage_norm={self.coverage_norm}, coverage_power={self.coverage_power}"
        )

    def assert_geometry_matches(
        self,
        x_s: torch.Tensor,
        x_r: torch.Tensor,
    ) -> None:
        """Raise RuntimeError if batch geometry differs from operator reference."""
        if not self.validate_geometry:
            return
        xs = x_s.detach().float().cpu()
        xr = x_r.detach().float().cpu()
        if xs.ndim == 2:
            xs = xs.unsqueeze(0)
        if xr.ndim == 2:
            xr = xr.unsqueeze(0)
        if xs.shape[-2] != self.S or xs.shape[-1] < 2:
            raise RuntimeError(
                f"x_s shape {tuple(x_s.shape)} incompatible with operator S={self.S} (expected …xSx2+)"
            )
        if xr.shape[-2] != self.R or xr.shape[-1] < 2:
            raise RuntimeError(
                f"x_r shape {tuple(x_r.shape)} incompatible with operator R={self.R} (expected …xRx2+)"
            )
        ref_s = self.x_s_ref.float()
        ref_r = self.x_r_ref.float()
        for b in range(xs.shape[0]):
            if not torch.allclose(xs[b, :, :2], ref_s[:, :2], rtol=self.geometry_rtol, atol=self.geometry_atol):
                raise RuntimeError(
                    f"x_s mismatch vs operator reference at batch index {b} "
                    f"(rtol={self.geometry_rtol}, atol={self.geometry_atol})"
                )
            if not torch.allclose(xr[b, :, :2], ref_r[:, :2], rtol=self.geometry_rtol, atol=self.geometry_atol):
                raise RuntimeError(
                    f"x_r mismatch vs operator reference at batch index {b} "
                    f"(rtol={self.geometry_rtol}, atol={self.geometry_atol})"
                )

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_s: torch.Tensor | None = None,
        x_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backproject [B,S,R] or [B,1,S,R] -> [B,1,H,W]."""
        if x_s is not None and x_r is not None:
            self.assert_geometry_matches(x_s, x_r)

        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected [B,1,S,R], got shape {tuple(x.shape)}")
            x = x[:, 0]
        if x.dim() != 3:
            raise ValueError(f"Expected [B,S,R] or [B,1,S,R], got shape {tuple(x.shape)}")
        B, S, R = x.shape
        if S != self.S or R != self.R:
            raise ValueError(
                f"ToF shape S,R={S},{R} does not match operator S,R={self.S},{self.R}"
            )

        y = x.reshape(B, self.M)
        contrib = y[:, self.rows] * self.vals.unsqueeze(0)
        out = x.new_zeros((B, self.P))
        out.scatter_add_(1, self.cols.unsqueeze(0).expand(B, -1), contrib)

        if self.coverage_norm:
            denom = self.coverage.clamp_min(0.0)
            if self.coverage_power != 1.0:
                denom = denom.pow(self.coverage_power)
            out = out / (denom.unsqueeze(0) + self.coverage_eps)

        return out.view(B, 1, self.H, self.W)

    def coverage_map_image(self) -> torch.Tensor:
        """[1,1,H,W] coverage for debugging or U-Net second channel."""
        return self.coverage.view(1, 1, self.H, self.W)

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "operator_path": self._operator_path,
            "S": self.S,
            "R": self.R,
            "H": self.H,
            "W": self.W,
            "coord_range_applied": self._coord_range_applied,
            "coord_range_lo": self._coord_range_lo,
            "coord_range_hi": self._coord_range_hi,
            "geometry_digest": self._geometry_digest,
            "measurement_index_formula": MEASUREMENT_INDEX_FORMULA,
            "pixel_index_formula": PIXEL_INDEX_FORMULA,
            "coord_convention": COORD_CONVENTION,
        }
