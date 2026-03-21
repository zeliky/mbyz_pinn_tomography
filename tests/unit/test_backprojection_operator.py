"""Backprojection operator: m = s*R+r (measurement), p = row*W+col (pixel)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tomo.geometry.build_backprojection_operator import (
    build_backprojection_operator,
    save_operator_npz,
)
from tomo.models.backprojection_layer import BackProjectionLayer


def _write_minimal_npz(
    path: Path,
    *,
    x_s: np.ndarray,
    x_r: np.ndarray,
    H: int,
    W: int,
    n_samples: int = 256,
) -> None:
    rows, cols, vals, coverage, meta = build_backprojection_operator(
        x_s,
        x_r,
        H,
        W,
        n_samples=n_samples,
        coord_range=None,
    )
    save_operator_npz(
        str(path),
        rows,
        cols,
        vals,
        coverage,
        x_s.astype(np.float64),
        x_r.astype(np.float64),
        H=H,
        W=W,
        S=meta["S"],
        R=meta["R"],
        n_samples=n_samples,
        scale_by_segment_length=meta["scale_by_segment_length"],
        coord_range_applied=meta["coord_range_applied"],
        coord_range_lo=meta["coord_range_lo"],
        coord_range_hi=meta["coord_range_hi"],
    )


def test_horizontal_ray_reference_mask() -> None:
    """Hand-checked: Tx at (col=0,row=4), Rx at (col=7,row=4) on 8x8 crosses row 4 all cols."""
    H, W = 8, 8
    x_s = np.array([[0.0, 4.0]], dtype=np.float64)
    x_r = np.array([[7.0, 4.0]], dtype=np.float64)
    rows, cols, vals, _cov, _meta = build_backprojection_operator(
        x_s, x_r, H, W, n_samples=512, coord_range=None
    )
    assert (rows == 0).all(), "m = s*R+r with S=1,R=1 -> m=0"
    pix = set(cols.tolist())
    expected = {4 * W + c for c in range(8)}
    assert pix == expected
    assert np.isclose(vals.sum(), 1.0, rtol=1e-5)


def test_measurement_flatten_index_m() -> None:
    """Single unit pulse at (s=1,r=2) with R=3 -> m = 1*3+2 = 5."""
    H, W = 4, 4
    S, R = 2, 3
    xs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    xr = np.array([[2.0, 2.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    rows, cols, vals, _cov, _meta = build_backprojection_operator(
        xs, xr, H, W, n_samples=64, coord_range=None
    )
    assert _meta["S"] == S and _meta["R"] == R
    m_target = 1 * R + 2
    mask = rows == m_target
    assert mask.any()
    assert np.isclose(vals[mask].sum(), 1.0, rtol=1e-4)


def test_backprojection_layer_forward_shape(tmp_path: Path) -> None:
    x_s = np.array([[0.0, 0.0]], dtype=np.float64)
    x_r = np.array([[3.0, 3.0]], dtype=np.float64)
    p = tmp_path / "op.npz"
    _write_minimal_npz(p, x_s=x_s, x_r=x_r, H=4, W=4)
    layer = BackProjectionLayer(str(p), validate_geometry=True)
    B, S, R = 2, 1, 1
    y = torch.zeros(B, S, R)
    y[:, 0, 0] = 1.0
    xs = torch.tensor(x_s, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    xr = torch.tensor(x_r, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    out = layer(y, x_s=xs, x_r=xr)
    assert out.shape == (B, 1, 4, 4)
    assert torch.isfinite(out).all()


def test_geometry_mismatch_raises(tmp_path: Path) -> None:
    x_s = np.array([[0.0, 0.0]], dtype=np.float64)
    x_r = np.array([[3.0, 3.0]], dtype=np.float64)
    p = tmp_path / "op.npz"
    _write_minimal_npz(p, x_s=x_s, x_r=x_r, H=4, W=4)
    layer = BackProjectionLayer(str(p), validate_geometry=True)
    y = torch.zeros(1, 1, 1)
    y[0, 0, 0] = 1.0
    bad_xs = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    xr = torch.tensor(x_r, dtype=torch.float32).unsqueeze(0)
    with pytest.raises(RuntimeError, match="x_s mismatch"):
        layer(y, x_s=bad_xs, x_r=xr)


def test_coverage_norm_changes_output(tmp_path: Path) -> None:
    x_s = np.array([[0.0, 0.0]], dtype=np.float64)
    x_r = np.array([[3.0, 3.0]], dtype=np.float64)
    p = tmp_path / "op.npz"
    _write_minimal_npz(p, x_s=x_s, x_r=x_r, H=4, W=4)
    y = torch.zeros(1, 1, 1)
    y[0, 0, 0] = 1.0
    xs = torch.tensor(x_s, dtype=torch.float32).unsqueeze(0)
    xr = torch.tensor(x_r, dtype=torch.float32).unsqueeze(0)
    on = BackProjectionLayer(str(p), coverage_norm=True)
    off = BackProjectionLayer(str(p), coverage_norm=False)
    o_on = on(y.clone(), x_s=xs, x_r=xr)
    o_off = off(y.clone(), x_s=xs, x_r=xr)
    assert torch.isfinite(o_on).all() and torch.isfinite(o_off).all()
    assert not torch.allclose(o_on, o_off, rtol=1e-3, atol=1e-3)

