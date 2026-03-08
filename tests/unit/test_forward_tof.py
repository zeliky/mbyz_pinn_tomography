"""Smoke tests for forward_tof: shape and finite output."""

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

try:
    from tomo.operators.matlab_fmm_wrapper import forward_tof
except RuntimeError:
    forward_tof = None


@pytest.mark.skipif(forward_tof is None, reason="py2mat.msfm2d not available")
def test_forward_tof_shape_and_finite() -> None:
    """Minimal grid: constant physical SoS, one source, one receiver; output (S, R), finite."""
    H, W = 32, 32
    sos_map_phys = np.full((H, W), 1.5, dtype=np.float64)
    x_s = np.array([[0.0, 0.0]], dtype=np.float64)
    x_r = np.array([[W - 1.0, H - 1.0]], dtype=np.float64)
    tof = forward_tof(sos_map_phys, x_s, x_r, scale_factor=1.0)
    assert tof.shape == (1, 1)
    assert np.isfinite(tof).all()
    assert tof[0, 0] > 0


@pytest.mark.skipif(forward_tof is None, reason="py2mat.msfm2d not available")
def test_forward_tof_multiple_sources_receivers() -> None:
    """Output shape (S, R) for S sources and R receivers."""
    H, W = 20, 20
    sos_map_phys = np.full((H, W), 1.0, dtype=np.float64)
    x_s = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
    x_r = np.array([[10.0, 10.0], [15.0, 15.0], [19.0, 19.0]], dtype=np.float64)
    tof = forward_tof(sos_map_phys, x_s, x_r, scale_factor=1.0)
    assert tof.shape == (2, 3)
    assert np.isfinite(tof).all()
    assert (tof >= 0).all()
