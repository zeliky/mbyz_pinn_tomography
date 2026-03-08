"""Unit tests for SoS conversion helpers in tomo.utils.units."""

import numpy as np
import pytest

from tomo.utils.units import (
    physical_delta_to_scaled,
    scaled_delta_to_physical,
    to_physical_sos,
    to_scaled_sos,
)


def test_to_scaled_sos_scalar_in_range() -> None:
    """to_scaled_sos maps [min_sos, max_sos] to [0, 1]."""
    lo, hi = 0.1, 2.1
    assert to_scaled_sos(lo, lo, hi, clamp=True) == 0.0
    assert to_scaled_sos(hi, lo, hi, clamp=True) == 1.0
    assert abs(to_scaled_sos(1.1, lo, hi, clamp=True) - 0.5) < 1e-9


def test_to_physical_sos_scalar() -> None:
    """to_physical_sos maps [0, 1] to [min_sos, max_sos]."""
    lo, hi = 0.1, 2.1
    assert to_physical_sos(0.0, lo, hi) == lo
    assert to_physical_sos(1.0, lo, hi) == hi
    assert abs(to_physical_sos(0.5, lo, hi) - 1.1) < 1e-9


def test_roundtrip_scaled_physical_scalar() -> None:
    """to_physical_sos(to_scaled_sos(x)) == x for x in [lo, hi]."""
    lo, hi = 0.1, 2.1
    for x in (lo, hi, 1.0, 1.5):
        scaled = to_scaled_sos(x, lo, hi, clamp=True)
        back = to_physical_sos(scaled, lo, hi)
        assert abs(back - x) < 1e-9, f"x={x} -> scaled={scaled} -> back={back}"


def test_roundtrip_scaled_physical_array() -> None:
    """Roundtrip for numpy array."""
    lo, hi = 0.1, 2.1
    x = np.array([0.2, 1.0, 2.0, 1.5])
    scaled = to_scaled_sos(x, lo, hi, clamp=True)
    back = to_physical_sos(scaled, lo, hi)
    np.testing.assert_array_almost_equal(back, x)


def test_physical_delta_to_scaled_scalar() -> None:
    """physical_delta_to_scaled: delta_phys -> delta_scaled."""
    lo, hi = 0.1, 2.1
    delta_phys = 0.5
    d_scaled = physical_delta_to_scaled(delta_phys, lo, hi)
    assert abs(d_scaled - 0.5 / (hi - lo)) < 1e-9


def test_scaled_delta_to_physical_scalar() -> None:
    """scaled_delta_to_physical: delta_scaled -> delta_phys."""
    lo, hi = 0.1, 2.1
    delta_scaled = 0.25
    d_phys = scaled_delta_to_physical(delta_scaled, lo, hi)
    assert abs(d_phys - 0.25 * (hi - lo)) < 1e-9


def test_roundtrip_delta_scalar() -> None:
    """scaled_delta_to_physical(physical_delta_to_scaled(d)) == d."""
    lo, hi = 0.1, 2.1
    for d in (0.0, 0.5, 1.0, -0.2):
        scaled = physical_delta_to_scaled(d, lo, hi)
        back = scaled_delta_to_physical(scaled, lo, hi)
        assert abs(back - d) < 1e-9, f"d={d} -> scaled={scaled} -> back={back}"


def test_roundtrip_delta_array() -> None:
    """Delta roundtrip for numpy array."""
    lo, hi = 0.1, 2.1
    d = np.array([0.0, 0.5, -0.1, 1.0])
    scaled = physical_delta_to_scaled(d, lo, hi)
    back = scaled_delta_to_physical(scaled, lo, hi)
    np.testing.assert_array_almost_equal(back, d)


def test_to_scaled_sos_clamp() -> None:
    """to_scaled_sos with clamp=True clips to [0, 1]."""
    lo, hi = 0.1, 2.1
    assert to_scaled_sos(0.0, lo, hi, clamp=True) == 0.0
    assert to_scaled_sos(3.0, lo, hi, clamp=True) == 1.0
