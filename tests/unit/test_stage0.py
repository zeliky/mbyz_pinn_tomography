"""Unit tests for Stage 0 initializer: SR net shapes, residual target, one-step training."""

import numpy as np
import pytest
import torch

from tomo.initializers.unet_initializer import (
    SRInitializerNet,
    c0_normalized_from_delta,
    delta_target_from_anatomy,
    normalized_c_base_from_config,
)
from tomo.training.stage0_initializer import (
    _ensure_tof_tumor_4d,
    _get_criterion,
    _train_epoch,
    ensure_tof_measurement_grid,
)


def test_sr_initializer_net_baseline_shapes() -> None:
    """SRInitializerNet (no attention): [B, 1, 64, 64] -> delta_c0 [B, 1, 128, 128]."""
    model = SRInitializerNet(use_attention=False)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_sr_initializer_net_attention_shapes() -> None:
    """SRInitializerNet (with attention): same shapes."""
    model = SRInitializerNet(use_attention=True)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ensure_tof_measurement_grid_shape() -> None:
    """Measurement grid [B,S,R] -> [B,1,S,R]."""
    x = torch.randn(2, 8, 8)
    y = ensure_tof_measurement_grid(x)
    assert y.shape == (2, 1, 8, 8)


def test_ensure_tof_tumor_4d() -> None:
    """_ensure_tof_tumor_4d adds channel dim when tensor is [B, H, W]."""
    x3 = torch.randn(3, 64, 64)
    x4 = _ensure_tof_tumor_4d(x3)
    assert x4.shape == (3, 1, 64, 64)
    x4_already = torch.randn(3, 1, 64, 64)
    assert _ensure_tof_tumor_4d(x4_already).shape == (3, 1, 64, 64)


def test_normalized_c_base_from_config() -> None:
    """normalized_c_base is in [0, 1] and consistent with min_sos/max_sos."""
    n = normalized_c_base_from_config(0.1, 2.1, 1.5)
    assert 0.0 <= n <= 1.0
    assert abs(n - (1.5 - 0.1) / (2.1 - 0.1)) < 1e-6


def test_delta_target_from_anatomy() -> None:
    """delta_target is non-negative and clamped."""
    anatomy = torch.tensor([[[[0.2, 0.8], [0.5, 1.0]]]])
    delta = delta_target_from_anatomy(anatomy, 0.5)
    assert (delta >= 0).all() and (delta <= 1).all()
    assert delta.shape == anatomy.shape


def test_c0_normalized_from_delta_in_range() -> None:
    """c0_pred = normalized_c_base + delta_pred stays in [0, 1]."""
    norm_base = 0.5
    delta_pred = torch.rand(2, 1, 128, 128)
    c0 = c0_normalized_from_delta(norm_base, delta_pred)
    assert c0.shape == delta_pred.shape
    assert c0.min() >= 0.0 and c0.max() <= 1.0


def test_stage0_one_step_backward_residual() -> None:
    """One training step with residual target: delta_target, loss on residual."""
    model = SRInitializerNet(use_attention=False)
    criterion = _get_criterion("smooth_l1")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    normalized_c_base = normalized_c_base_from_config(0.1, 2.1, 1.5)
    anatomy = torch.rand(2, 1, 128, 128)
    batch = {
        "tof_diff_normalized": torch.rand(2, 1, 64, 64),
        "sos_map_normalized": anatomy,
    }
    device = torch.device("cpu")
    train_loss, train_mse_recon = _train_epoch(
        model,
        [batch],
        criterion,
        optimizer,
        device,
        normalized_c_base,
        log_first_batch_tensors=False,
    )
    assert isinstance(train_loss, float)
    assert isinstance(train_mse_recon, float)
    assert train_loss >= 0.0


def test_stage0_forward_pass_scaled_to_physical() -> None:
    """Smoke test: one forward pass, c0 in [0,1], physical in [min_sos, max_sos]."""
    from tomo.utils.units import to_physical_sos

    model = SRInitializerNet(use_attention=False)
    model.eval()
    min_sos, max_sos = 0.1, 2.1
    normalized_c_base = normalized_c_base_from_config(min_sos, max_sos, 1.5)
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        delta_pred = model(x)
    c0_normalized = c0_normalized_from_delta(normalized_c_base, delta_pred)
    assert c0_normalized.shape == (2, 1, 128, 128)
    assert c0_normalized.min() >= 0.0 and c0_normalized.max() <= 1.0
    c0_np = c0_normalized[0, 0].numpy()
    c0_phys = to_physical_sos(c0_np, min_sos, max_sos)
    assert c0_phys.shape == (128, 128)
    assert float(np.min(c0_phys)) >= min_sos - 1e-6 and float(np.max(c0_phys)) <= max_sos + 1e-6


def test_stage0_backproj_unet_one_step(tmp_path) -> None:
    """One optimizer step: Stage0BackprojUNet with tiny operator on minimal grid."""
    pytest.importorskip("monai")

    from tomo.geometry.build_backprojection_operator import (
        build_backprojection_operator,
        save_operator_npz,
    )
    from tomo.models.stage0_backproj_unet import Stage0BackprojUNet

    H = W = 32
    x_s = np.array([[0.0, 0.0]], dtype=np.float64)
    x_r = np.array([[float(W - 1), float(H - 1)]], dtype=np.float64)
    rows, cols, vals, coverage, meta = build_backprojection_operator(
        x_s, x_r, H, W, n_samples=128, coord_range=None
    )
    op_path = tmp_path / "op.npz"
    save_operator_npz(
        str(op_path),
        rows,
        cols,
        vals,
        coverage,
        x_s,
        x_r,
        H=H,
        W=W,
        S=meta["S"],
        R=meta["R"],
        n_samples=128,
        scale_by_segment_length=meta["scale_by_segment_length"],
        coord_range_applied=meta["coord_range_applied"],
        coord_range_lo=meta["coord_range_lo"],
        coord_range_hi=meta["coord_range_hi"],
    )

    model = Stage0BackprojUNet(
        str(op_path),
        add_coverage_channel=False,
        output_activation="none",
        unet_channels=(8, 16, 32, 64, 128),
        unet_strides=(2, 2, 2, 2),
    )
    criterion = _get_criterion("smooth_l1")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    normalized_c_base = normalized_c_base_from_config(0.1, 2.1, 1.5)
    B = 2
    anatomy = torch.rand(B, 1, H, W)
    xs = torch.tensor(x_s, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    xr = torch.tensor(x_r, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    batch = {
        "tof_diff_normalized": torch.rand(B, 1, 1, 1),
        "sos_map_normalized": anatomy,
        "x_s": xs,
        "x_r": xr,
    }
    device = torch.device("cpu")
    train_loss, train_mse_recon = _train_epoch(
        model,
        [batch],
        criterion,
        optimizer,
        device,
        normalized_c_base,
        log_first_batch_tensors=False,
    )
    assert isinstance(train_loss, float)
    assert train_mse_recon >= 0.0
