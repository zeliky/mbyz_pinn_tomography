"""Unit tests for Stage 0 initializer: SR net shapes, residual target, one-step training."""

import torch

from tomo.initializers.unet_initializer import (
    SRInitializerNet,
    c0_normalized_from_delta,
    delta_target_from_anatomy,
    normalized_c_base_from_config,
)
from tomo.training.stage0_initializer import (
    _ensure_raw_tof_4d,
    _get_criterion,
    _train_epoch,
)


def test_sr_initializer_net_baseline_shapes() -> None:
    """SRInitializerNet (no attention): [B, 1, 32, 32] -> delta_c0 [B, 1, 128, 128]."""
    model = SRInitializerNet(use_attention=False)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_sr_initializer_net_attention_shapes() -> None:
    """SRInitializerNet (with attention): same shapes."""
    model = SRInitializerNet(use_attention=True)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ensure_raw_tof_4d() -> None:
    """_ensure_raw_tof_4d adds channel dim when raw_tof is [B, 32, 32]."""
    x3 = torch.randn(3, 32, 32)
    x4 = _ensure_raw_tof_4d(x3)
    assert x4.shape == (3, 1, 32, 32)
    x4_already = torch.randn(3, 1, 32, 32)
    assert _ensure_raw_tof_4d(x4_already).shape == (3, 1, 32, 32)


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
        "raw_tof": torch.randn(2, 32, 32),
        "anatomy": anatomy,
    }
    device = torch.device("cpu")
    train_loss, train_mse_recon = _train_epoch(
        model,
        [batch],
        criterion,
        optimizer,
        device,
        normalized_c_base,
    )
    assert isinstance(train_loss, float)
    assert isinstance(train_mse_recon, float)
    assert train_loss >= 0.0


def test_stage0_forward_pass_scaled_to_physical() -> None:
    """Smoke test: one forward pass, c0 in [0,1], physical in [min_sos, max_sos]."""
    import numpy as np

    from tomo.utils.units import to_physical_sos

    model = SRInitializerNet(use_attention=False)
    model.eval()
    min_sos, max_sos = 0.1, 2.1
    normalized_c_base = normalized_c_base_from_config(min_sos, max_sos, 1.5)
    x = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        delta_pred = model(x)
    c0_normalized = c0_normalized_from_delta(normalized_c_base, delta_pred)
    assert c0_normalized.shape == (2, 1, 128, 128)
    assert c0_normalized.min() >= 0.0 and c0_normalized.max() <= 1.0
    c0_np = c0_normalized[0, 0].numpy()
    c0_phys = to_physical_sos(c0_np, min_sos, max_sos)
    assert c0_phys.shape == (128, 128)
    assert float(np.min(c0_phys)) >= min_sos - 1e-6 and float(np.max(c0_phys)) <= max_sos + 1e-6
