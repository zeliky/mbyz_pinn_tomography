"""Unit tests for Stage 0 initializer: SR net shapes and one-step training."""

import torch

from tomo.initializers.unet_initializer import SRInitializerNet
from tomo.training.stage0_initializer import (
    _ensure_raw_tof_4d,
    _get_criterion,
    _train_epoch,
)


def test_sr_initializer_net_baseline_shapes() -> None:
    """SRInitializerNet (no attention): [B, 1, 32, 32] -> [B, 1, 128, 128]."""
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


def test_stage0_one_step_backward() -> None:
    """One training step: dummy batch, forward, loss, backward."""
    model = SRInitializerNet(use_attention=False)
    criterion = _get_criterion("smooth_l1")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch = {
        "raw_tof": torch.randn(2, 32, 32),
        "anatomy": torch.rand(2, 1, 128, 128),
    }
    device = torch.device("cpu")
    train_loss, train_mse = _train_epoch(
        model,
        [batch],
        criterion,
        optimizer,
        device,
    )
    assert isinstance(train_loss, float)
    assert isinstance(train_mse, float)
    assert train_loss >= 0.0
