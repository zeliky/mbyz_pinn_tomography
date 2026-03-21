"""Regression: Stage 1 initializer uses tof_diff_normalized; physics uses tof_tumor_raw."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tomo.state.sos_state import SoSState
from tomo.training import stage1_operator_baseline as s1ob
from tomo.training.stage1_operator_baseline import run_stage1_operator_baseline


class _FakeLoader:
    """Mimics DataLoader: each ``iter()`` gets a fresh one-batch iterator."""

    def __init__(self, batch: dict) -> None:
        self._batch = batch

    def __iter__(self):
        return iter([self._batch])


class _FakeDataModule:
    def __init__(self, batch: dict) -> None:
        self._batch = batch

    def setup(self, stage=None) -> None:
        pass

    def val_dataloader(self):
        return _FakeLoader(self._batch)


class _RecordingFakeInitializer:
    """Records Observation ToF; returns flat c_map_2d == c_base_phys (zero delta_c0)."""

    def __init__(
        self,
        *,
        num_nodes: int = 128 * 128,
        c0_fallback: float = 1.5,
        min_sos: float = 0.1,
        max_sos: float = 2.1,
        c_base: float = 1.5,
        device: torch.device | None = None,
        net=None,
        **_: object,
    ) -> None:
        self._device = device or torch.device("cpu")
        self._net = MagicMock()
        self._net.eval = MagicMock()
        self._net.parameters = lambda: []
        self._c_base = float(c_base)
        self.observed_tof: list[torch.Tensor] = []

    def load_checkpoint(self, path: str, device: torch.device | None = None) -> None:
        del path, device

    def __call__(self, observation) -> SoSState:
        self.observed_tof.append(observation.tof_observed.detach().cpu().clone())
        c = torch.full(
            (ANATOMY_H, ANATOMY_W),
            self._c_base,
            dtype=torch.float32,
            device=self._device,
        )
        return SoSState(c_values=c.reshape(-1), step_idx=0, c_map_2d=c)


TOF_H, TOF_W = 64, 64
ANATOMY_H, ANATOMY_W = 128, 128


def _make_batch(*, diff_fill: float, raw_fill: float) -> dict[str, torch.Tensor]:
    tof_diff = torch.full((1, 1, TOF_H, TOF_W), diff_fill, dtype=torch.float32)
    tof_raw = torch.full((1, 1, TOF_H, TOF_W), raw_fill, dtype=torch.float32)
    x_s = torch.zeros((1, 1, 2), dtype=torch.float64)
    x_r = torch.ones((1, 1, 2), dtype=torch.float64) * (TOF_W - 1.0)
    return {
        "tof_diff_normalized": tof_diff,
        "tof_tumor_raw": tof_raw,
        "x_s": x_s,
        "x_r": x_r,
    }


def test_stage1_initializer_uses_diff_search_uses_raw(monkeypatch, tmp_path: Path) -> None:
    diff_v, raw_v = 7.0, 3.0
    batch = _make_batch(diff_fill=diff_v, raw_fill=raw_v)

    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"")

    captured_search_batches: list[dict] = []

    ft_calls = [0]

    def fake_forward_tof(sos_phys, x_s, x_r, scale_factor=1.0):
        del x_s, x_r, scale_factor
        ft_calls[0] += 1
        if ft_calls[0] == 1:
            return np.array([[1.0]], dtype=np.float64)
        h, w = sos_phys.shape[-2], sos_phys.shape[-1]
        assert (h, w) == (ANATOMY_H, ANATOMY_W)
        return np.full((TOF_H, TOF_W), raw_v, dtype=np.float64)

    def fake_stage1a_search(batch, *args, **kwargs):
        del args, kwargs
        captured_search_batches.append(batch)
        return {
            "best_alpha": 1.0,
            "best_loss": 0.0,
            "best_c_map": np.full((ANATOMY_H, ANATOMY_W), 1.5),
            "tof_pred": np.full((TOF_H, TOF_W), raw_v),
            "diagnostics": {},
        }

    fake_init = _RecordingFakeInitializer(device=torch.device("cpu"))

    monkeypatch.setattr(s1ob, "TomographyDataModule", lambda cfg: _FakeDataModule(batch))
    monkeypatch.setattr(s1ob, "UNetInitializer", lambda **kw: fake_init)
    monkeypatch.setattr(s1ob, "forward_tof", fake_forward_tof)
    monkeypatch.setattr(s1ob, "run_stage1a_search", fake_stage1a_search)
    monkeypatch.setattr(s1ob, "run_stage1b_search", MagicMock())

    out_dir = tmp_path / "s1a"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "data": {
            "data_root": ".",
            "normalization_source": "config",
            "min_sos": 0.1,
            "max_sos": 2.1,
            "min_tof": 0.0,
            "max_tof": 1000.0,
            "min_tof_diff": -1.0,
            "max_tof_diff": 1.0,
            "tof_grid_height": TOF_H,
            "tof_grid_width": TOF_W,
            "anatomy_height": ANATOMY_H,
            "anatomy_width": ANATOMY_W,
            "batch_size": 1,
            "num_workers": 0,
        },
        "training": {
            "checkpoint_path": str(ckpt),
            "checkpoint_dir": str(out_dir),
            "checkpoint_dir_1b": str(tmp_path / "s1b"),
            "c_base_phys": 1.5,
            "loader_split": "validation",
        },
    }

    summary = run_stage1_operator_baseline(
        full_config,
        device=torch.device("cpu"),
        max_samples=1,
        run_stage1b=False,
    )
    assert summary["num_samples"] == 1

    expected_diff = _make_batch(diff_fill=diff_v, raw_fill=raw_v)["tof_diff_normalized"]
    for obs_tof in fake_init.observed_tof:
        assert torch.allclose(
            obs_tof.float().cpu(), expected_diff.squeeze(0).float().cpu()
        )
        assert not torch.allclose(
            obs_tof.float().cpu(),
            torch.full_like(obs_tof, raw_v).float().cpu(),
        )

    assert len(captured_search_batches) == 1
    sb = captured_search_batches[0]
    phys = sb["tof_tumor_raw"]
    if phys.dim() == 4:
        phys = phys[0]
    assert torch.allclose(phys.float().cpu(), torch.full_like(phys, raw_v).float().cpu())


def test_stage1_preflight_fails_without_tof_diff_normalized(monkeypatch, tmp_path: Path) -> None:
    raw_v = 3.0
    batch = _make_batch(diff_fill=7.0, raw_fill=raw_v)
    del batch["tof_diff_normalized"]

    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"")

    ft_calls = [0]

    def fake_forward_tof(sos_phys, x_s, x_r, scale_factor=1.0):
        del x_s, x_r, scale_factor
        ft_calls[0] += 1
        if ft_calls[0] == 1:
            return np.array([[1.0]], dtype=np.float64)
        return np.full((32, 32), raw_v, dtype=np.float64)

    monkeypatch.setattr(s1ob, "TomographyDataModule", lambda cfg: _FakeDataModule(batch))
    monkeypatch.setattr(s1ob, "UNetInitializer", lambda **kw: _RecordingFakeInitializer())
    monkeypatch.setattr(s1ob, "forward_tof", fake_forward_tof)

    full_config = {
        "data": {
            "data_root": ".",
            "normalization_source": "config",
            "min_sos": 0.1,
            "max_sos": 2.1,
            "min_tof": 0.0,
            "max_tof": 1000.0,
            "min_tof_diff": -1.0,
            "max_tof_diff": 1.0,
            "tof_grid_height": TOF_H,
            "tof_grid_width": TOF_W,
            "anatomy_height": ANATOMY_H,
            "anatomy_width": ANATOMY_W,
            "batch_size": 1,
            "num_workers": 0,
        },
        "training": {
            "checkpoint_path": str(ckpt),
            "checkpoint_dir": str(tmp_path / "s1a"),
            "c_base_phys": 1.5,
            "loader_split": "validation",
        },
    }

    with pytest.raises(SystemExit) as exc:
        run_stage1_operator_baseline(
            full_config,
            device=torch.device("cpu"),
            max_samples=0,
            run_stage1b=False,
        )
    assert exc.value.code == 1
