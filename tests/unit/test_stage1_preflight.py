"""Unit tests for Stage 1 preflight: checkpoint missing, loader empty."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for imports when running tests from repo root
_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tomo.training.stage1_operator_baseline import run_stage1_operator_baseline


def test_stage1_preflight_fails_when_checkpoint_missing() -> None:
    """Preflight should exit with clear message when checkpoint_path does not exist."""
    full_config = {
        "data": {
            "data_root": ".",
            "min_sos": 0.1,
            "max_sos": 2.1,
            "min_tof": 0.0,
            "max_tof": 1000.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "training": {
            "checkpoint_path": "/nonexistent/path/to/checkpoint.pt",
            "c_base_phys": 1.5,
            "loader_split": "validation",
        },
    }
    with pytest.raises(SystemExit) as exc_info:
        run_stage1_operator_baseline(
            full_config,
            device=None,
            max_samples=0,
            run_stage1b=False,
        )
    assert exc_info.value.code == 1
