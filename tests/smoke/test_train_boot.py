"""Smoke test: train script boots with dummy config."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_train_boot() -> None:
    """Run scripts.train with overrides to verify boot (no real data)."""
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "train.py"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.train"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "Train boot OK" in result.stdout
