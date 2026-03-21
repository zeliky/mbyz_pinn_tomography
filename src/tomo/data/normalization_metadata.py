"""Load `normalization.json` from dataset root (written by MATLAB) and merge into data config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "normalization.json"

BOUND_KEYS = ("min_sos", "max_sos", "min_tof", "max_tof")


def normalization_manifest_path(data_root: str) -> Path:
    return Path(data_root) / MANIFEST_NAME


def load_normalization_bounds(data_root: str) -> dict[str, float]:
    """Read normalization.json; return min_sos, max_sos, min_tof, max_tof as floats."""
    path = normalization_manifest_path(data_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset normalization manifest not found: {path}. "
            "Run MATLAB runProduction1 (writes normalization.json) or set normalization_source: config."
        )
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    out: dict[str, float] = {}
    for k in BOUND_KEYS:
        if k not in raw:
            raise KeyError(f"{MANIFEST_NAME} missing required key {k!r}")
        out[k] = float(raw[k])
    if out["max_sos"] <= out["min_sos"]:
        raise ValueError(
            f"Invalid SOS bounds in {path}: max_sos ({out['max_sos']}) must exceed min_sos ({out['min_sos']})."
        )
    if out["max_tof"] <= out["min_tof"]:
        raise ValueError(
            f"Invalid ToF bounds in {path}: max_tof ({out['max_tof']}) must exceed min_tof ({out['min_tof']})."
        )
    return out


def resolve_data_config(data_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of data_config with min/max SOS and ToF set per normalization_source.

    - normalization_source ``config``: keep YAML values only (ignore manifest).
    - ``dataset``: require normalization.json under data_root.
    - ``auto`` (default): use manifest if present, else keep YAML.
    """
    c = dict(data_config)
    source = str(c.get("normalization_source", "auto")).lower()
    data_root = c.get("data_root")
    if not data_root:
        return c
    root_path = Path(data_root)
    manifest = root_path / MANIFEST_NAME

    if source == "config":
        return c
    if source == "dataset":
        bounds = load_normalization_bounds(str(root_path))
        c.update(bounds)
        return c
    if source == "auto":
        if manifest.is_file():
            bounds = load_normalization_bounds(str(root_path))
            c.update(bounds)
        return c
    raise ValueError(
        f"Unknown normalization_source: {source!r}. Use 'auto', 'dataset', or 'config'."
    )
