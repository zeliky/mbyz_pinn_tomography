"""Dataset: comprehensive `.mat` samples from `tof_generator` (struct `D`) under `data_root/{split}/mat/`."""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from tomo.utils.units import to_scaled_sos

_MODE_TO_SUBDIR = {"train": "train", "validation": "validate", "test": "test"}

# Struct D fields written by save_comprehensive_sample.m (full save).
D_NUMERIC_KEYS = (
    "tof_tumor_raw",
    "tof_healthy_raw",
    "tof_diff_raw",
    "sos_map",
    "sos_healthy_base",
    "tumor_mask",
    "x_s",
    "x_r",
)
D_OPTIONAL_MAP_KEYS = ("tof_maps_tumor", "tof_maps_healthy", "tof_maps_diff")

# Populated in __getitem__ (not stored in .mat).
DERIVED_KEYS = ("sos_map_normalized", "tof_tumor_normalized_grid")

SAMPLE_KEYS = D_NUMERIC_KEYS + D_OPTIONAL_MAP_KEYS + DERIVED_KEYS


def _strip_mat_meta(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _as_float_array(v: Any) -> np.ndarray:
    if v is None:
        return np.zeros((0, 0), dtype=np.float64)
    if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.item()
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    return arr


def _maybe_transpose_coords(arr: np.ndarray) -> np.ndarray:
    """Use [N, 2] layout for sensor/receiver coordinates."""
    if arr.size == 0:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        return arr.T
    return arr


def load_comprehensive_mat(path: str) -> dict[str, np.ndarray]:
    """Load one comprehensive sample .mat; return numpy arrays keyed like struct D."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    mat_data = _strip_mat_meta(raw)
    out: dict[str, np.ndarray] = {}
    for key in D_NUMERIC_KEYS:
        if key not in mat_data:
            out[key] = np.zeros((0, 0), dtype=np.float64)
            continue
        arr = _as_float_array(mat_data[key])
        if key in ("x_s", "x_r"):
            arr = _maybe_transpose_coords(arr)
        out[key] = arr
    for key in D_OPTIONAL_MAP_KEYS:
        if key not in mat_data:
            out[key] = np.zeros((0, 0), dtype=np.float64)
        else:
            out[key] = _as_float_array(mat_data[key])
    return out


class TofDataset(Dataset):
    """Index `mat/*.mat` under `data_root/{train|validate|test}/mat`; batch keys match struct D + derived tensors."""

    SAMPLE_KEYS = SAMPLE_KEYS

    def __init__(
        self,
        modes: Sequence[str],
        *,
        data_root: str,
        min_sos: float,
        max_sos: float,
        min_tof: float,
        max_tof: float,
        sources_amount: int = 32,
        receivers_amount: int = 32,
        anatomy_width: int = 128,
        anatomy_height: int = 128,
    ) -> None:
        super().__init__()
        self._modes = list(modes)
        self._data_root = data_root
        self._min_sos = min_sos
        self._max_sos = max_sos
        self._min_tof = min_tof
        self._max_tof = max_tof
        self._sources_amount = sources_amount
        self._receivers_amount = receivers_amount
        self._anatomy_width = anatomy_width
        self._anatomy_height = anatomy_height
        self._files_index: list[dict[str, Any]] = []
        self._build_files_index()

    def _mat_dir(self, mode: str) -> str:
        sub = _MODE_TO_SUBDIR[mode]
        return os.path.join(self._data_root, sub, "mat")

    def _previews_dir(self, mode: str) -> str:
        sub = _MODE_TO_SUBDIR[mode]
        return os.path.join(self._data_root, sub, "previews")

    def _build_files_index(self) -> None:
        self._files_index = []
        if not os.path.isdir(self._data_root):
            return
        for mode in self._modes:
            mat_dir = self._mat_dir(mode)
            prev_dir = self._previews_dir(mode)
            if not os.path.isdir(mat_dir):
                continue
            for file_name in sorted(os.listdir(mat_dir)):
                if not file_name.endswith(".mat"):
                    continue
                sample_id = file_name[: -len(".mat")]
                mat_path = os.path.join(mat_dir, file_name)
                entry: dict[str, Any] = {
                    "sample_id": sample_id,
                    "mat": mat_path,
                    "anatomy_no_tumors_png": None,
                    "anatomy_with_tumors_png": None,
                    "tof_png": None,
                    "tof_diff_png": None,
                }
                if os.path.isdir(prev_dir):
                    p1 = os.path.join(prev_dir, f"anatomy_noTumors_{sample_id}.png")
                    p2 = os.path.join(prev_dir, f"anatomy_withTumors_{sample_id}.png")
                    p3 = os.path.join(prev_dir, f"tof_{sample_id}.png")
                    p4 = os.path.join(prev_dir, f"tof_diff_{sample_id}.png")
                    if os.path.isfile(p1):
                        entry["anatomy_no_tumors_png"] = p1
                    if os.path.isfile(p2):
                        entry["anatomy_with_tumors_png"] = p2
                    if os.path.isfile(p3):
                        entry["tof_png"] = p3
                    if os.path.isfile(p4):
                        entry["tof_diff_png"] = p4
                self._files_index.append(entry)

    def __len__(self) -> int:
        return len(self._files_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._files_index[idx]
        mat_np = load_comprehensive_mat(entry["mat"])
        tt = mat_np["tof_tumor_raw"]
        if tt.size == 0:
            raise ValueError(
                f"Missing tof_tumor_raw in {entry['mat']} (need full save; mat_minimal is unsupported for training)."
            )
        denom = self._max_tof - self._min_tof
        if denom <= 0:
            raise ValueError("max_tof must be greater than min_tof")
        normalized_tof = (tt - self._min_tof) / denom
        tof_upsampled = np.repeat(np.repeat(normalized_tof, 4, axis=0), 4, axis=1)
        tof_tumor_normalized_grid = np.expand_dims(tof_upsampled, axis=0)

        sos = mat_np["sos_map"]
        if sos.size == 0:
            raise ValueError(f"Missing sos_map in {entry['mat']}")
        sos_norm = to_scaled_sos(sos, self._min_sos, self._max_sos, clamp=True)
        sos_map_normalized = np.expand_dims(sos_norm, axis=0)

        out: dict[str, Any] = {}
        for key in D_NUMERIC_KEYS + D_OPTIONAL_MAP_KEYS:
            out[key] = torch.as_tensor(mat_np[key], dtype=torch.float32)
        out["sos_map_normalized"] = torch.as_tensor(sos_map_normalized, dtype=torch.float32)
        out["tof_tumor_normalized_grid"] = torch.as_tensor(
            tof_tumor_normalized_grid, dtype=torch.float32
        )
        return out
