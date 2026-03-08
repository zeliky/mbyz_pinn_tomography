"""Dataset producing TOF/anatomy samples from MATLAB files (legacy-compatible)."""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from tomo.utils.units import to_scaled_sos


def _load_mat(path: str) -> dict[str, Any]:
    """Load a .mat file and return legacy-style dict: x_s, x_r, raw_tof, expanded_tof, sos, tof_maps."""
    mat_data = loadmat(path)
    source_positions = (
        np.array(mat_data["sources"]).transpose() if "sources" in mat_data else np.array([])
    )
    receiver_positions = (
        np.array(mat_data["receivers"]).transpose() if "receivers" in mat_data else np.array([])
    )
    sos = np.array(mat_data["V"]) if "V" in mat_data else np.array([])
    t_obs = np.array(mat_data["t_obs"])
    expanded_tof = (
        np.array(mat_data["exp_t_obs"]) if "exp_t_obs" in mat_data else np.array([])
    )
    tof_maps = np.array(mat_data["tmap"]) if "tmap" in mat_data else np.array([])
    return {
        "x_s": source_positions,
        "x_r": receiver_positions,
        "raw_tof": t_obs,
        "expanded_tof": expanded_tof,
        "sos": sos,
        "tof_maps": tof_maps,
    }


class TofDataset(Dataset):
    """Legacy-compatible dataset: load .mat files, preprocess TOF and SoS, return 8-key sample dict."""

    SAMPLE_KEYS = (
        "anatomy",
        "tof",
        "raw_sos",
        "raw_tof",
        "expanded_tof",
        "tof_maps",
        "x_s",
        "x_r",
    )

    def __init__(
        self,
        modes: Sequence[str],
        *,
        train_path: str,
        validation_path: str,
        test_path: str,
        tof_path: str,
        min_sos: float,
        max_sos: float,
        min_tof: float,
        max_tof: float,
        sources_amount: int = 32,
        receivers_amount: int = 32,
        anatomy_width: int = 128,
        anatomy_height: int = 128,
        index_from_mat_only: bool = False,
    ) -> None:
        super().__init__()
        self._modes = list(modes)
        self._modes_path = {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }
        self._tof_path = tof_path
        self._min_sos = min_sos
        self._max_sos = max_sos
        self._min_tof = min_tof
        self._max_tof = max_tof
        self._sources_amount = sources_amount
        self._receivers_amount = receivers_amount
        self._anatomy_width = anatomy_width
        self._anatomy_height = anatomy_height
        self._index_from_mat_only = index_from_mat_only
        self._files_index: list[dict[str, Any]] = []
        self._build_files_index()

    def _build_files_index(self) -> None:
        if not os.path.isdir(self._tof_path):
            self._files_index = []
            return

        mat_pattern = re.compile(r"ToF(.*)_(\d+)\.mat")

        if self._index_from_mat_only:
            # Build index from .mat files only; no dependency on PNG filenames.
            self._files_index = []
            for file_name in sorted(os.listdir(self._tof_path)):
                match = mat_pattern.match(file_name)
                if match:
                    path = os.path.join(self._tof_path, file_name)
                    self._files_index.append({"anatomy": None, "tof": None, "mat": path})
            return

        patterns = {
            "anatomy": re.compile(r"anatomy(.*)_(\d+)\.png"),
            "tof": re.compile(r"tof(.*)_(\d+)\.png"),
        }
        file_index: dict[str, dict[str, Any]] = {}
        for mode in self._modes:
            base_path = self._modes_path.get(mode)
            if base_path is None or not os.path.isdir(base_path):
                continue
            for file_name in os.listdir(base_path):
                for key, pattern in patterns.items():
                    match = pattern.match(file_name)
                    if match:
                        tumor_id = match.group(2)
                        if tumor_id not in file_index:
                            file_index[tumor_id] = {"anatomy": None, "tof": None, "mat": None}
                        file_index[tumor_id][key] = os.path.join(base_path, file_name)
        for file_name in os.listdir(self._tof_path):
            match = mat_pattern.match(file_name)
            if match:
                tumor_id = match.group(2)
                if tumor_id in file_index:
                    file_index[tumor_id]["mat"] = os.path.join(self._tof_path, file_name)
        self._files_index = [e for e in file_index.values() if e.get("mat") is not None]

    def _load_mat(self, path: str) -> dict[str, Any]:
        return _load_mat(path)

    def __len__(self) -> int:
        return len(self._files_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._files_index[idx]
        mat_data = self._load_mat(entry["mat"])

        raw_tof = mat_data["raw_tof"]
        normalized_tof = (raw_tof - self._min_tof) / (self._max_tof - self._min_tof)
        tof_upsampled = np.repeat(np.repeat(normalized_tof, 4, axis=0), 4, axis=1)
        tof = np.expand_dims(tof_upsampled, axis=0)

        sos = mat_data["sos"]
        normalized_anatomy = to_scaled_sos(sos, self._min_sos, self._max_sos, clamp=True)
        anatomy = np.expand_dims(normalized_anatomy, axis=0)

        return {
            "anatomy": torch.as_tensor(anatomy, dtype=torch.float32),
            "tof": torch.as_tensor(tof, dtype=torch.float32),
            "raw_sos": torch.as_tensor(mat_data["sos"], dtype=torch.float32),
            "raw_tof": torch.as_tensor(mat_data["raw_tof"], dtype=torch.float32),
            "expanded_tof": torch.as_tensor(mat_data["expanded_tof"], dtype=torch.float32),
            "tof_maps": torch.as_tensor(mat_data["tof_maps"], dtype=torch.float32),
            "x_s": torch.as_tensor(mat_data["x_s"], dtype=torch.float32),
            "x_r": torch.as_tensor(mat_data["x_r"], dtype=torch.float32),
        }
