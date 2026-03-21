"""Build sparse straight-ray backprojection operator (sample-based) and save as .npz.

Measurement index (sparse row): m = s * R + r
Pixel index (sparse column): p = row * W + col

Coordinate convention matches forward_tof / Stage 1: (x, y) = (col, row) in grid space.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

MEASUREMENT_INDEX_FORMULA = "m = s * R + r"
PIXEL_INDEX_FORMULA = "p = row * W + col"
COORD_CONVENTION = "xy_col_row"


def apply_grid_coords(
    x_s: np.ndarray,
    x_r: np.ndarray,
    grid_shape: tuple[int, int],
    coord_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Same remapping as Stage 1/2 `_ensure_grid_coords`: optional affine to [0, W-1] for both axes."""
    if coord_range is None:
        return np.asarray(x_s, dtype=np.float64), np.asarray(x_r, dtype=np.float64)
    lo, hi = coord_range
    _H, W = grid_shape
    scale = (W - 1) / (hi - lo) if hi != lo else 1.0
    xs = (np.asarray(x_s, dtype=np.float64) - lo) * scale
    xr = (np.asarray(x_r, dtype=np.float64) - lo) * scale
    xs = np.clip(xs, 0, W - 1)
    xr = np.clip(xr, 0, W - 1)
    return xs, xr


def _continuous_xy_to_pixel(x: float, y: float, H: int, W: int) -> tuple[int, int]:
    """Map continuous (x,y)=(col,row) to integer pixel (row, col)."""
    col = int(np.floor(x))
    row = int(np.floor(y))
    col = int(np.clip(col, 0, W - 1))
    row = int(np.clip(row, 0, H - 1))
    return row, col


def build_backprojection_operator(
    x_s: np.ndarray,
    x_r: np.ndarray,
    H: int,
    W: int,
    *,
    n_samples: int = 512,
    scale_by_segment_length: bool = False,
    coord_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (rows, cols, vals, coverage, meta_sidecar).

    rows: measurement index m = s*R+r; cols: pixel index p = row*W+col.
    """
    x_s = np.asarray(x_s, dtype=np.float64)
    x_r = np.asarray(x_r, dtype=np.float64)
    if x_s.ndim != 2 or x_s.shape[1] < 2:
        raise ValueError(f"x_s must be (S, 2+), got {x_s.shape}")
    if x_r.ndim != 2 or x_r.shape[1] < 2:
        raise ValueError(f"x_r must be (R, 2+), got {x_r.shape}")
    x_s = x_s[:, :2].copy()
    x_r = x_r[:, :2].copy()

    x_s, x_r = apply_grid_coords(x_s, x_r, (H, W), coord_range)

    S = x_s.shape[0]
    R = x_r.shape[0]
    P = H * W
    rows_list: list[int] = []
    cols_list: list[int] = []
    vals_list: list[float] = []

    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)

    for s in range(S):
        p0 = x_s[s]
        for r in range(R):
            p1 = x_r[r]
            m = s * R + r
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-12:
                row, col = _continuous_xy_to_pixel(float(p0[0]), float(p0[1]), H, W)
                p_idx = row * W + col
                w = 1.0 * (seg_len if scale_by_segment_length else 1.0)
                rows_list.append(m)
                cols_list.append(p_idx)
                vals_list.append(w)
                continue

            counts: dict[int, int] = {}
            for t in ts:
                x = (1.0 - t) * p0[0] + t * p1[0]
                y = (1.0 - t) * p0[1] + t * p1[1]
                row, col = _continuous_xy_to_pixel(float(x), float(y), H, W)
                p_idx = row * W + col
                counts[p_idx] = counts.get(p_idx, 0) + 1

            total = sum(counts.values())
            if total <= 0:
                continue
            for p_idx, c in counts.items():
                frac = c / total
                val = frac * (seg_len if scale_by_segment_length else 1.0)
                rows_list.append(m)
                cols_list.append(p_idx)
                vals_list.append(val)

    rows = np.asarray(rows_list, dtype=np.int64)
    cols = np.asarray(cols_list, dtype=np.int64)
    vals = np.asarray(vals_list, dtype=np.float32)
    coverage = np.zeros(P, dtype=np.float32)
    for c_idx, v in zip(cols, vals, strict=False):
        coverage[c_idx] += v
    meta = {
        "S": S,
        "R": R,
        "H": H,
        "W": W,
        "n_samples": n_samples,
        "scale_by_segment_length": scale_by_segment_length,
        "coord_range_applied": coord_range is not None,
        "coord_range_lo": float(coord_range[0]) if coord_range else np.nan,
        "coord_range_hi": float(coord_range[1]) if coord_range else np.nan,
    }
    return rows, cols, vals, coverage, meta


def _geometry_digest(x_s: np.ndarray, x_r: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x_s.astype(np.float64)).tobytes())
    h.update(np.ascontiguousarray(x_r.astype(np.float64)).tobytes())
    return h.hexdigest()[:32]


def save_operator_npz(
    path: str,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    coverage: np.ndarray,
    x_s_ref: np.ndarray,
    x_r_ref: np.ndarray,
    *,
    H: int,
    W: int,
    S: int,
    R: int,
    n_samples: int,
    scale_by_segment_length: bool,
    coord_range_applied: bool,
    coord_range_lo: float,
    coord_range_hi: float,
    weight_mode: str = "sample_count_normalized",
) -> None:
    """Write operator + reference geometry + metadata strings."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    digest = _geometry_digest(x_s_ref, x_r_ref)
    np.savez_compressed(
        path,
        rows=rows,
        cols=cols,
        vals=vals,
        coverage=coverage,
        x_s_ref=np.asarray(x_s_ref, dtype=np.float64),
        x_r_ref=np.asarray(x_r_ref, dtype=np.float64),
        H=np.array(H, dtype=np.int32),
        W=np.array(W, dtype=np.int32),
        S=np.array(S, dtype=np.int32),
        R=np.array(R, dtype=np.int32),
        n_samples=np.array(n_samples, dtype=np.int32),
        scale_by_segment_length=np.array(scale_by_segment_length),
        coord_range_applied=np.array(coord_range_applied),
        coord_range_lo=np.array(coord_range_lo, dtype=np.float64),
        coord_range_hi=np.array(coord_range_hi, dtype=np.float64),
        geometry_digest=np.array(digest),
        measurement_index_formula=np.array(MEASUREMENT_INDEX_FORMULA),
        pixel_index_formula=np.array(PIXEL_INDEX_FORMULA),
        coord_convention=np.array(COORD_CONVENTION),
        weight_mode=np.array(weight_mode),
    )


def _npz_str(arr: Any, default: str = "") -> str:
    if arr is None:
        return default
    v = np.asarray(arr)
    if v.size == 0:
        return default
    if v.dtype.kind in "SU":
        return str(v.reshape(-1)[0])
    return str(v.item())


def load_operator_arrays(path: str) -> dict[str, Any]:
    """Load .npz operator; return dict of arrays and scalar metadata."""
    raw = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {k: raw[k] for k in raw.files}
    raw.close()

    def _scalar(key: str, default: Any = None) -> Any:
        if key not in out:
            return default
        v = out[key]
        if hasattr(v, "item"):
            return v.item()
        return v

    out["_H"] = int(_scalar("H"))
    out["_W"] = int(_scalar("W"))
    out["_S"] = int(_scalar("S"))
    out["_R"] = int(_scalar("R"))
    out["_n_samples"] = int(_scalar("n_samples", 512))
    out["_coord_range_applied"] = bool(_scalar("coord_range_applied", False))
    out["_coord_range_lo"] = float(_scalar("coord_range_lo", np.nan))
    out["_coord_range_hi"] = float(_scalar("coord_range_hi", np.nan))
    out["_geometry_digest"] = _npz_str(out.get("geometry_digest"), "")
    out["_measurement_formula"] = _npz_str(
        out.get("measurement_index_formula"), MEASUREMENT_INDEX_FORMULA
    )
    out["_pixel_formula"] = _npz_str(out.get("pixel_index_formula"), PIXEL_INDEX_FORMULA)
    out["_coord_convention"] = _npz_str(out.get("coord_convention"), COORD_CONVENTION)
    return out


def scan_mat_geometry_consistency(
    mat_dir: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all *.mat under mat_dir; fail if x_s/x_r differ from first sample.

    Returns canonical (x_s, x_r) from first file and list of scanned paths.
    """
    from tomo.data.dataset import load_comprehensive_mat

    if not os.path.isdir(mat_dir):
        raise FileNotFoundError(f"Not a directory: {mat_dir}")
    names = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))
    if not names:
        raise ValueError(f"No .mat files in {mat_dir}")
    if max_files is not None:
        names = names[: max_files]

    paths = [os.path.join(mat_dir, f) for f in names]
    first = load_comprehensive_mat(paths[0])
    ref_s = np.asarray(first["x_s"], dtype=np.float64)
    ref_r = np.asarray(first["x_r"], dtype=np.float64)
    if ref_s.size == 0 or ref_r.size == 0:
        raise ValueError(f"Missing x_s/x_r in {paths[0]}")

    for p in paths[1:]:
        d = load_comprehensive_mat(p)
        xs = np.asarray(d["x_s"], dtype=np.float64)
        xr = np.asarray(d["x_r"], dtype=np.float64)
        if xs.shape != ref_s.shape or not np.allclose(xs, ref_s, rtol=rtol, atol=atol):
            raise ValueError(f"x_s mismatch vs {paths[0]} in file {p}")
        if xr.shape != ref_r.shape or not np.allclose(xr, ref_r, rtol=rtol, atol=atol):
            raise ValueError(f"x_r mismatch vs {paths[0]} in file {p}")

    return ref_s, ref_r, paths
