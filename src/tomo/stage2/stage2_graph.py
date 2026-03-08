"""Build hybrid graph for Stage 2: source, receivers, ROI, support nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from tomo.stage2.roi_builder import ROIMasks


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class Stage2GraphInput:
    """Inputs for building Stage 2 graph."""

    c_stage1: np.ndarray  # (H, W)
    delta_stage1: np.ndarray  # (H, W)
    backproj_map: np.ndarray  # (H, W)
    x_s: np.ndarray  # (1, 2) active source
    x_r: np.ndarray  # (R, 2) receivers
    tof_obs: np.ndarray  # (R,) tof for this source
    tof_pred: np.ndarray  # (R,)
    tof_error: np.ndarray  # (R,)
    roi_masks: ROIMasks


def _pos_from_coords(coords: np.ndarray) -> np.ndarray:
    """Convert (row, col) coords to (x, y) = (col, row) positions."""
    return np.column_stack([coords[:, 1], coords[:, 0]]).astype(np.float32)


def _build_edges(
    src_pos: np.ndarray,
    rcv_pos: np.ndarray,
    roi_pos: np.ndarray,
    support_pos: np.ndarray,
    roi_k: int = 20,
    rcv_k: int = 15,
    spatial_k: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge list and return (edges, edge_distances)."""
    n_src = 1
    n_rcv = rcv_pos.shape[0]
    n_roi = roi_pos.shape[0]
    n_support = support_pos.shape[0]

    offset_rcv = n_src
    offset_roi = offset_rcv + n_rcv
    offset_support = offset_roi + n_roi

    all_pos = np.vstack([src_pos, rcv_pos, roi_pos, support_pos])
    edges_set: set[tuple[int, int]] = set()

    def add_edge(i: int, j: int):
        if i != j:
            edges_set.add((i, j))
            edges_set.add((j, i))

    tree_roi = cKDTree(roi_pos)
    tree_all_spatial = cKDTree(np.vstack([roi_pos, support_pos]))

    for i in range(n_roi):
        idx_roi = offset_roi + i
        _, nbrs = tree_all_spatial.query(roi_pos[i], k=min(spatial_k, n_roi + n_support))
        nbrs = np.atleast_1d(nbrs)
        for j in nbrs:
            j = int(j)
            if j == i:
                continue
            if j < n_roi:
                j_global = offset_roi + j
            else:
                j_global = offset_support + (j - n_roi)
            add_edge(idx_roi, j_global)

    for i in range(n_support):
        idx_sup = offset_support + i
        _, nbrs = tree_all_spatial.query(support_pos[i], k=min(spatial_k, n_roi + n_support))
        nbrs = np.atleast_1d(nbrs)
        for j in nbrs:
            j = int(j)
            if j < n_roi:
                j_global = offset_roi + j
                add_edge(idx_sup, j_global)
            elif j != i + n_roi:
                j_global = offset_support + (j - n_roi)
                add_edge(idx_sup, j_global)

    if n_roi > 0:
        _, src_to_roi = tree_roi.query(src_pos[0], k=min(roi_k, n_roi))
        src_to_roi = np.atleast_1d(src_to_roi)
        for j in src_to_roi:
            add_edge(0, offset_roi + int(j))

    for r in range(n_rcv):
        idx_rcv = offset_rcv + r
        if n_roi > 0:
            _, roi_to_rcv = tree_roi.query(rcv_pos[r], k=min(rcv_k, n_roi))
            roi_to_rcv = np.atleast_1d(roi_to_rcv)
            for j in roi_to_rcv:
                add_edge(offset_roi + int(j), idx_rcv)
        add_edge(0, idx_rcv)

    edges_arr = np.array(list(edges_set), dtype=np.int64)
    row, col = edges_arr[:, 0], edges_arr[:, 1]
    dists_arr = np.array(
        [float(np.linalg.norm(all_pos[r] - all_pos[c])) for r, c in zip(row, col)],
        dtype=np.float32,
    )
    return edges_arr, dists_arr


def build_stage2_graph(
    inp: Stage2GraphInput,
    *,
    device: torch.device | None = None,
    roi_k: int = 20,
    rcv_k: int = 15,
    spatial_k: int = 9,
) -> Data:
    """Build PyG Data for one source + receivers + ROI + support.

    Node order: [source, receivers..., ROI..., support...]
    Node features: [x, y, c_stage1, delta_stage1, backproj, role, tof_obs?, tof_pred?, tof_error?, dist_to_src?]
    """
    roi_masks = inp.roi_masks
    roi_coords = roi_masks.roi_coords
    support_coords = roi_masks.support_coords

    roi_pos = _pos_from_coords(roi_coords)
    support_pos = _pos_from_coords(support_coords)

    src_pos = np.asarray(inp.x_s, dtype=np.float32).reshape(1, 2)
    rcv_pos = np.asarray(inp.x_r, dtype=np.float32)

    H, W = inp.c_stage1.shape
    n_src = 1
    n_rcv = rcv_pos.shape[0]
    n_roi = roi_pos.shape[0]
    n_support = support_pos.shape[0]
    n_total = n_src + n_rcv + n_roi + n_support

    edges_arr, edge_dists = _build_edges(
        src_pos, rcv_pos, roi_pos, support_pos,
        roi_k=roi_k, rcv_k=rcv_k, spatial_k=spatial_k,
    )

    src_xy = src_pos[0]
    dist_to_src = np.linalg.norm(roi_pos - src_xy, axis=1)
    dist_to_src_support = np.linalg.norm(support_pos - src_xy, axis=1)

    c_roi = inp.c_stage1[roi_coords[:, 0], roi_coords[:, 1]]
    delta_roi = inp.delta_stage1[roi_coords[:, 0], roi_coords[:, 1]]
    backproj_roi = inp.backproj_map[roi_coords[:, 0], roi_coords[:, 1]]

    c_support = inp.c_stage1[support_coords[:, 0], support_coords[:, 1]]
    delta_support = inp.delta_stage1[support_coords[:, 0], support_coords[:, 1]]

    role_src = 1.0
    role_rcv = 2.0
    role_roi = 3.0
    role_support = 4.0

    scale = max(H, W)
    x_list = []

    x_src = np.array([
        src_xy[0] / scale, src_xy[1] / scale,
        0.0, 0.0, 0.0, role_src,
        float(np.mean(inp.tof_error)), 0.0, 0.0, 0.0
    ], dtype=np.float32)
    x_list.append(x_src)

    for r in range(n_rcv):
        xr = rcv_pos[r, 0] / scale
        yr = rcv_pos[r, 1] / scale
        x_rcv = np.array([
            xr, yr, 0.0, 0.0, 0.0, role_rcv,
            inp.tof_obs[r], inp.tof_pred[r], inp.tof_error[r], 0.0
        ], dtype=np.float32)
        x_list.append(x_rcv)

    for i in range(n_roi):
        xr = roi_pos[i, 0] / scale
        yr = roi_pos[i, 1] / scale
        x_roi = np.array([
            xr, yr, c_roi[i], delta_roi[i], backproj_roi[i], role_roi,
            0.0, 0.0, 0.0, dist_to_src[i] / scale
        ], dtype=np.float32)
        x_list.append(x_roi)

    for i in range(n_support):
        xr = support_pos[i, 0] / scale
        yr = support_pos[i, 1] / scale
        x_sup = np.array([
            xr, yr, c_support[i], delta_support[i], 0.0, role_support,
            0.0, 0.0, 0.0, dist_to_src_support[i] / scale
        ], dtype=np.float32)
        x_list.append(x_sup)

    x = torch.tensor(np.vstack(x_list), dtype=torch.float32)
    edge_index = torch.tensor(edges_arr.T, dtype=torch.long)
    edge_attr = torch.tensor(edge_dists, dtype=torch.float32).unsqueeze(-1)
    pos = torch.tensor(np.vstack([src_pos, rcv_pos, roi_pos, support_pos]), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=n_total,
        n_src=n_src,
        n_rcv=n_rcv,
        n_roi=n_roi,
        n_support=n_support,
        roi_node_start=n_src + n_rcv,
        roi_node_end=n_src + n_rcv + n_roi,
    )

    if device is not None:
        data = data.to(device)

    return data
