#!/usr/bin/env python3
"""CLI: build backprojection operator .npz from dataset .mat files or a single .mat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from tomo.data.dataset import load_comprehensive_mat
from tomo.geometry.build_backprojection_operator import (
    apply_grid_coords,
    build_backprojection_operator,
    save_operator_npz,
    scan_mat_geometry_consistency,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build backprojection operator .npz")
    p.add_argument("--output", "-o", required=True, help="Output path, e.g. artifacts/op.npz")
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=512)
    p.add_argument("--scale_by_segment_length", action="store_true")
    p.add_argument("--mat", type=str, default=None, help="Single .mat path (skips scan)")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Dataset root; scans <split>/mat/*.mat when set with --split",
    )
    p.add_argument("--split", type=str, default="train", choices=("train", "validation", "test"))
    p.add_argument("--max_files", type=int, default=None, help="Max .mat files to scan for consistency")
    p.add_argument("--coord_lo", type=float, default=None, help="coord_range low (with --coord_hi)")
    p.add_argument("--coord_hi", type=float, default=None, help="coord_range high (with --coord_lo)")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()

    coord_range = None
    if args.coord_lo is not None or args.coord_hi is not None:
        if args.coord_lo is None or args.coord_hi is None:
            p.error("Provide both --coord_lo and --coord_hi or neither")
        coord_range = (float(args.coord_lo), float(args.coord_hi))

    if args.mat:
        d = load_comprehensive_mat(args.mat)
        x_s = np.asarray(d["x_s"], dtype=np.float64)
        x_r = np.asarray(d["x_r"], dtype=np.float64)
        if x_s.size == 0 or x_r.size == 0:
            print(f"Missing x_s/x_r in {args.mat}", file=sys.stderr)
            sys.exit(1)
        paths = [args.mat]
    elif args.data_root:
        sub = {"train": "train", "validation": "validate", "test": "test"}[args.split]
        mat_dir = Path(args.data_root) / sub / "mat"
        x_s, x_r, paths = scan_mat_geometry_consistency(
            str(mat_dir), rtol=args.rtol, atol=args.atol, max_files=args.max_files
        )
        print(f"Geometry OK over {len(paths)} file(s); using x_s shape {x_s.shape}, x_r shape {x_r.shape}")
    else:
        p.error("Provide --mat or --data_root")

    rows, cols, vals, coverage, meta = build_backprojection_operator(
        x_s,
        x_r,
        args.H,
        args.W,
        n_samples=args.n_samples,
        scale_by_segment_length=args.scale_by_segment_length,
        coord_range=coord_range,
    )

    x_s_ref, x_r_ref = apply_grid_coords(x_s, x_r, (args.H, args.W), coord_range)

    save_operator_npz(
        args.output,
        rows,
        cols,
        vals,
        coverage,
        x_s_ref,
        x_r_ref,
        H=args.H,
        W=args.W,
        S=meta["S"],
        R=meta["R"],
        n_samples=args.n_samples,
        scale_by_segment_length=args.scale_by_segment_length,
        coord_range_applied=meta["coord_range_applied"],
        coord_range_lo=meta["coord_range_lo"],
        coord_range_hi=meta["coord_range_hi"],
    )
    print(
        f"Wrote {args.output} ({len(paths)} mat(s) checked); "
        f"M={meta['S'] * meta['R']} rays, P={args.H * args.W} pixels"
    )


if __name__ == "__main__":
    main()
