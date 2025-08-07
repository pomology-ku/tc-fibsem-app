#!/usr/bin/env python
"""
tiff_tiler.py

Split a 3-D multi-TIFF (image) and its label TIFF into arbitrary tiles along
x, y, z axes and save each pair as individual TIFF stacks.

Example:
    python tiff_tiler.py \
        --img      sample_img.tif \
        --label    sample_lbl.tif \
        --outdir   tiles_out \
        --split-x  4 --split-y 4 --split-z 2 \
        --compress lzw
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tile a 3-D multi-TIFF (and its label) along x, y, z."
    )
    p.add_argument("--img",      required=True, help="Input image TIFF (multi-page)")
    p.add_argument("--label",    required=True, help="Input label TIFF (multi-page)")
    p.add_argument("--outdir",   required=True, help="Destination directory")
    p.add_argument("--split-x",  type=int, default=1, help="#tiles along X (width)")
    p.add_argument("--split-y",  type=int, default=1, help="#tiles along Y (height)")
    p.add_argument("--split-z",  type=int, default=1, help="#tiles along Z (depth)")
    p.add_argument(
        "--compress",
        choices=["none", "lzw", "deflate", "zstd"],
        default="none",
        help="TIFF compression (default: none)",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------


def compute_slices(size: int, n_split: int) -> list[slice]:
    """Return a list of slice() objects that split an axis into n_split chunks."""
    step = size // n_split
    remainder = size % n_split
    slices = []
    start = 0
    for i in range(n_split):
        end = start + step + (1 if i < remainder else 0)
        slices.append(slice(start, end))
        start = end
    return slices


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -- 1. Read TIFF stacks ---------------------------------------------------
    img = tiff.imread(args.img)    # shape: (Z, Y, X) or (Z, Y, X, C)
    lbl = tiff.imread(args.label)  # shape must match img (except channel)

    if img.shape[:3] != lbl.shape[:3]:
        raise ValueError("Image and label TIFF must have identical Z, Y, X sizes")

    has_channel = img.ndim == 4
    z_size, y_size, x_size = img.shape[:3]

    # -- 2. Prepare slices for tiling -----------------------------------------
    x_slices = compute_slices(x_size, args.split_x)
    y_slices = compute_slices(y_size, args.split_y)
    z_slices = compute_slices(z_size, args.split_z)

    # -- 3. Save each tile pair ----------------------------------------------
    compress_opt = {} if args.compress == "none" else {"compression": args.compress}

    tile_idx = 0
    for kz, z_sl in enumerate(z_slices):
        for ky, y_sl in enumerate(y_slices):
            for kx, x_sl in enumerate(x_slices):
                if has_channel:
                    img_tile = img[z_sl, y_sl, x_sl, :]
                else:
                    img_tile = img[z_sl, y_sl, x_sl]
                lbl_tile = lbl[z_sl, y_sl, x_sl]

                tile_name = f"tile_z{kz:02d}_y{ky:02d}_x{kx:02d}"
                img_path = outdir / f"{tile_name}_img.tif"
                lbl_path = outdir / f"{tile_name}_lbl.tif"

                tiff.imwrite(img_path, img_tile, **compress_opt)
                tiff.imwrite(lbl_path, lbl_tile, **compress_opt)
                tile_idx += 1

    print(
        f"✓ Saved {tile_idx} tile pairs to '{outdir}' "
        f"({args.split_z}×{args.split_y}×{args.split_x})"
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
