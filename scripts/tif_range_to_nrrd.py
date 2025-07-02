#!/usr/bin/env python
"""
tif_range_to_nrrd.py

Usage:
    python tif_range_to_nrrd.py \
        --indir  /path/to/tiff_slices \
        --outfile labels.nrrd \
        --lower 0 --upper 40
"""

import os
import re
import argparse
import numpy as np
import tifffile as tiff
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stack TIFF slices, threshold, and save as NRRD (SimpleITK)."
    )
    p.add_argument("--indir", required=True, help="Directory containing slice_####.tiff")
    p.add_argument("--outfile", required=True, help="Output NRRD file path (.nrrd)")
    p.add_argument("--lower", type=float, required=True, help="Lower intensity bound (inclusive)")
    p.add_argument("--upper", type=float, required=True, help="Upper intensity bound (inclusive)")
    p.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("dx", "dy", "dz"),
        default=(1.0, 1.0, 1.0),
        help="Voxel spacing along X, Y, Z (default: 1 1 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1. 入力 TIFF を番号順に取得 ---------------------------------------
    pat = re.compile(r"slice[_\-]?(\d+)\.tif{1,2}f?$", re.I)
    files = [f for f in os.listdir(args.indir) if pat.match(f)]
    if not files:
        raise RuntimeError(f"No TIFFs like slice_####.tiff found in {args.indir}")

    files.sort(key=lambda f: int(pat.match(f).group(1)))
    print(f"Found {len(files)} slice(s) under {args.indir}")

    # --- 2. スライス読み込み & 閾値マスク ----------------------------------
    vol = []
    lo, hi = args.lower, args.upper
    for f in files:
        img = tiff.imread(os.path.join(args.indir, f))  # 2-D ndarray
        mask = ((img >= lo) & (img <= hi)).astype(np.uint8)  # 0/1
        vol.append(mask)

    vol = np.stack(vol, axis=0)  # (Z, Y, X)
    print("Volume stacked:", vol.shape, vol.dtype)

    # --- 3. SimpleITK Image へ変換 & メタ設定 ------------------------------
    img_sitk = sitk.GetImageFromArray(vol)          # Z,Y,X → ITK Image
    img_sitk.SetSpacing(tuple(args.spacing))        # (dx, dy, dz)

    # --- 4. 書き出し -------------------------------------------------------
    sitk.WriteImage(img_sitk, args.outfile, True)   # True = gzip 圧縮
    print(f"Saved: {args.outfile}  (compressed NRRD)")


if __name__ == "__main__":
    main()