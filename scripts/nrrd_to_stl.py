#!/usr/bin/env python3
"""
nrrd_to_stl.py
Usage:
    python nrrd_to_stl.py i seg.nrrd -o ./mesh/seg \
        --decimate-labels 2,3 \
        --reduction 0.85 --min-label 1 --max-label 3 \
        --preserve-topology --no-splitting
"""

import argparse, pathlib, sys, vtk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert NRRD labels to STL, with optional per-label decimation")
    p.add_argument("-i", "--input", required=True, help="input NRRD")
    p.add_argument("-o", "--output-prefix", required=True,
                   help="output STL prefix (prefix_labelX.stl)")
    p.add_argument("-r", "--reduction", type=float, default=0.8,
                   help="vtkDecimatePro target reduction ratio (0.8=80%削減)")
    p.add_argument("--decimate-labels", default=None,
                   help="labels to decimate, e.g. 2,3,5  (default: decimate ALL)")
    p.add_argument("--min-label", type=int, default=None,
                   help="minimum label value to export (default: auto)")
    p.add_argument("--max-label", type=int, default=None,
                   help="maximum label value to export (default: auto)")
    p.add_argument("--preserve-topology", action="store_true",
                   help="Preserve topology in vtkDecimatePro")
    p.add_argument("--no-splitting", action="store_true",
                   help="Disable edge splitting in vtkDecimatePro")
    p.add_argument("--ascii", action="store_true",
                   help="write STL in ASCII (default: binary)")
    return p.parse_args()


# ---------------------------------------------------------------------
def collect_labels(reader, min_lab, max_lab):
    stats = vtk.vtkImageAccumulate()
    stats.SetInputConnection(reader.GetOutputPort())
    stats.Update()
    imin, imax = int(stats.GetMin()[0]), int(stats.GetMax()[0])
    lo = max(min_lab if min_lab is not None else imin, 1)
    hi = max_lab if max_lab is not None else imax
    return [v for v in range(lo, hi + 1)]


def decimate(poly, reduction, preserve, splitting):
    """vtkDecimatePro ラッパー"""
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)

    dec = vtk.vtkDecimatePro()
    dec.SetInputConnection(clean.GetOutputPort())
    dec.SetTargetReduction(reduction)
    if preserve:
        dec.PreserveTopologyOn()
    if splitting:
        dec.SplittingOn()
    else:
        dec.SplittingOff()
    dec.BoundaryVertexDeletionOff()
    dec.Update()
    return dec.GetOutput()


# ---------------------------------------------------------------------
def main():
    args = parse_args()
    prefix = pathlib.Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- NRRD 読み込み（遅延） ----------------------------------------
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(str(args.input))
    reader.Update()

    labels = collect_labels(reader, args.min_label, args.max_label)
    if not labels:
        sys.exit("No valid labels found.")

    # decimate したいラベル集合
    if args.decimate_labels:
        dec_set = {int(v) for v in args.decimate_labels.split(",")}
    else:
        dec_set = set(labels)              # 指定がなければ全部デシメーション

    # --- ラベルごとに抽出 & (オプションで) デシメーション --------------
    for lab in labels:
        fe = vtk.vtkDiscreteFlyingEdges3D()
        fe.SetInputConnection(reader.GetOutputPort())
        fe.SetValue(0, lab)
        fe.Update()
        poly = fe.GetOutput()
        if poly.GetNumberOfCells() == 0:
            print(f"[skip] label {lab} – empty")
            continue

        if lab in dec_set:
            poly_out = decimate(poly,
                                reduction=args.reduction,
                                preserve=args.preserve_topology,
                                splitting=not args.no_splitting)
            tri_before, tri_after = poly.GetNumberOfCells(), poly_out.GetNumberOfCells()
            ratio = tri_after / tri_before
            msg = f"decimated {tri_before}→{tri_after} ({ratio:.1%})"
        else:
            poly_out = poly
            msg = "no decimation"

        writer = vtk.vtkSTLWriter()
        writer.SetInputData(poly_out)
        writer.SetFileName(f"{prefix}_label{lab}.stl")
        writer.SetFileTypeToASCII() if args.ascii else writer.SetFileTypeToBinary()
        writer.Write()
        print(f"[OK] label {lab}: {msg}")


if __name__ == "__main__":
    main()