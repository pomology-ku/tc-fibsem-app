import os
import argparse
import numpy as np
import SimpleITK as sitk
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
import tifffile
import math

def _rigid_matrix(data):
    """RigidModel2D → 3, 4, 6 パラメータの全てを numpy 3×3 に変換"""
    if len(data) == 3:          # theta  tx  ty
        theta, tx, ty = data
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[ c, -s, tx],
                         [ s,  c, ty],
                         [ 0,  0,  1 ]], dtype=float)
    elif len(data) == 4:        # cosθ  sinθ  tx  ty
        c, s, tx, ty = data
        return np.array([[ c, -s, tx],
                         [ s,  c, ty],
                         [ 0,  0,  1 ]], dtype=float)
    elif len(data) == 6:        # full 2×3 行列
        m00, m01, m02, m10, m11, m12 = data
        return np.array([[m00, m01, m02],
                         [m10, m11, m12],
                         [ 0 ,  0 ,  1 ]], dtype=float)
    else:
        raise ValueError(f"Unsupported RigidModel2D param length: {len(data)}")

def get_affine_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    A = np.eye(3, dtype=float)
    for t in root.findall(".//iict_transform"):
        cls  = t.attrib["class"]
        data = list(map(float, t.attrib["data"].split()))

        if "RigidModel2D" in cls:
            A = _rigid_matrix(data) @ A           # 1) 回転＋並進
        elif "TranslationModel2D" in cls:
            if len(data) != 2:
                raise ValueError("TranslationModel2D expects 2 params")
            tx, ty = data
            A = np.array([[1, 0, tx],
                          [0, 1, ty],
                          [0, 0, 1 ]], dtype=float) @ A  # 2) 追加平行移動
        else:
            raise NotImplementedError(f"Unsupported transform: {cls}")

    return A

def main(args):
    label_path = args.label_nrrd
    xml_dir = args.xml_dir
    output_path = args.output
    registered_stack_path = args.registered_stack

    # --- NRRD読み込み ---
    nrrd_img = sitk.ReadImage(label_path)
    arr = sitk.GetArrayFromImage(nrrd_img)  # (Z, Y, X)
    Z, H0, W0 = arr.shape
    print(f"Input label NRRD: {label_path} (shape: {Z} slices, {H0} height, {W0} width)")

    # --- 登録後のmulti-TIFFから (H', W') を取得 ---
    reg_stack = tifffile.imread(registered_stack_path)
    if reg_stack.ndim != 3:
        raise ValueError(f"Registered stack must be 3D (Z,H,W), but got shape: {reg_stack.shape}")
    print(f"Registered stack size: {reg_stack.shape} (Z, H, W)")

    M_list = []
    xmin = ymin =  np.inf
    xmax = ymax = -np.inf

    corners = np.array([[0,     0,    1],
                        [W0-1,  0,    1],
                        [0,     H0-1, 1],
                        [W0-1,  H0-1, 1]], dtype=float).T  # 3×4
    for z in range(Z):
        xml_file = os.path.join(args.xml_dir, f"slice_{z+1:04d}.xml")
        if not os.path.exists(xml_file):
            raise FileNotFoundError(xml_file)

        M = get_affine_from_xml(xml_file)
        M_list.append(M)

        warped = (M @ corners)[:2]          # 2×4
        xmin = min(xmin, warped[0].min())
        xmax = max(xmax, warped[0].max())
        ymin = min(ymin, warped[1].min())
        ymax = max(ymax, warped[1].max())

    # ---------- ③ キャンバス拡張オフセット ----------
    off_x, off_y = -xmin, -ymin
    W_new = int(np.ceil(xmax - xmin + 1))
    H_new = int(np.ceil(ymax - ymin + 1))
    print(f"Canvas size : (H={H_new}, W={W_new})  offset=({off_x:.2f},{off_y:.2f})")

    T_offset = np.array([[1, 0, off_x],
                         [0, 1, off_y],
                         [0, 0,   1 ]], dtype=float)

    # ---------- ④ 出力配列を確保 ----------
    reg_arr = np.zeros((Z, H_new, W_new), dtype=np.uint8)

    for z in tqdm(range(Z), desc="Warping slices"):
        M_fwd = T_offset @ M_list[z]
        M_cv  = M_fwd[:2].astype(np.float32)

        # print(arr[z].sum(), f"before warping slice {z+1}")
        reg_arr[z] = cv2.warpAffine(
            arr[z].astype(np.uint8),
            M_cv,
            (W_new, H_new),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )
        # print(reg_arr[z].sum(), f"after warping slice {z+1}")

    out_img = sitk.GetImageFromArray(reg_arr)
    out_img = sitk.GetImageFromArray(reg_arr)
    out_img.SetSpacing(nrrd_img.GetSpacing())  
    out_img.SetOrigin(nrrd_img.GetOrigin()) 
    out_img.SetDirection(nrrd_img.GetDirection()) 
    sitk.WriteImage(out_img, output_path)
    print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply RVSMT slice-wise XML transforms to label NRRD"
    )
    parser.add_argument("--label_nrrd", required=True, help="Path to input label.nrrd")
    parser.add_argument("--xml_dir", required=True, help="Directory containing slice_XXXX.xml files")
    parser.add_argument("--registered_stack", required=True, help="Path to registered multi-TIFF stack")
    parser.add_argument("--output", required=True, help="Path to output registered_label.nrrd")

    args = parser.parse_args()
    main(args)

