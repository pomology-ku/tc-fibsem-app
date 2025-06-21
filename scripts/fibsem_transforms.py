# scripts/fibsem_transforms.py
from monai.transforms import MapTransform
import numpy as np
import tifffile as tiff
import torch
from pathlib import Path
from typing import Union, Sequence

class ExtractValidSlicesd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        label = d[self.keys[0]]
        if label.shape[0] != 1:
            raise ValueError("Only single-channel labels supported")
        label_3d = label[0]
        valid_slices = [i for i in range(label_3d.shape[-1]) if np.any(np.isin(label_3d[..., i], [1, 2]))]
        if not valid_slices:
            return None  # skip
        idx = np.random.choice(valid_slices)
        d[self.keys[0]] = label[..., idx]
        d[self.keys[1]] = d[self.keys[1]][..., idx]
        return d

class SelectSliceByKeyd(MapTransform):
    """
    dict["slice_idx"] で指定された Z 面(軸0)を切り出す。
      - 入力が ndarray/Tensor : (Z,H,W) or (C,Z,H,W)
      - 入力が str/Path       : multi-TIFF ファイル
    出力は (H,W) または (C,H,W)
    """
    def __call__(self, data):
        d = dict(data)
        z = int(d["slice_idx"])

        for k in self.keys:
            src = d[k]

            # ---------- ケース A: パス -----------------------------
            if isinstance(src, (str, Path)):
                with tiff.TiffFile(src) as tf:
                    z_clamp = max(0, min(len(tf.pages) - 1, z))
                    arr = tf.pages[z_clamp].asarray()          # (H,W)
                d[k] = torch.as_tensor(arr)                    # Tensor 化
                continue

            # ---------- ケース B: メモリ上 -------------------------
            is_torch = torch.is_tensor(src)
            arr = src.cpu().numpy() if is_torch else src

            if arr.ndim == 3:          # (Z,H,W)
                out = arr[z]
            elif arr.ndim == 4:        # (C,Z,H,W)
                out = arr[:, z]
            else:
                raise RuntimeError(f"{k}: unexpected shape {arr.shape}")

            if is_torch:
                out = torch.as_tensor(out, dtype=src.dtype, device=src.device)
            d[k] = out

        return d

class StackNeighborSlicesd(MapTransform):
    """
    入力:
      - ndarray / Tensor: (Z,H,W) or (1,Z,H,W)
      - str / Path: マルチ TIFF ファイルパス
    出力: (2k+1,H,W)  … 中央 z0±k をチャネルスタック
    """
    def __init__(self, keys, k: int = 1):
        super().__init__(keys)
        self.k = int(k)

    # ---- TIFF から必要面だけ読む ----------------------------------
    def _load_slice_stack(self, path: Union[str, Path], idxs: Sequence[int]):
        with tiff.TiffFile(path) as tf:
            z_max = len(tf.pages)
            idxs = [max(0, min(z_max - 1, i)) for i in idxs]
            # pages[i].asarray() は 1 ページずつメモリに展開
            return np.stack([tf.pages[i].asarray() for i in idxs], axis=0)

    # ---- メイン ---------------------------------------------------
    def __call__(self, data):
        d = dict(data)
        z0 = int(d["slice_idx"])

        for key in self.keys:
            src = d[key]

            # ---------- ケース A: パス文字列 --------------------------
            if isinstance(src, (str, Path)):
                idxs = [z0 + off for off in range(-self.k, self.k + 1)]
                stack = self._load_slice_stack(src, idxs)      # (2k+1,H,W)
                d[key] = torch.as_tensor(stack)                # Tensor に統一
                continue

            # ---------- ケース B: 既にメモリ上 ------------------------
            is_torch = torch.is_tensor(src)
            vol = src.cpu().numpy() if is_torch else src

            if vol.ndim == 4: vol = vol[0]         # (1,Z,H,W) → (Z,H,W)
            if vol.ndim != 3:
                raise ValueError(f"{key}: expect (Z,H,W), got {vol.shape}")

            z, _, _ = vol.shape
            idxs = [np.clip(z0 + off, 0, z - 1) for off in range(-self.k, self.k + 1)]
            stack = np.stack([vol[i] for i in idxs], axis=0)

            if is_torch:
                stack = torch.as_tensor(stack, dtype=src.dtype, device=src.device)
            d[key] = stack

        return d
