# scripts/fibsem_transforms.py
from monai.transforms import MapTransform
import numpy as np
import tifffile as tiff
import torch

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
    keys=("label",) など必要なキーだけに適用してください。
    """
    def __call__(self, data):
        d = dict(data)
        z = int(d["slice_idx"])
        for k in self.keys:
            arr = d[k]
            if arr.ndim == 3:          # (Z,H,W)
                d[k] = arr[z]          # ← 軸0で切る
            elif arr.ndim == 4:        # (C,Z,H,W)
                d[k] = arr[:, z]       # Cは残す
            else:
                raise RuntimeError(f"Unexpected shape {arr.shape}")
        return d

class StackNeighborSlicesd(MapTransform):
    """
    入力 : (Z,H,W) もしくは (1,Z,H,W)
    出力 : (2k+1,H,W)  ― Z±k をチャネルスタック
    """
    def __init__(self, keys, k: int = 1):
        super().__init__(keys)
        self.k = int(k)

    def __call__(self, data):
        d = dict(data)
        z0 = int(d["slice_idx"])

        for key in self.keys:            # keys="image"
            vol = d[key]
            is_torch = torch.is_tensor(vol)
            vol_np = vol.cpu().numpy() if is_torch else vol

            # ---- shape 整形 ----
            if vol_np.ndim == 4:         # (1,Z,H,W) → squeeze
                vol_np = vol_np[0]
            if vol_np.ndim != 3:
                raise ValueError(f"Need (Z,H,W), got {vol_np.shape}")

            # ここで **axis 0 が Z** と確定
            z, h, w = vol_np.shape
            idxs = [np.clip(z0 + off, 0, z - 1) for off in range(-self.k, self.k + 1)]
            stack = np.stack([vol_np[i] for i in idxs], axis=0)   # (2k+1,H,W)

            if is_torch:
                stack = torch.as_tensor(stack, dtype=vol.dtype, device=vol.device)
            d[key] = stack
        return d
