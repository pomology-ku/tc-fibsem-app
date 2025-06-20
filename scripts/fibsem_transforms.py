# scripts/fibsem_transforms.py
from monai.transforms import MapTransform
import numpy as np
import tifffile as tiff

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
    dict["slice_idx"] で指定された z 面を切り出す。
    keys=("label", "image")
    """
    def __call__(self, data):
        d = dict(data)
        z = int(d["slice_idx"])
        for k in self.keys:
            arr = d[k]
            if arr.ndim == 3:           # (H,W,S)
                d[k] = arr[..., z]
            elif arr.ndim == 4:         # (C,H,W,S)
                d[k] = arr[..., z]
            else:
                raise RuntimeError(f"Unexpected shape {arr.shape}")
        return d