"""
FibsemSegInfer
==============

 * 1 つの Multi-TIFF を入力
 * ラベル値 {0,1,2} が 1 枚でもあるスライスだけ推論
 * 出力は 2-D mask (same H×W) を PNG/NPY/TIFF で保存し，パスを返却
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import tifffile as tiff
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import SimpleInferer, sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped,
)

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType

# ----------  ユーティリティ (元の ExtractValidSlicesd と等価なフィルタ) ----------
def find_labeled_slices(img: np.ndarray) -> Sequence[int]:
    """img shape: [H,W,S]。値 1/2 を含むスライス index を返す"""
    labeled = []
    for z in range(img.shape[-1]):
        if np.logical_or(img[..., z] == 1, img[..., z] == 2).any():
            labeled.append(z)
    return labeled


# ----------  InferTask 実装  ----------
class FibsemSegInfer(InferTask):
    """
    MONAI Label が呼び出す推論タスク。
    """

    def __init__(
        self,
        model_dir: str,
        network: torch.nn.Module | None = None,
        device: torch.device | str = "cuda",
        roi_size: tuple[int, int] = (256, 256),
        overlap: float = 0.25,
    ):
        super().__init__(
            type=InferType.SEGMENTATION,
            dimension=2,
            description="FIB-SEM 2-D UNet segmentation (background/cell_wall/tannin_cell)",
            labels={"background": 0, "cell_wall": 1, "tannin_cell": 2},
        )

        self.device = torch.device(device)
        self.model_dir = model_dir
        self.roi_size = roi_size
        self.overlap = overlap

        # ネットワークを自前でロード
        self.network = (
            network
            if network
            else UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=3,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        )
        ckpt = os.path.join(model_dir, "model.pt")
        if os.path.exists(ckpt):
            self.network.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.network.to(self.device).eval()

    # --------------------------------------------------------------------- #
    #  1) 前処理・後処理 (MONAI Transform)                                   #
    # --------------------------------------------------------------------- #
    def pre_transforms(self) -> Compose:
        return Compose(
            [
                LoadImaged(keys=["image"], image_only=True),  # numpy [H,W,S]
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=np.float32),
            ]
        )

    # この例では後処理は不要（logits→argmax は run_infer 内で）
    def post_transforms(self):
        return None

    # --------------------------------------------------------------------- #
    #  2) 推論本体                                                           #
    # --------------------------------------------------------------------- #
    def __call__(
        self,
        data: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            data["image"] : path to multi-tiff
        Returns:
            {"label": <output file>, "params": {...}}
        """
        img_path = data["image"]
        vol = tiff.imread(str(img_path))  # [S,H,W]  or  [H,W,S]
        if vol.ndim == 3 and vol.shape[0] == 1:  # まれに [1,H,W]
            vol = vol[0]

        # Z 軸を最後にして [H,W,S]
        if vol.shape[0] == vol.shape[1]:  # assume square slice ⇒ shape is [S,H,W]
            vol = np.moveaxis(vol, 0, -1)

        target_slices = find_labeled_slices(vol)
        if not target_slices:
            raise RuntimeError("No labeled slices (value 1/2) found in image")

        # DataLoader に渡す dict list
        samples = []
        for z in target_slices:
            samples.append({"image": vol[..., z][np.newaxis, ...], "slice_idx": z})  # add channel dim

        ds = Dataset(samples, transform=self.pre_transforms())
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        outputs = {}
        for batch in loader:
            sl_idx = int(batch["slice_idx"])
            img = batch["image"].to(self.device)  # [B=1,1,H,W]

            # Sliding-window (単一スライスでも幅が足りないとき用)
            with torch.no_grad():
                logits = sliding_window_inference(
                    img, roi_size=self.roi_size, sw_batch_size=4, predictor=self.network, overlap=self.overlap
                )
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)[0]  # [H,W]

            outputs[sl_idx] = pred

        # --------------- 保存 ---------------
        # stack back to [H,W,S] full volume (background 0 for skipped slices)
        full_pred = np.zeros_like(vol, dtype=np.uint8)
        for z, mask in outputs.items():
            full_pred[..., z] = mask

        tmpdir = tempfile.mkdtemp(prefix="monailabel_")   # /tmp/monailabel_xxxxx
        out_file = os.path.join(tmpdir, Path(img_path).stem + "_pred.tiff")
        tiff.imwrite(out_file, np.moveaxis(full_pred, -1, 0).astype(np.uint8))

        return {
            "label": out_file,
            "params": {
                "pred_slices": list(outputs.keys()),
                "total_slices": int(vol.shape[-1]),
            },
        }
    
    def is_valid(self) -> bool:
        return True 
