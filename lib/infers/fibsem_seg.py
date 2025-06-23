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
    Activations, AsDiscrete,
    KeepLargestConnectedComponent, RemoveSmallObjects,
)
import SimpleITK as sitk
from tqdm.auto import tqdm 

from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
import logging

import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


# ----------  InferTask 実装  ----------
class FibsemSegInfer(InferTask):
    """
    • TIFF は (Z, Y, X) 配列としてそのまま扱う  
    • 0 軸 (=Z) 毎に 2D-UNet へ入力  
    • 予測 volume を (Z, Y, X) で SimpleITK へ渡し，同じ向きで保存
    """

    def __init__(
        self,
        studies: str,
        model_dir: str,
        device: str | torch.device = "cuda",
        roi_size: tuple[int, int] = (256, 256),
        overlap: float = 0.25,
        ckpt_name: str = "model.pt",
    ):
        super().__init__(
            type=InferType.SEGMENTATION,
            dimension=2,
            description="FIB-SEM slice-wise 2D-UNet",
            labels={"background": 0, "cell_wall": 1, #"tannin_cell": 2
                    },
        )
        self.studies = studies
        self.id = "tc-fibsem-seg"   

        self.device   = torch.device(device)
        self.roi_size = roi_size
        self.overlap  = overlap

        self.in_channels = 3    # 2.5D入力 (3-slice stack)
        self.out_channels = len(self.labels) 

        # ---------- 重みファイル ----------
        ckpt_path = os.path.join(model_dir, ckpt_name)
        self.path = ckpt_path

        # ---------- ネットワーク ----------
        # self.network = UNet(
        #     spatial_dims=2,
        #     in_channels=3, # 2.5D
        #     out_channels=len(self.labels),
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        # )
        self.network = smp.Unet(
            encoder_name="resnet18",      # バックボーンを選択 (例: "efficientnet-b4", "resnext50_32x4d")
            encoder_weights="imagenet",   # 'imagenet' を指定して事前学習済みの重みをロード
            in_channels=self.in_channels,   
            classes=self.out_channels,  
        ).to(device)
        ckpt = os.path.join(model_dir, ckpt_name)
        logger.info(f"[FibsemSegInfer] try load checkpoint: {ckpt}")

        if os.path.exists(ckpt):
            self.network.load_state_dict(torch.load(ckpt, map_location="cpu"))
            logger.info(f"[FibsemSegInfer] checkpoint loaded ✓")
        else:
            logger.warning(f"[FibsemSegInfer] checkpoint NOT found – network is randomly initialised")
        self.network.to(self.device).eval()

        # ---------- 前処理 ----------
        self.pre_tf = Compose(
            [
                #EnsureChannelFirstd(keys="image"),   # → (1, H, W)
                ScaleIntensityd(keys="image"),
                EnsureTyped(keys="image", dtype=np.float32),
            ]
        )

        # --------- 3 クラス共通ポスト処理 ---------
        self.post_tf = Compose([
            Activations(softmax=True, dim=1),       # (B,C,H,W) → 確率
            AsDiscrete(argmax=True, dim=1, keepdim=False),         # (B,H,W)   → 整数ラベル 0/1/2
            KeepLargestConnectedComponent(   # クラス1・2の最大成分のみ残す
                applied_labels=[1,],# 2],
                independent=True,
                connectivity=2,
                num_components=1,
            ),
            RemoveSmallObjects(
                min_size=20,                 # 20pixel 未満を除去
                connectivity=2,
            ),
        ])
        

    # ------------------------------------------------------------------ #
    def __call__(self, data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        img_path = Path(data["image"]) 

        # ① TIFF 読み込み (Z,Y,X)
        vol = tiff.imread(img_path)
        if vol.ndim == 3 and vol.shape[0] == 1:      # 1-slice の例外
            vol = vol[0]

        logger.info(f"[DEBUG] input volume shape (Z,H,W): {vol.shape}")

        target_z = list(range(vol.shape[0]))
        Z, H, W = vol.shape
        samples = []
        for z in target_z:
            # 中央 z±1 を取る（端はクランプ）
            idxs = [max(0, min(Z - 1, z + off)) for off in (-1, 0, 1)]
            stack = np.stack([vol[i] for i in idxs], axis=0)   # (3, H, W)
            samples.append({"image": stack, "z": z})           # ← (C=3, H, W)

        loader  = DataLoader(Dataset(samples, transform=self.pre_tf),
                             batch_size=1, num_workers=0)

        # ④ 推論ループ
        pred_vol = np.zeros(vol.shape, dtype=np.uint8)      # (Z,Y,X)
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc="Infer (slices)"):
                z = int(batch["z"])
                img = batch["image"].to(self.device) 

                logits = sliding_window_inference(
                    img, roi_size=self.roi_size,
                    sw_batch_size=4, predictor=self.network, overlap=self.overlap
                )

                pred = self.post_tf(logits)
                pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

                # pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
                
                if z < 10:  # ← 詳細デバッグは z=0…9 のみ
                    logger.info(
                        f"[DEBUG] z={z:03d} | "
                        f"img:{tuple(img.shape)} "
                        f"logits:{tuple(logits.shape)} "
                        f"pred.shape:{pred.shape} "
                        f"unique:{np.unique(pred)}"
                    )
                
                pred_vol[z] = pred

        logger.info(f"Unique labels in pred_vol: {np.unique(pred_vol)}")    # Z へ書き戻し
        
        out_file = Path(self.studies) / f"{img_path.stem}_pred.nrrd"
        logger.info(f"[FibsemSegInfer] saving prediction to {out_file}")
        sitk_img = sitk.GetImageFromArray(pred_vol)         # axis0=Z
        sitk.WriteImage(sitk_img, out_file)

        return out_file, {                      # ① 出力ファイル
            "pred_slices": target_z,            # ② 追加メタ情報
            "total_slices": int(vol.shape[0]),
            "mime_type":     "nrrd"             # （任意）クライアントが自動判定しやすい
        }

    # MONAI Label から “常に利用可” と見せたい場合
    def is_valid(self) -> bool:
        return True
    
    def clear_cache(self):
        # 重みは保持しつつ、不要なテンソルだけ削除
        if hasattr(self, "_cache"):
            self._cache.clear()
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()
    
