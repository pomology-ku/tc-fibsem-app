# lib/trainers/fibsem_seg.py
import logging, os, random, json
from typing import List, Dict, Sequence, Any, Optional
import shutil, glob

import numpy as np
import torch
import tifffile as tiff
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast             # ↳ AMP optional
from ignite.engine import Events
from monai.data import DataLoader, CacheDataset, partition_dataset, pad_list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    Rand2DElasticd, RandAffined, RandZoomd,
    RandGaussianNoised, RandShiftIntensityd,
    RandAxisFlipd, RandRotate90d,
    ScaleIntensityd, EnsureTyped,
    Activationsd, AsDiscreted,
    SaveImaged, SpatialPadd,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss, HausdorffDTLoss
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler, CheckpointSaver, ValidationHandler, LrScheduleHandler, MeanDice, from_engine
)

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask         # ← MONAI-Label 抽象基底
from monai.inferers import sliding_window_inference
from scripts.fibsem_transforms import ExtractValidSlicesd, SelectSliceByKeyd, StackNeighborSlicesd
from monai.utils import set_determinism

import segmentation_models_pytorch as smp 

set_determinism(seed=42)

logger = logging.getLogger(__name__)


class FibsemSegTrain(TrainTask):
    """
    1 つの multi-TIFF から有効スライスを抜き出し、
    2D-UNet を end-to-end で訓練する TrainTask。
    """

    def __init__(
        self,
        app_dir: str,
        description: str = "2D-UNet trainer for tc-fibsem-seg",
        in_channels: int = 3, # 2.5D
        out_channels: int = 2,#3,
        spatial_size: Optional[Sequence[int]] = None,
    ):
        super().__init__(description)
        self.app_dir = app_dir
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size or [512, 512]
        self.version = "0.1"

    # --------------- 主要 API --------------- #
    def __call__(self, request: Dict[str, Any], datastore: Datastore):
        """
        MONAI-Label ランタイムが呼び出すエントリポイント。
        `request` には Slicer UI で指定した max_epochs, val_split などが入ってくる。
        """
        # ---------------- ハイパーパラメータ ---------------- #
        max_epochs   = int(request.get("max_epochs", 100))
        batch_size   = int(request.get("train_batch_size", 8))
        num_workers  = int(request.get("num_workers", 4))
        lr           = float(request.get("learning_rate", 1e-4))
        val_split    = float(request.get("val_split", 0.2))
        device       = device = torch.device("cuda:0") #torch.device(request.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        logger.info(f">>> TRAIN   on {device};  epochs={max_epochs};  batch={batch_size};  val_split={val_split}")

        # ---------------- DataList ---------------- #
        datalist = _make_datalist(datastore)     # 行数 = 有効スライス
        train_list, val_list = partition_dataset(
            datalist, ratios=[1 - val_split, val_split], shuffle=True, seed=0
        )

        if not val_list:          # val_split <= 0 なら空 list を返す
            val_list = []

        # ---------------- Transforms ---------------- #
        base = [
            # LoadImaged(keys=("image", "label")),
            # EnsureTyped(keys=("image", "label")),

            StackNeighborSlicesd(keys="image", k=1),   # image: (3,H,W)
            SelectSliceByKeyd(keys="label"),           # ← label だけ 2D へ

            EnsureTyped(keys=("image", "label")),

            EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
            EnsureChannelFirstd(keys="image", channel_dim=0, strict_check=False),
        ]

        PATCH_DEBUG_DIR = os.path.join(self.app_dir, "debug")
        
        train_tf = base + [

            #    ここで size を UNet の ROI サイズ (例: 256×256) に合わせる
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=(256, 256),
                pos=1, neg=1,           
                num_samples=32,
            ),

            # SaveImaged(
            #     keys="label",                        # 画像だけでOK。ラベルも見たいなら ("image","label")
            #     output_dir=PATCH_DEBUG_DIR,
            #     output_postfix="patch_label",              # ファイル名: <原画像名>_patch.nii.gz など
            #     output_ext=".png",                  # ".png", ".nii.gz" なども可
            #     resample=False,                      # クロップなのでリサンプル不要
            #     separate_folder=False,               # 1 ディレクトリにまとめる
            # ),
            # SaveImaged(
            #     keys="image",                        # 画像だけでOK。ラベルも見たいなら ("image","label")
            #     output_dir=PATCH_DEBUG_DIR,
            #     output_postfix="patch_img",              # ファイル名: <原画像名>_patch.nii.gz など
            #     output_ext=".png",                  # ".png", ".nii.gz" なども可
            #     resample=False,                      # クロップなのでリサンプル不要
            #     separate_folder=False,               # 1 ディレクトリにまとめる
            # ),

            # ② 幾何学変換（順序：Crop → 変換 が定石）
            RandAffined(
                keys=("image", "label"),
                rotate_range=None,
                scale_range=(0.1, 0.1),         # ±10% 拡大縮小
                shear_range=(0.1, 0.1),
                prob=0.3,
            ),
            Rand2DElasticd(                      # 弾性変形
                keys=("image", "label"),
                spacing=(64, 64),
                magnitude_range=(0.5, 1.5),
                prob=0.2,
            ),
            RandZoomd(
                keys=("image", "label"),
                min_zoom=0.9, max_zoom=1.1,
                prob=0.2,
            ),
            RandAxisFlipd(keys=("image", "label"), prob=0.5),
            RandRotate90d(keys=("image", "label"), prob=0.5),

            # ③ intensity 系
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.5, std=0.05),

            # ④ 正規化
            ScaleIntensityd(keys="image"),
        ]
        val_tf = base + [
            EnsureTyped(keys=("image", "label")),
            ScaleIntensityd(keys="image"),

            # 入力が 512×512 未満なら pad、超えていればそのまま
            SpatialPadd(
                keys=("image", "label"),       # dict 版なので keys が必須
                spatial_size=(512, 512),
                method="symmetric",
            ),
        ]

        # dbg_ds = CacheDataset(train_list, Compose(train_tf),
        #               cache_rate=0.0, num_workers=0)  # ★ workers=0
        # for i, patches in enumerate(dbg_ds[:3]):          # 3スライス分
        #     for j, p in enumerate(patches):
        #         print(f"[DBG] {i}.{j}  image {p['image'].shape}  label {p['label'].shape}")

        # ---------------- Dataset / Loader ---------------- #
        train_ds = CacheDataset(train_list, Compose(train_tf), num_workers=num_workers, cache_rate=1.0)
        val_ds   = CacheDataset(val_list,   Compose(val_tf),   num_workers=num_workers, cache_rate=1.0)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)

        # ---------------- Network / Loss / Optimizer ---------------- #
        # net = UNet(
        #     spatial_dims=2,
        #     in_channels=self.in_channels,
        #     out_channels=self.out_channels,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        # ).to(device)

        net = smp.Unet(
            encoder_name="resnet18",      # バックボーンを選択 (例: "efficientnet-b4", "resnext50_32x4d")
            encoder_weights="imagenet",   # 'imagenet' を指定して事前学習済みの重みをロード
            in_channels=self.in_channels,   # 現在のコード通り 3 (2.5D入力)
            classes=self.out_channels,      # 現在のコード通り 2
        )
        net.out_channels = self.out_channels  # ← UNet の出力チャンネル数を設定
        for p in net.encoder.parameters():
            p.requires_grad = False        # エンコーダーは固定（転移学習）
        net.to(device)

        dice = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # ---------------- Inferer / Post ---------------- #
        post_trans = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False),
                        to_onehot=net.out_channels),
        ])

        # ---------------- Engines ---------------- #
        def val_inferer(inputs, network):          # ★ network を受け取る
            return sliding_window_inference(
                inputs,
                roi_size=(256, 256),
                sw_batch_size=4,
                predictor=network,                 # ←★ ここに渡す
                overlap=0.5,
            )

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_loader,
            network=net,
            inferer=val_inferer, 
            postprocessing=post_trans,
            key_val_metric={
                "val_mean_dice": MeanDice(include_background=False, output_transform=_pred_label),
            },
            val_handlers=[
                StatsHandler(tag_name="val_mean_dice"),
            ],
        )

        trainer = SupervisedTrainer(
            max_epochs=max_epochs,
            device=device,
            train_data_loader=train_loader,
            network=net,
            loss_function=dice,
            optimizer=optimizer,
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            ValidationHandler(validator=evaluator, epoch_level=True, interval=1),
        )
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda out: (
                out if isinstance(out, (float, int))              # スカラー
                else out["loss"] if isinstance(out, dict) and "loss" in out
                else out[0] if isinstance(out, (list, tuple)) and out  # list/tuple[0]
                else None
            ),
        ).attach(trainer)
        LrScheduleHandler(torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)).attach(trainer)
        CheckpointSaver(
            save_dir=os.path.join(self.app_dir, "model_checkpoints"),
            save_dict={"model": net},
            save_final=True,
            key_metric_name="val_mean_dice",
            key_metric_n_saved=1,
        ).attach(trainer)


        # --------------------- RUN --------------------- #
        trainer.run()

        # ① CheckpointSaver が書いた最新モデル (*.pt) を取得
        ckpt_dir = os.path.join(self.app_dir, "model_checkpoints")
        best_ckpt = max(
            glob.glob(os.path.join(ckpt_dir, "*.pt")),
            key=os.path.getmtime,            # 最終更新が新しいもの＝best
        )

        # ② infer 用ディレクトリへコピー
        publish_dir = os.path.join(self.app_dir, "model", "tc-fibsem-seg")
        os.makedirs(publish_dir, exist_ok=True)
        dst = os.path.join(publish_dir, "model.pt")          # ← infer 側と同じファイル名に
        shutil.copy2(best_ckpt, dst)
        logger.info(f"★ Published {best_ckpt}  →  {dst}")

        # ③ MONAI-Label へ戻り値で知らせると UI にも反映される
        return {
            "status": "finished",
            "epochs": max_epochs,
            "model_path": dst,                # <- key 名は任意だが model_path が慣例
        }
    


# -------------------------------------------------------------------------
# util
# -------------------------------------------------------------------------

def _vol_valid_z(label_vol: np.ndarray) -> List[int]:
    """
    (Z,Y,X) ラベル volume から，値 1 または 2 が存在する Z index を返す
    """
    z_idx = []
    for z in range(label_vol.shape[0]):              # ★ axis=0 を Z とする
        if np.any(np.isin(label_vol[z], (1, 2))):
            z_idx.append(z)
    return z_idx

def _make_datalist(store: Datastore,
                         max_slices_per_vol: Optional[int] = None, 
                         ) -> List[Dict]:
    """
    Datastore → slice-level datalist
       {"image": <tiff>, "label": <tiff>, "slice_idx": z}
    """
    samples: list[dict] = []

    for it in store.datalist():
        if not it or not os.path.exists(it["label"]):
            continue

        img_tiff, lbl_tiff = it["image"], it["label"]
        lbl_vol = tiff.imread(lbl_tiff)              # (Z,Y,X)
        # print(f"shape: {lbl_vol.shape}")
        

        valid_z = _vol_valid_z(lbl_vol)
        if not valid_z:
            continue

        if max_slices_per_vol:                      # スライス数上限
            random.shuffle(valid_z)
            valid_z = valid_z[:max_slices_per_vol]

        for z in valid_z:
            samples.append({"image": img_tiff,
                            "label": lbl_tiff,
                            "slice_idx": z})

        logger.info(f"{Path(img_tiff).name}: keep {len(valid_z)} / {lbl_vol.shape[0]} slices")

    if not samples:
        raise RuntimeError("ラベル付き slice が 1 枚も見つかりません。")

    random.shuffle(samples)
    logger.info(f"Datalist: {len(samples)} slice-samples ／ {len({s['image'] for s in samples})} volumes")
    return samples

def make_slice_datalist(image_path: str, label_path: str):
    lbl_vol = tiff.imread(label_path)        # shape = (S,H,W) or (H,W,S)
    if lbl_vol.shape[0] == lbl_vol.shape[1]: # S,H,W のとき
        lbl_vol = np.moveaxis(lbl_vol, 0, -1)

    valid = [z for z in range(lbl_vol.shape[-1])
             if np.any(np.isin(lbl_vol[..., z], (1, 2)))]  # 1 or 2 がある面

    dl = [{"image": image_path,
           "label": label_path,
           "slice_idx": int(z)} for z in valid]
    return dl

def _pred_label(out):
    """
    state.output から (pred, label) タプルを返す。
      out : dict, tuple, list[dict] いずれにも対応
    """
    # list[dict] → dict
    if isinstance(out, list):
        out = out[0] if out else {}

    # dict → pred/label 抽出
    if isinstance(out, dict):
        return out["pred"], out["label"]

    # 既に tuple の場合
    return out  # (pred, label)