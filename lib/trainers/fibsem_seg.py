# lib/trainers/fibsem_seg.py
import logging, os, random, json
from typing import List, Dict, Sequence, Any, Optional

import numpy as np
import torch
import tifffile as tiff
from sklearn.model_selection import train_test_split
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast             # ↳ AMP optional
from ignite.engine import Events
from monai.data import DataLoader, CacheDataset, partition_dataset, pad_list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandAxisFlipd, RandRotate90d,
    RandGaussianNoised, ScaleIntensityd, Activationsd, AsDiscreted,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler, CheckpointSaver, ValidationHandler, LrScheduleHandler, MeanDice, from_engine
)

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.train import TrainTask         # ← MONAI-Label 抽象基底
from scripts.fibsem_transforms import ExtractValidSlicesd, SelectSliceByKeyd

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
        in_channels: int = 1,
        out_channels: int = 3,
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
        max_epochs   = int(request.get("max_epochs", 20))
        batch_size   = int(request.get("train_batch_size", 4))
        num_workers  = int(request.get("num_workers", 4))
        lr           = float(request.get("learning_rate", 1e-3))
        val_split    = float(request.get("val_split", 0.2))
        device       = torch.device(request.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

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
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            SelectSliceByKeyd(keys=("label", "image")),
            #ExtractValidSlicesd(keys=("label", "image")),
        ]
        train_tf = base + [
            RandAxisFlipd(keys=("image", "label"), prob=0.5),
            RandRotate90d(keys=("image", "label"), prob=0.5),
            RandGaussianNoised(keys="image", prob=0.5, std=0.05),
            ScaleIntensityd(keys="image"),
        ]
        val_tf = base + [
            ScaleIntensityd(keys="image")
        ]

        # ---------------- Dataset / Loader ---------------- #
        train_ds = CacheDataset(train_list, Compose(train_tf), num_workers=num_workers, cache_rate=1.0)
        val_ds   = CacheDataset(val_list,   Compose(val_tf),   num_workers=num_workers, cache_rate=1.0)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)

        # ---------------- Network / Loss / Optimizer ---------------- #
        net = UNet(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # ---------------- Inferer / Post ---------------- #
        post_trans = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=("pred", "label"), argmax=(True, False),
                        to_onehot=net.out_channels),
        ])

        # ---------------- Engines ---------------- #

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=val_loader,
            network=net,
            postprocessing=post_trans,
            key_val_metric={
                "val_mean_dice": MeanDice(
                    include_background=False,
                    output_transform=_pred_label
                )
            },
            val_handlers=[
                StatsHandler(
                    tag_name="val_mean_dice",
                    output_transform=lambda *_: None,                   # ここはログだけなので None 返しで OK
                    global_epoch_transform=lambda epoch: epoch
                )
            ],
        )

        trainer = SupervisedTrainer(
            max_epochs=max_epochs,
            device=device,
            train_data_loader=train_loader,
            network=net,
            loss_function=loss_fn,
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
        return {"status": "finished", "epochs": max_epochs}


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