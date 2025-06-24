import logging
import os
from typing import Dict
import torch

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.first import First
import segmentation_models_pytorch as smp

# ---- custom tasks -----------------------------------------------------------
from lib.infers.fibsem_seg import FibsemSegInfer
from lib.trainers.fibsem_seg import FibsemSegTrain

LOG = logging.getLogger(__name__)


class MyApp(MONAILabelApp):
    """Minimal MONAI Label App for FIB‑SEM segmentation.

    * 1 × multi‑tiff  ➜ 2‑D slices
    * 3‑class UNet (background / cell_wall / tannin_cell)
    * Training & inference defined in ``lib/``
    """

    def __init__(self, app_dir: str, studies: str, conf: Dict[str, str]):
        self.name = "tc-fibsem-seg"
        self.conf = conf

        backbone = conf.get("encoder", "resnet18")
        self.name = f"tc-fibsem-seg-{backbone}"
        self.model_dir = os.path.join(app_dir, "model", self.name)

        # モデルがなければこのタイミングで用意しておく
        model_path = os.path.join(self.model_dir, "model.pt")
        if not os.path.exists(model_path):
            LOG.info(f"Model not found at '{model_path}', creating a new one.")
            os.makedirs(self.model_dir, exist_ok=True)
            backbone = conf.get("encoder", "resnet18")
            pretrained_model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3,
                classes=2,
            )
            torch.save(pretrained_model.state_dict(), model_path)
            LOG.info(f"Saved initial pretrained model to '{model_path}'")

        # ———— その後で親クラス初期化 ————
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=self.name,
            description="2-D UNet for FIB-SEM multi-tiff",
            version="0.1.0",
            labels=["background", "cell_wall"],
        )

    def init_infers(self) -> Dict[str, InferTask]:
        encoder = self.conf.get("encoder", "resnet18")
        return {
            "tc-fibsem-seg": FibsemSegInfer(
                studies=self.studies,
                model_dir=self.model_dir,
                encoder=encoder,
                roi_size=(256, 256),
            )
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        if self.conf.get("skip_trainers", "false").lower() in ("true", "1"):  # server flag
            return {}
        encoder = self.conf.get("encoder", "resnet18")
        trainer = FibsemSegTrain(app_dir=self.app_dir, encoder=encoder, model_dir=self.model_dir)
        LOG.info("+++ Adding Trainer  tc-fibsem-seg  =>  %s", trainer)
        return {"tc-fibsem-seg": trainer}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": First(),
        }

