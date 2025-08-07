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
    """MONAI Label App for FIB‑SEM segmentation.

    * 1 × multi‑tiff  ➜ 2‑D slices
    * Training & inference are defined in ``lib/``.
    """

    DEFAULT_LABELS = ["cell_wall"]

    # -------------------------------------------------
    #  Constructor
    # -------------------------------------------------
    def __init__(self, app_dir: str, studies: str, conf: Dict[str, str]):
        self.conf = conf
        encoder = conf.get("encoder", "resnet18")
        self.name = f"tc-fibsem-seg-{encoder}"

        self.app_dir = app_dir
        self.studies = studies
        self.model_dir = os.path.join(app_dir, "model", self.name)
        os.makedirs(self.model_dir, exist_ok=True)

        # Ensure pre‑trained weights exist ----------------------------------
        model_path = os.path.join(self.model_dir, "model.pt")
        if not os.path.exists(model_path):
            LOG.info("Model not found – exporting ImageNet‑pretrained UNet backbone to %s", model_path)
            net = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=len(self.DEFAULT_LABELS),
            )
            torch.save(net.state_dict(), model_path)

        # Parent class init -------------------------------------------------
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=self.name,
            description="2.5‑D UNet for FIB‑SEM multi‑tiff segmentation",
            version="0.2.0",
            labels=self.DEFAULT_LABELS,
        )

    # -------------------------------------------------
    #  Tasks
    # -------------------------------------------------
    def init_infers(self) -> Dict[str, InferTask]:
        roi = tuple(map(int, self.conf.get("roi_size", "256,256").split(",")))
        return {
            self.name: FibsemSegInfer(
                studies=self.studies,
                model_dir=self.model_dir,
                encoder=self.conf.get("encoder", "resnet18"),
                roi_size=roi,
                device=self.conf.get("device", "cuda"),
            )
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        if self.conf.get("skip_trainers", "false").lower() in ("true", "1"):
            return {}
        trainer = FibsemSegTrain(
            app_dir=self.app_dir,
            model_dir=self.model_dir,
            encoder=self.conf.get("encoder", "resnet18"),
        )
        LOG.info("+++ Adding Trainer  %s", trainer)
        return {self.name: trainer}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": First(),
        }

