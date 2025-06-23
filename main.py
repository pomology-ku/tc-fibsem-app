import logging
import os
from typing import Dict

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.random import Random
from monailabel.tasks.activelearning.first import First

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
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="tc-fibsem-seg",
            description="2‑D UNet for FIB‑SEM multi‑tiff",
            version="0.1.0",
            labels=["background", "cell_wall", #"tannin_cell"
                    ],
        )

    # ---------------------------------------------------------------------
    # tasks
    # ---------------------------------------------------------------------
    def init_infers(self) -> dict[str, InferTask]:
        return {
            "tc-fibsem-seg": FibsemSegInfer(
                studies=self.studies,
                model_dir=os.path.join(self.app_dir, "model", "tc-fibsem-seg"),
                roi_size=(256, 256),  # 必要なら変更
            )
        }

    def init_trainers(self) -> Dict[str, TrainTask]:
        if self.conf.get("skip_trainers", "false").lower() in ("true", "1"):  # server flag
            return {}
        trainer = FibsemSegTrain(app_dir=self.app_dir)
        LOG.info("+++ Adding Trainer  tc-fibsem-seg  =>  %s", trainer)
        return {"tc-fibsem-seg": trainer}

    def init_strategies(self):
        return {
            "random": Random(),
            "first": First(),
        }


# -------------------------------------------------------------------------
# optional CLI helper (head‑less debug)
# -------------------------------------------------------------------------

def main():
    import argparse
    import shutil
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    home = str(Path.home())
    default_studies = f"{home}/tc-fibsem"

    parser = argparse.ArgumentParser(description="Run training or inference without starting the server")
    parser.add_argument("--studies", "-s", default=default_studies)
    parser.add_argument("--mode", "-m", choices=["infer", "train", "batch"], default="infer")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--epochs", "-e", type=int, default=20)
    args = parser.parse_args()

    app_dir = os.path.dirname(__file__)
    conf = {"models": "tc-fibsem-seg"}
    app = MyApp(app_dir, args.studies, conf)

    if args.mode == "infer":
        sample = app.next_sample({"strategy": "first"}) or app.datastore().datalist()[0]
        image_id = sample["id"] if isinstance(sample, dict) else None
        device = args.device
        res = app.infer({"model": "tc-fibsem-seg", "image": image_id, "device": device})
        print("Label saved at:", res["file"])

    elif args.mode == "train":
        app.train({
            "model": "tc-fibsem-seg",
            "max_epochs": args.epochs,
            "val_split": 0.1,
            "device": args.device,
            "multi_gpu": False,
        })

    elif args.mode == "batch":
        app.batch_infer({"model": "tc-fibsem-seg", "images": "unlabeled"})


if __name__ == "__main__":
    main()
