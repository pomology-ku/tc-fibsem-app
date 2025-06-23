#!/usr/bin/env python3
import os, sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from monailabel.datastore.local import LocalDatastore
from main import MyApp

def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for tc-fibsem-seg MONAILabel App"
    )
    parser.add_argument(
        "--app-dir", "-a", required=True,
        help="Path to the app directory (where MyApp lives)"
    )
    parser.add_argument(
        "--studies", "-s", required=True,
        help="Path to your studies folder (multi-tiff data root)"
    )
    parser.add_argument(
        "--conf", "-c", nargs="*", default=[],
        help="Additional key=value pairs for app.conf (e.g. skip_trainers=true)"
    )

    # サブコマンド機能
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── train サブコマンド ─────────────────────────────
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("--max-epochs",   type=int,   default=100)
    p_train.add_argument("--batch-size",   type=int,   default=8)
    p_train.add_argument("--num-workers",  type=int,   default=4)
    p_train.add_argument("--learning-rate",type=float, default=1e-4)
    p_train.add_argument("--val-split",    type=float, default=0.2)
    p_train.add_argument("--device",       type=str,   default="cuda:0")

    # ─── infer サブコマンド ─────────────────────────────
    p_inf = subparsers.add_parser("infer", help="Run inference")
    p_inf.add_argument("--image",  required=True,
                       help="Image name (as in your studies folder)")
    p_inf.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # conf を dict に変換
    conf = {}
    for kv in args.conf:
        if "=" in kv:
            k, v = kv.split("=", 1)
            conf[k] = v
    # アプリ／Datastore 初期化
    app = MyApp(args.app_dir, args.studies, conf)
    datastore = LocalDatastore(args.studies, 
                               extensions=("*.TIFF","*.tif"),
                               images_dir=args.studies,
                               labels_dir=args.studies+ "/labels",
                               )
    

    if args.command == "train":
        trainer = app.init_trainers().get("tc-fibsem-seg")
        if trainer is None:
            raise RuntimeError("Trainer 'tc-fibsem-seg' not available")
        req = {
            "max_epochs": args.max_epochs,
            "train_batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "val_split": args.val_split,
            "device": args.device,
        }
        result = trainer(req, datastore)
        print("Training result:", result)

    elif args.command == "infer":
        inferer = app.init_infers().get("tc-fibsem-seg")
        if inferer is None:
            raise RuntimeError("InferTask 'tc-fibsem-seg' not available")
        req = {
            "image": args.image,
            "device": args.device,
        }
        result = inferer(req, datastore)
        print("Inference result:", result)

if __name__ == "__main__":
    main()
