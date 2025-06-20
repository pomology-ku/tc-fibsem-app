# SPDX-License-Identifier: Apache-2.0
"""
Active-learning strategy:
  各未ラベル画像に対してモデルの **平均エントロピー** を計算し，
  もっとも不確実性（= 情報利得）が高い 1 件を返す
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import tifffile as tiff
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.utils.others.generic import device_list  # モデルなし時のフォールバック


class HighestUncertainty(Strategy):
    """
    Predictive-entropy サンプリング
    """

    def __init__(
        self,
        # 推論タスクを渡しておくと，そのネットワークをそのまま利用
        infer_task,
        n_slices: int = 10,       # 画像 1 件あたり評価する slice 数
        device: str | torch.device | None = None,
    ):
        super().__init__("Highest predictive entropy")
        self.infer_task = infer_task
        self.n_slices = n_slices
        self.device = (
            torch.device(device) if device else infer_task.device if hasattr(infer_task, "device") else "cpu"
        )
        self.network = infer_task.network.to(self.device).eval()

    # ------------------------------------------------------------------ #
    #  画像 1 件に対する “平均エントロピー” を求めるユーティリティ
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _score(self, img_path: str | Path) -> float:
        """
        Args:
            img_path : マルチ TIFF 1 ファイル
        Returns:
            平均エントロピー (高いほど不確実)
        """
        vol = tiff.imread(str(img_path))  # [S,H,W] or [H,W,S]
        if vol.ndim == 3 and vol.shape[0] == vol.shape[1]:  # [S,H,W] なら Z 軸を最後へ
            vol = np.moveaxis(vol, 0, -1)

        # ── slice を等間隔で subsample ───────────────────────────────
        z_indices = np.linspace(0, vol.shape[-1] - 1, self.n_slices, dtype=int)
        imgs = vol[..., z_indices].astype(np.float32) / 255.0  # (H,W,n)
        imgs = torch.from_numpy(imgs).unsqueeze(1).to(self.device)  # (n,1,H,W)

        # ── ネットワーク推論 → softmax → エントロピー ───────────────
        logits: Sequence[torch.Tensor] = []
        for sl in imgs:
            logits.append(self.network(sl[None, ...]))         # (1,C,H,W)
        logits = torch.cat(logits, dim=0)                      # (n,C,H,W)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(torch.clamp(probs, 1e-6))).sum(dim=1)  # (n,H,W)
        return entropy.mean().item()

    # ------------------------------------------------------------------ #
    #  Strategy エントリポイント
    # ------------------------------------------------------------------ #
    def __call__(self, request: Dict, datastore: Datastore):
        candidates = datastore.get_unlabeled_images()
        if not candidates:
            return None

        best_img, best_score = None, -1.0
        for img_id in candidates:
            score = self._score(datastore.get_image_uri(img_id))
            if score > best_score:
                best_img, best_score = img_id, score

        return {"id": best_img, "uncertainty": best_score}
