#scripts/slice_dataset.py
import numpy as np
import monai
from monai.data import Dataset
from monai.transforms import LoadImage

class SliceDataset(Dataset):
    """
    1つの3D画像ファイルを受け取り、有効な2Dスライスを個別のデータサンプルとして扱うデータセット。

    Args:
        data (list): MONAILabelから渡されるデータリスト。[{'image': 'path', 'label': 'path'}] の形式。
        transform (callable, optional): 各2Dスライスに適用される変換。
    """
    def __init__(self, data, transform=None):
        if not data or len(data) != 1:
            # MONAILabelからは単一のファイルパスがリストで渡されることを想定
            print(f"警告: SliceDatasetは単一の画像/ラベルペアを期待しますが、{len(data)}個のデータを受け取りました。")
            self.items = []
            return

        self.file_paths = data[0]  # {'image': '...', 'label': '...'} を取得
        self.transform = transform

        # 画像読み込み用のローダーを準備
        # ここではメタデータを読まないシンプルなLoadImageを使用
        loader = LoadImage(image_only=True)
        
        # ラベルを一度だけ読み込み、有効なスライスをスキャンする
        label_3d = loader(self.file_paths['label'])
        
        # ラベル値が1または2であるスライスのインデックスをすべて見つける
        # 想定されるラベルの次元数に合わせて調整してください (例: label_3d.shape[2] or shape[0])
        # ここでは、Z軸が最後の次元にあると仮定します: (H, W, Z) or (C, H, W, Z)
        z_dim = -1
        if label_3d.ndim == 4 and label_3d.shape[0] == 1:
            label_3d = label_3d[0] # Channel次元を削除

        self.slice_indices = [
            i for i in range(label_3d.shape[z_dim]) 
            if np.any(np.isin(label_3d[..., i], [1, 2]))
        ]

        if not self.slice_indices:
            print(f"警告: ファイル '{self.file_paths['label']}' から有効なスライスが見つかりませんでした。")
        else:
            print(f"情報: {len(self.slice_indices)}個の有効なスライスが見つかりました。")

    def __len__(self):
        """データセットの全長（有効なスライスの総数）を返す"""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        指定されたインデックスに対応する単一の2Dスライスを読み込み、変換を適用して返す
        """
        # データセット内のインデックスを、実際のファイルのスライスインデックスに変換
        slice_idx = self.slice_indices[idx]

        # このスライス用のデータディクショナリを作成
        # 'slice_idx'を渡すことで、後続のトランスフォームが利用できる
        item_data = {
            "image": self.file_paths['image'],
            "label": self.file_paths['label'],
            "slice_idx": slice_idx
        }

        # 後続のトランスフォーム（LoadImagedなど）を適用
        if self.transform:
            item_data = self.transform(item_data)
            
        return item_data