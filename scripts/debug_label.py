import numpy as np
from monai.transforms import LoadImage
import sys

def debug_label_file(file_path):
    """
    指定されたラベルファイルを読み込み、その内容をデバッグします。
    """
    if not file_path:
        print("エラー: ラベルファイルのパスを指定してください。")
        return

    print(f"--- ラベルファイル '{file_path}' のデバッグを開始します ---")

    try:
        # データセットと同じ方法で画像を読み込む
        loader = LoadImage(image_only=True)
        label = loader(file_path)
        
        print(f"\n[1] 読み込んだデータの情報:")
        print(f" - 型 (dtype): {label.dtype}")
        print(f" - 次元 (shape): {label.shape}")

        # 画像内に存在する全てのユニークなピクセル値を出力
        unique_values = np.unique(label)
        print(f"\n[2] ファイル内に存在するユニークなピクセル値:")
        print(f" - {unique_values}")
        
        # --- SliceDatasetと同じロジックで有効なスライスを探す ---
        
        # チャンネル次元があれば削除
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]
            print("\n[INFO] チャンネル次元を削除しました。新しい次元:", label.shape)

        # チェックするラベル値を設定（★★ここを自分のラベル値に合わせて変更可能★★）
        target_label_values = [1, 2] 
        
        # Z軸（スライス）の次元を決定（通常は -1 or 2）
        z_dim_index = -1 
        num_slices = label.shape[z_dim_index]
        
        print(f"\n[3] スライス検索の実行:")
        print(f" - スライス数: {num_slices}")
        print(f" - 検索対象のラベル値: {target_label_values}")

        valid_slices = []
        for i in range(num_slices):
            # スライスを抽出
            current_slice = label[..., i]
            # スライス内にターゲットのラベル値が存在するかチェック
            if np.any(np.isin(current_slice, target_label_values)):
                valid_slices.append(i)

        print(f"\n[4] 結果:")
        if not valid_slices:
            print(" - ！！！有効なスライスは一枚も見つかりませんでした。！！！")
        else:
            print(f" - {len(valid_slices)}個の有効なスライスが見つかりました: {valid_slices}")

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    # コマンドラインからファイルパスを受け取る
    if len(sys.argv) > 1:
        file_path_arg = sys.argv[1]
        debug_label_file(file_path_arg)
    else:
        print("使用法: python scripts/debug_label.py /path/to/your/label.TIFF")
