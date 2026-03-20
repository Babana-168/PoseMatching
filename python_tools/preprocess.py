"""
画像前処理: Image0.pngから余計なノイズを除去してウサギだけを抽出
"""
import numpy as np
from PIL import Image
from scipy import ndimage

# ------------------------
# 設定
# ------------------------
INPUT_IMAGE = "Image0.png"
OUTPUT_IMAGE = "Image0_clean.png"
THRESHOLD = 15  # 二値化の閾値

# ------------------------
# 画像読み込み・二値化
# ------------------------
img = Image.open(INPUT_IMAGE).convert("L")
img_array = np.array(img)

# 二値化
binary = (img_array > THRESHOLD).astype(np.uint8)

# ------------------------
# 最大連結成分を抽出（ウサギ本体だけを残す）
# ------------------------
labeled, num_features = ndimage.label(binary)
print(f"検出された連結成分数: {num_features}")

if num_features > 0:
    # 各連結成分のサイズを計算
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))

    # 最大の連結成分を見つける
    largest_component = np.argmax(component_sizes) + 1

    # 最大連結成分だけを残す
    clean_mask = (labeled == largest_component).astype(np.uint8)

    print(f"最大連結成分のサイズ: {int(component_sizes[largest_component - 1])} ピクセル")
else:
    clean_mask = binary

# ------------------------
# 元の輝度値を保持してマスク適用
# ------------------------
clean_image = img_array * clean_mask

# ------------------------
# 保存
# ------------------------
result = Image.fromarray(clean_image)
result.save(OUTPUT_IMAGE)

print(f"前処理完了: {OUTPUT_IMAGE}")
print(f"元の白ピクセル数: {binary.sum()}")
print(f"処理後の白ピクセル数: {clean_mask.sum()}")
