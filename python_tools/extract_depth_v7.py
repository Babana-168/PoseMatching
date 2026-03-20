# extract_depth_v7.py
# 縞の位置追跡による深度抽出
#
# 各行で縞の境界位置を検出し、
# 基準行からの「ずれ」を深度として出力
#
# py -3.11 -u extract_depth_v7.py

import numpy as np
import cv2
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

BASE_DIR = Path(__file__).parent
INPUT_DEPTH = BASE_DIR / "Image0_depth.png"
INPUT_IMAGE = BASE_DIR / "Image0.png"
OUT_DIR = BASE_DIR / "depth_distortion_v7"


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    return mask


def detect_stripes_per_row(depth_img, mask):
    """
    各行で縞の境界位置を検出
    """
    H, W = depth_img.shape[:2]

    # 緑チャンネル（縞のコントラストが高い）
    g = depth_img[:, :, 1].astype(np.float32)

    stripe_positions = {}  # stripe_positions[y] = [x1, x2, ...]

    for y in range(H):
        row_mask = mask[y, :]
        if row_mask.sum() < 50:
            continue

        valid_x = np.where(row_mask > 0)[0]
        x_min, x_max = valid_x.min(), valid_x.max()

        if x_max - x_min < 50:
            continue

        # この行の緑チャンネル
        row_g = g[y, x_min:x_max+1]

        # スムージング（控えめに）
        row_smooth = gaussian_filter1d(row_g, sigma=1)

        # 勾配
        grad = np.gradient(row_smooth)

        # ピーク検出（緩いパラメータ）
        peaks_pos, _ = find_peaks(grad, height=3, distance=5)
        peaks_neg, _ = find_peaks(-grad, height=3, distance=5)

        all_peaks = np.sort(np.concatenate([peaks_pos, peaks_neg])) + x_min

        if len(all_peaks) >= 3:
            stripe_positions[y] = all_peaks.tolist()

    return stripe_positions


def compute_depth_from_stripes(stripe_positions, mask):
    """
    縞の位置から深度を計算

    各縞について、基準行からの水平方向のずれを計算
    ずれ量 = 深度情報
    """
    H, W = mask.shape
    depth_map = np.zeros((H, W), dtype=np.float32)

    valid_rows = sorted(stripe_positions.keys())
    if not valid_rows:
        return depth_map

    # 基準行（中央）を選択
    mid_idx = len(valid_rows) // 2
    ref_y = valid_rows[mid_idx]
    ref_stripes = np.array(stripe_positions[ref_y])

    print(f"Reference row {ref_y}: {len(ref_stripes)} stripes")

    # 各縞の番号を割り当て
    n_stripes = len(ref_stripes)

    # 各行で、縞ごとの水平位置を記録
    stripe_x_per_row = {}  # stripe_x_per_row[y][stripe_idx] = x

    for y in valid_rows:
        stripes = np.array(stripe_positions[y])
        stripe_x_per_row[y] = {}

        # マッチング：最も近い基準縞に対応付け
        for i, x in enumerate(stripes):
            # 最も近い基準縞を探す
            dists = np.abs(ref_stripes - x)
            best_idx = np.argmin(dists)

            if dists[best_idx] < 30:  # 30ピクセル以内
                # 既に割り当てられている場合は、より近い方を採用
                if best_idx in stripe_x_per_row[y]:
                    if dists[best_idx] < abs(stripe_x_per_row[y][best_idx] - ref_stripes[best_idx]):
                        stripe_x_per_row[y][best_idx] = x
                else:
                    stripe_x_per_row[y][best_idx] = x

    # 各縞の「基準からのずれ」を計算
    for stripe_idx in range(n_stripes):
        # この縞の全行での位置を収集
        y_list = []
        x_list = []
        offset_list = []

        for y in valid_rows:
            if stripe_idx in stripe_x_per_row[y]:
                x = stripe_x_per_row[y][stripe_idx]
                ref_x = ref_stripes[stripe_idx]
                offset = x - ref_x

                y_list.append(y)
                x_list.append(x)
                offset_list.append(offset)

        if len(y_list) < 5:
            continue

        # この縞の周辺ピクセルに深度を設定
        for i, (y, x, offset) in enumerate(zip(y_list, x_list, offset_list)):
            # 次の縞までの範囲
            if stripe_idx < n_stripes - 1 and (stripe_idx + 1) in stripe_x_per_row.get(y, {}):
                x_end = stripe_x_per_row[y][stripe_idx + 1]
            else:
                x_end = x + 15  # デフォルト幅

            for xx in range(max(0, int(x) - 2), min(W, int(x_end))):
                if mask[y, xx] > 0:
                    depth_map[y, xx] = offset

    return depth_map


def fill_and_smooth(depth_map, mask):
    """穴埋めと平滑化"""
    H, W = depth_map.shape
    filled = depth_map.copy()

    # 水平方向の補間
    for y in range(H):
        row = filled[y, :]
        row_mask = mask[y, :]

        valid = (row != 0) & (row_mask > 0)
        if valid.sum() < 5:
            continue

        valid_x = np.where(valid)[0]
        valid_v = row[valid]

        all_x = np.where(row_mask > 0)[0]
        if len(all_x) > 0:
            interp_v = np.interp(all_x, valid_x, valid_v)
            filled[y, all_x] = interp_v

    # 垂直方向の補間
    for x in range(W):
        col = filled[:, x]
        col_mask = mask[:, x]

        valid = (col != 0) & (col_mask > 0)
        if valid.sum() < 5:
            continue

        valid_y = np.where(valid)[0]
        valid_v = col[valid]

        all_y = np.where(col_mask > 0)[0]
        if len(all_y) > 0:
            interp_v = np.interp(all_y, valid_y, valid_v)
            filled[all_y, x] = interp_v

    # 平滑化
    smooth = cv2.GaussianBlur(filled, (21, 21), 0)
    smooth[mask == 0] = 0

    return smooth


def normalize_depth(depth_map, mask):
    """深度を0-1に正規化"""
    valid = mask > 0
    if valid.sum() == 0:
        return np.zeros_like(depth_map)

    p_min = np.percentile(depth_map[valid], 2)
    p_max = np.percentile(depth_map[valid], 98)

    if p_max > p_min:
        norm = (depth_map - p_min) / (p_max - p_min)
        norm = np.clip(norm, 0, 1)
    else:
        norm = np.zeros_like(depth_map)

    norm[mask == 0] = 0
    return norm


def main():
    ensure_dir(OUT_DIR)

    print("=== Extract Depth v7: Stripe Position Tracking ===")

    depth_img = imread_unicode(INPUT_DEPTH)
    sil_img = imread_unicode(INPUT_IMAGE)

    H, W = depth_img.shape[:2]
    print(f"Image size: {W}x{H}")

    mask = get_mask(sil_img)

    # 縞の検出
    print("\nDetecting stripes...")
    stripe_positions = detect_stripes_per_row(depth_img, mask)
    print(f"Detected stripes in {len(stripe_positions)} rows")

    # 深度計算
    print("\nComputing depth from stripe offsets...")
    depth_raw = compute_depth_from_stripes(stripe_positions, mask)

    # 穴埋め・平滑化
    print("\nFilling and smoothing...")
    depth_smooth = fill_and_smooth(depth_raw, mask)

    # 正規化
    depth_norm = normalize_depth(depth_smooth, mask)

    # 反転版も生成
    depth_inv = 1.0 - depth_norm
    depth_inv[mask == 0] = 0

    # === 出力 ===

    # 1. 生の深度
    raw_color = cv2.applyColorMap((normalize_depth(depth_raw, mask) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    raw_color[mask == 0] = 0
    cv2.imwrite(str(OUT_DIR / "1_depth_raw.png"), raw_color)

    # 2. 平滑化深度
    smooth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    smooth_color[mask == 0] = 0
    cv2.imwrite(str(OUT_DIR / "2_depth_smooth.png"), smooth_color)

    # 3. 反転版
    inv_color = cv2.applyColorMap((depth_inv * 255).astype(np.uint8), cv2.COLORMAP_JET)
    inv_color[mask == 0] = 0
    cv2.imwrite(str(OUT_DIR / "3_depth_inverted.png"), inv_color)

    # 4. 比較
    comparison = np.hstack([raw_color, smooth_color, inv_color])
    cv2.imwrite(str(OUT_DIR / "4_comparison.png"), comparison)

    # グレースケール出力
    cv2.imwrite(str(OUT_DIR / "depth_smooth.png"), (depth_norm * 255).astype(np.uint8))
    cv2.imwrite(str(OUT_DIR / "depth_inverted.png"), (depth_inv * 255).astype(np.uint8))

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
