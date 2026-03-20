# compare_depth.py
# test31の最適解（96%）で深度をレンダリングして、元画像の深度と比較
# v2: 水平オフセット深度を使用
#
# py -3.11 -u compare_depth.py

import math
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent

INPUT_IMAGE = BASE_DIR / "Image0.png"
INPUT_DEPTH = BASE_DIR / "Image0_depth.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"

OUT_DIR = BASE_DIR / "depth_comparison_v3"

# v5の曲率深度
CURVATURE_DEPTH_PATH = BASE_DIR / "depth_distortion_v5" / "depth_combined_40.png"

RENDER_SIZE = 1024
NORM_SIZE = 384

BBOX_MARGIN_RATIO = 0.12
CAMERA_FOV = 45.0
INITIAL_RX = -90.0

# test31の最適解（IoU 96.6%）
THETA = 95.274
PHI = -79.650
ROLL = 58.576
CAM_X = -0.4372
CAM_Y = -0.458
CAM_Z = 2.7599


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_obj(path):
    vertices, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idxs = []
                for p in parts[1:]:
                    idx = p.split("/")[0]
                    if idx:
                        idxs.append(int(idx) - 1)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def normalize_vertices(v):
    v = v.copy()
    v -= v.mean(axis=0)
    v /= np.abs(v).max() * 2
    return v


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]], dtype=np.float32)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]], dtype=np.float32)


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]], dtype=np.float32)


def render_depth(v, faces, theta, phi, roll, cam_x, cam_y, cam_z, size):
    """深度マップをレンダリング"""
    rx = INITIAL_RX + phi
    ry = -theta
    rz = roll
    R = rot_x(rx) @ rot_y(ry) @ rot_z(rz)

    verts = (R @ v.T).T
    vx = verts[:, 0] - cam_x
    vy = verts[:, 1] - cam_y
    vz = cam_z - verts[:, 2]
    vz_raw = vz.copy()
    vz = np.clip(vz, 0.1, None)

    f = size / (2.0 * math.tan(math.radians(CAMERA_FOV / 2.0)))
    px = (vx / vz) * f + size * 0.5
    py = size * 0.5 - (vy / vz) * f

    sil = np.zeros((size, size), dtype=np.uint8)
    zbuf = np.full((size, size), np.inf, dtype=np.float32)

    for a, b, c in faces:
        if vz[a] <= 0.1 or vz[b] <= 0.1 or vz[c] <= 0.1:
            continue

        pts = np.array([[px[a], py[a]], [px[b], py[b]], [px[c], py[c]]], dtype=np.int32)
        cv2.fillConvexPoly(sil, pts.reshape(-1, 1, 2), 255)

        avg_z = (vz_raw[a] + vz_raw[b] + vz_raw[c]) / 3.0
        mask_tri = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(mask_tri, pts.reshape(-1, 1, 2), 1)
        update = (mask_tri > 0) & (avg_z < zbuf)
        zbuf[update] = avg_z

    sil_bin = (sil > 0).astype(np.uint8)

    # 深度正規化
    valid = zbuf < np.inf
    depth = np.zeros((size, size), dtype=np.float32)
    if valid.any():
        d_min, d_max = zbuf[valid].min(), zbuf[valid].max()
        if d_max > d_min:
            depth[valid] = 1.0 - (zbuf[valid] - d_min) / (d_max - d_min)

    return sil_bin, depth


def preprocess_depth_image(depth_img, mask):
    """元画像の深度を前処理（Hueベース）"""
    hsv = cv2.cvtColor(depth_img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)

    valid = mask > 0
    depth_raw = np.zeros_like(hue)
    h_min, h_max = hue[valid].min(), hue[valid].max()
    if h_max > h_min:
        depth_raw[valid] = (hue[valid] - h_min) / (h_max - h_min)

    # 縞除去
    depth_blur = cv2.GaussianBlur(depth_raw, (51, 51), 0)
    depth_u8 = (depth_blur * 255).astype(np.uint8)
    depth_median = cv2.medianBlur(depth_u8, 21)
    depth_bilateral = cv2.bilateralFilter(depth_median, 15, 75, 75)
    depth_smooth = cv2.GaussianBlur(depth_bilateral.astype(np.float32), (31, 31), 0) / 255.0

    depth_smooth[~valid] = 0

    if valid.sum() > 0:
        d_min, d_max = depth_smooth[valid].min(), depth_smooth[valid].max()
        if d_max > d_min:
            depth_smooth[valid] = (depth_smooth[valid] - d_min) / (d_max - d_min)

    return depth_raw, depth_smooth


def extract_horizontal_offset_depth(mask):
    """水平オフセットから深度を抽出（各行で左右の位置を深度とする）"""
    H, W = mask.shape
    depth_map = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        row_mask = mask[y, :]
        if row_mask.sum() < 10:
            continue

        valid_x = np.where(row_mask > 0)[0]
        x_min, x_max = valid_x.min(), valid_x.max()

        if x_max - x_min < 10:
            continue

        # 各ピクセルで、「そのピクセルがマスク内で左右どの位置にあるか」
        for x in valid_x:
            # 0 (左端) から 1 (右端) にスケール
            depth_map[y, x] = (x - x_min) / (x_max - x_min)

    return depth_map


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(x0, y0, x1, y1, w, h, margin_ratio):
    bw, bh = x1 - x0, y1 - y0
    m = int(max(bw, bh) * margin_ratio)
    return max(0, x0 - m), max(0, y0 - m), min(w - 1, x1 + m), min(h - 1, y1 + m)


def crop_and_resize(img, bbox, size, interpolation=cv2.INTER_LINEAR):
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1+1, x0:x1+1]
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def main():
    ensure_dir(OUT_DIR)

    print("=== Depth Comparison ===")
    print(f"theta={THETA}, phi={PHI}, roll={ROLL}")
    print(f"cam=({CAM_X}, {CAM_Y}, {CAM_Z})")

    # 画像読み込み
    img = imread_unicode(INPUT_IMAGE)
    depth_img = imread_unicode(INPUT_DEPTH)
    H, W = img.shape[:2]

    # マスク作成
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # 元画像の深度
    depth_raw, depth_hue = preprocess_depth_image(depth_img, mask)

    # 水平オフセット深度（新方式）
    depth_offset = extract_horizontal_offset_depth(mask)

    # v5の曲率深度を読み込み
    depth_curv_img = cv2.imread(str(CURVATURE_DEPTH_PATH), cv2.IMREAD_GRAYSCALE)
    if depth_curv_img is not None:
        depth_curvature = depth_curv_img.astype(np.float32) / 255.0
    else:
        depth_curvature = None
        print("Warning: Could not load curvature depth")

    # バウンディングボックス
    bb = bbox_from_mask(mask)
    x0, y0, x1, y1 = expand_bbox(*bb, W, H, BBOX_MARGIN_RATIO)

    # メッシュ読み込み
    print(f"Loading: {MODEL_PATH.name}")
    v, faces = load_obj(MODEL_PATH)
    v = normalize_vertices(v)
    print(f"Vertices: {len(v)}, Faces: {len(faces)}")

    # レンダリング
    print("Rendering...")
    sil_render, depth_render = render_depth(v, faces, THETA, PHI, ROLL, CAM_X, CAM_Y, CAM_Z, RENDER_SIZE)

    # レンダリング結果のバウンディングボックス
    bb_render = bbox_from_mask(sil_render)
    bx0, by0, bx1, by1 = expand_bbox(*bb_render, RENDER_SIZE, RENDER_SIZE, BBOX_MARGIN_RATIO)

    # 正規化サイズにリサイズ
    target_sil = crop_and_resize(mask, (x0, y0, x1, y1), NORM_SIZE, cv2.INTER_NEAREST)
    target_sil = (target_sil > 0).astype(np.uint8)

    target_depth_raw = crop_and_resize(depth_raw, (x0, y0, x1, y1), NORM_SIZE)
    target_depth_hue = crop_and_resize(depth_hue, (x0, y0, x1, y1), NORM_SIZE)
    target_depth_offset = crop_and_resize(depth_offset, (x0, y0, x1, y1), NORM_SIZE)
    if depth_curvature is not None:
        target_depth_curv = crop_and_resize(depth_curvature, (x0, y0, x1, y1), NORM_SIZE)
        target_depth_curv[target_sil == 0] = 0
    else:
        target_depth_curv = None
    target_depth_raw[target_sil == 0] = 0
    target_depth_hue[target_sil == 0] = 0
    target_depth_offset[target_sil == 0] = 0

    render_sil_norm = crop_and_resize(sil_render.astype(np.uint8), (bx0, by0, bx1, by1), NORM_SIZE, cv2.INTER_NEAREST)
    render_sil_norm = (render_sil_norm > 0).astype(np.uint8)

    render_depth_norm = crop_and_resize(depth_render, (bx0, by0, bx1, by1), NORM_SIZE)
    render_depth_norm[render_sil_norm == 0] = 0

    # IoU計算
    inter = np.logical_and(target_sil, render_sil_norm).sum()
    union = np.logical_or(target_sil, render_sil_norm).sum()
    iou = inter / union if union > 0 else 0

    print(f"\nIoU: {iou:.4f} ({iou*100:.2f}%)")

    # === 出力 ===

    # 1. シルエット比較
    overlay = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    overlay[:, :, 2] = target_sil * 255       # Red: target
    overlay[:, :, 1] = render_sil_norm * 255  # Green: render
    cv2.imwrite(str(OUT_DIR / "1_silhouette_overlay.png"), overlay)

    # 2. 元画像の深度（Hue、平滑化後）
    target_depth_hue_color = cv2.applyColorMap((target_depth_hue * 255).astype(np.uint8), cv2.COLORMAP_JET)
    target_depth_hue_color[target_sil == 0] = 0
    cv2.imwrite(str(OUT_DIR / "2_target_depth_hue.png"), target_depth_hue_color)

    # 3. 水平オフセット深度（新方式）
    target_depth_offset_color = cv2.applyColorMap((target_depth_offset * 255).astype(np.uint8), cv2.COLORMAP_JET)
    target_depth_offset_color[target_sil == 0] = 0
    cv2.imwrite(str(OUT_DIR / "3_target_depth_offset.png"), target_depth_offset_color)

    # 4. レンダリングした深度
    render_depth_color = cv2.applyColorMap((render_depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    render_depth_color[render_sil_norm == 0] = 0
    cv2.imwrite(str(OUT_DIR / "4_render_depth.png"), render_depth_color)

    # === 水平オフセット深度との比較 ===
    overlap_mask = (target_sil > 0) & (render_sil_norm > 0)

    # 方向を合わせる（両方試す）
    mse_normal = np.mean((target_depth_offset[overlap_mask] - render_depth_norm[overlap_mask]) ** 2)
    mse_inv = np.mean((target_depth_offset[overlap_mask] - (1 - render_depth_norm[overlap_mask])) ** 2)

    print(f"\n=== Horizontal Offset vs Render ===")
    print(f"MSE (normal):   {mse_normal:.6f}")
    print(f"MSE (inverted): {mse_inv:.6f}")

    if mse_inv < mse_normal:
        print("-> Inverted matches better")
        render_for_compare = 1 - render_depth_norm
        best_mse = mse_inv
    else:
        print("-> Normal matches better")
        render_for_compare = render_depth_norm
        best_mse = mse_normal

    render_for_compare[render_sil_norm == 0] = 0
    render_compare_color = cv2.applyColorMap((render_for_compare * 255).astype(np.uint8), cv2.COLORMAP_JET)
    render_compare_color[render_sil_norm == 0] = 0

    # 5. 水平オフセット vs レンダリング（横並び）
    comparison_offset = np.hstack([target_depth_offset_color, render_compare_color])
    cv2.imwrite(str(OUT_DIR / "5_offset_vs_render.png"), comparison_offset)

    # 6. 差分ヒートマップ
    depth_diff = np.abs(target_depth_offset - render_for_compare)
    depth_diff[~overlap_mask] = 0
    depth_diff_color = cv2.applyColorMap((depth_diff * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    depth_diff_color[~overlap_mask] = 0
    cv2.imwrite(str(OUT_DIR / "6_depth_difference.png"), depth_diff_color)

    # === Hue深度との比較（参考） ===
    mse_hue_normal = np.mean((target_depth_hue[overlap_mask] - render_depth_norm[overlap_mask]) ** 2)
    mse_hue_inv = np.mean((target_depth_hue[overlap_mask] - (1 - render_depth_norm[overlap_mask])) ** 2)
    print(f"\n=== Hue vs Render (reference) ===")
    print(f"MSE (normal):   {mse_hue_normal:.6f}")
    print(f"MSE (inverted): {mse_hue_inv:.6f}")

    # === 曲率深度との比較 ===
    if target_depth_curv is not None:
        mse_curv_normal = np.mean((target_depth_curv[overlap_mask] - render_depth_norm[overlap_mask]) ** 2)
        mse_curv_inv = np.mean((target_depth_curv[overlap_mask] - (1 - render_depth_norm[overlap_mask])) ** 2)
        print(f"\n=== Curvature Depth (v5) vs Render ===")
        print(f"MSE (normal):   {mse_curv_normal:.6f}")
        print(f"MSE (inverted): {mse_curv_inv:.6f}")

        # 曲率深度のカラーマップ
        target_curv_color = cv2.applyColorMap((target_depth_curv * 255).astype(np.uint8), cv2.COLORMAP_JET)
        target_curv_color[target_sil == 0] = 0
        cv2.imwrite(str(OUT_DIR / "8_target_depth_curvature.png"), target_curv_color)

        # 曲率 vs レンダリング
        comparison_curv = np.hstack([target_curv_color, render_compare_color])
        cv2.imwrite(str(OUT_DIR / "9_curvature_vs_render.png"), comparison_curv)

    # 7. 全体比較（2x2）
    # 上段: Hue深度 vs レンダリング
    # 下段: オフセット深度 vs レンダリング
    row1 = np.hstack([target_depth_hue_color, render_depth_color])
    row2 = np.hstack([target_depth_offset_color, render_compare_color])
    full_comparison = np.vstack([row1, row2])
    cv2.imwrite(str(OUT_DIR / "7_full_comparison.png"), full_comparison)

    # グレースケール出力
    cv2.imwrite(str(OUT_DIR / "target_offset_gray.png"), (target_depth_offset * 255).astype(np.uint8))
    cv2.imwrite(str(OUT_DIR / "render_depth_gray.png"), (render_for_compare * 255).astype(np.uint8))

    print(f"\n=== Summary ===")
    print(f"IoU: {iou:.4f} ({iou*100:.2f}%)")
    print(f"Depth MSE (offset): {best_mse:.6f}")

    print(f"\nOutput: {OUT_DIR}")
    print("Files:")
    print("  1_silhouette_overlay.png - シルエット比較")
    print("  2_target_depth_hue.png   - Hue深度")
    print("  3_target_depth_offset.png- 水平オフセット深度")
    print("  4_render_depth.png       - レンダリング深度")
    print("  5_offset_vs_render.png   - オフセット vs レンダリング")
    print("  7_full_comparison.png    - 全体比較")


if __name__ == "__main__":
    main()
