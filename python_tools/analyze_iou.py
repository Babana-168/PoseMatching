# analyze_iou.py
# IoUの詳細分析 - 不一致部分の可視化と分析
#
# py -3.11 analyze_iou.py

import math
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent

INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit_low.obj"

OUT_DIR = BASE_DIR / "iou_analysis"

RENDER_SIZE = 512
NORM_SIZE = 256

BBOX_MARGIN_RATIO = 0.12
CAMERA_FOV = 45.0
INITIAL_RX = -90.0

# test49の最適解
BEST_THETA = 94.674
BEST_PHI = -79.350
BEST_ROLL = 58.876
BEST_CAM_X = -0.4372
BEST_CAM_Y = -0.418
BEST_CAM_Z = 2.7199


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def imread_unicode(path):
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


def render_silhouette(v, faces, theta, phi, roll, cam_x, cam_y, cam_z, size):
    rx = INITIAL_RX + phi
    ry = -theta
    rz = roll

    R = rot_x(rx) @ rot_y(ry) @ rot_z(rz)
    verts = (R @ v.T).T

    vx = verts[:, 0] - cam_x
    vy = verts[:, 1] - cam_y
    vz = cam_z - verts[:, 2]
    vz = np.clip(vz, 0.1, None)

    f = size / (2.0 * math.tan(math.radians(CAMERA_FOV / 2.0)))
    px = (vx / vz) * f + size * 0.5
    py = size * 0.5 - (vy / vz) * f

    sil = np.zeros((size, size), dtype=np.uint8)
    for a, b, c in faces:
        pts = np.array([[px[a], py[a]], [px[b], py[b]], [px[c], py[c]]], dtype=np.int32)
        cv2.fillConvexPoly(sil, pts.reshape(-1, 1, 2), 255)

    return (sil > 0).astype(np.float32)


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

    print("=" * 60)
    print("IoU Analysis")
    print("=" * 60)

    # 画像読み込み
    sil_img = imread_unicode(INPUT_IMAGE)
    H, W = sil_img.shape[:2]

    # ターゲットマスク
    gray = cv2.cvtColor(sil_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # バウンディングボックス
    bb = bbox_from_mask(mask)
    x0, y0, x1, y1 = expand_bbox(*bb, W, H, BBOX_MARGIN_RATIO)

    target_sil = crop_and_resize(mask, (x0, y0, x1, y1), NORM_SIZE, cv2.INTER_NEAREST)
    target_sil = (target_sil > 0).astype(np.float32)

    # メッシュ
    v_np, faces_np = load_obj(MODEL_PATH)
    v_np = normalize_vertices(v_np)

    # レンダリング
    sil = render_silhouette(v_np, faces_np, BEST_THETA, BEST_PHI, BEST_ROLL,
                            BEST_CAM_X, BEST_CAM_Y, BEST_CAM_Z, RENDER_SIZE)
    bb_render = bbox_from_mask(sil)
    bx0, by0, bx1, by1 = expand_bbox(*bb_render, RENDER_SIZE, RENDER_SIZE, BBOX_MARGIN_RATIO)
    model_sil = crop_and_resize(sil, (bx0, by0, bx1, by1), NORM_SIZE, cv2.INTER_NEAREST)

    # マスクをbool化
    target = target_sil > 0
    model = model_sil > 0

    # 詳細分析
    intersection = np.logical_and(target, model).sum()
    union = np.logical_or(target, model).sum()
    target_only = np.logical_and(target, ~model).sum()  # ターゲットにあってモデルにない
    model_only = np.logical_and(model, ~target).sum()   # モデルにあってターゲットにない

    iou = intersection / union if union > 0 else 0

    print(f"\n[Pixel counts]")
    print(f"  Target pixels:       {target.sum():6d}")
    print(f"  Model pixels:        {model.sum():6d}")
    print(f"  Intersection:        {intersection:6d}")
    print(f"  Union:               {union:6d}")
    print(f"  Target only (FN):    {target_only:6d}  (赤: モデルが足りない)")
    print(f"  Model only (FP):     {model_only:6d}  (緑: モデルがはみ出し)")

    print(f"\n[IoU breakdown]")
    print(f"  IoU:                 {iou:.6f} ({iou*100:.4f}%)")
    print(f"  100% - IoU:          {(1-iou)*100:.4f}%")
    print(f"  Error pixels:        {target_only + model_only:6d}")
    print(f"    Target only rate:  {target_only/union*100:.4f}%")
    print(f"    Model only rate:   {model_only/union*100:.4f}%")

    # 理論上の最大IoU（モデルとターゲットが同じ面積の場合）
    print(f"\n[Theoretical analysis]")
    pixel_diff = abs(target.sum() - model.sum())
    print(f"  Pixel count difference: {pixel_diff}")
    # 面積差がある場合、完全重畳でもIoU < 100%
    if target.sum() > model.sum():
        max_possible_iou = model.sum() / target.sum()
    else:
        max_possible_iou = target.sum() / model.sum()
    print(f"  If perfectly aligned (area diff only): {max_possible_iou*100:.4f}%")

    # 出力画像
    # 1. オーバーレイ（従来通り）
    overlay = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    overlay[:, :, 2] = (target * 255).astype(np.uint8)   # 赤 = ターゲット
    overlay[:, :, 1] = (model * 255).astype(np.uint8)    # 緑 = モデル
    cv2.imwrite(str(OUT_DIR / "1_overlay.png"), overlay)

    # 2. 不一致部分のみ（拡大版）
    mismatch = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    mismatch[np.logical_and(target, ~model), 2] = 255   # 赤 = FN (ターゲットのみ)
    mismatch[np.logical_and(model, ~target), 1] = 255   # 緑 = FP (モデルのみ)
    mismatch[np.logical_and(target, model), :] = 50     # グレー = 一致
    cv2.imwrite(str(OUT_DIR / "2_mismatch.png"), mismatch)

    # 3. 輪郭のみの比較
    target_u8 = (target * 255).astype(np.uint8)
    model_u8 = (model * 255).astype(np.uint8)
    target_contour = cv2.Canny(target_u8, 100, 200)
    model_contour = cv2.Canny(model_u8, 100, 200)

    contour_overlay = np.zeros((NORM_SIZE, NORM_SIZE, 3), dtype=np.uint8)
    contour_overlay[:, :, 2] = target_contour   # 赤 = ターゲット輪郭
    contour_overlay[:, :, 1] = model_contour    # 緑 = モデル輪郭
    cv2.imwrite(str(OUT_DIR / "3_contours.png"), contour_overlay)

    # 4. 各シルエット単体
    cv2.imwrite(str(OUT_DIR / "4_target.png"), target_u8)
    cv2.imwrite(str(OUT_DIR / "5_model.png"), model_u8)

    # 5. 元画像とモデルの重ね合わせ（元サイズ）
    sil_full = render_silhouette(v_np, faces_np, BEST_THETA, BEST_PHI, BEST_ROLL,
                                  BEST_CAM_X, BEST_CAM_Y, BEST_CAM_Z, RENDER_SIZE)
    # ターゲットのBBに合わせてリサイズ・配置
    bb_w, bb_h = x1 - x0, y1 - y0
    sil_resized = cv2.resize(model_sil, (bb_w, bb_h), interpolation=cv2.INTER_NEAREST)

    overlay_original = sil_img.copy()
    sil_color = np.zeros((bb_h, bb_w, 3), dtype=np.uint8)
    sil_color[:, :, 1] = (sil_resized * 255).astype(np.uint8)

    # 半透明合成
    roi = overlay_original[y0:y1, x0:x1]
    overlay_original[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.7, sil_color, 0.3, 0)
    cv2.imwrite(str(OUT_DIR / "6_original_overlay.png"), overlay_original)

    print(f"\n[Output files]")
    print(f"  {OUT_DIR}/1_overlay.png        - 赤=ターゲット, 緑=モデル, 黄=一致")
    print(f"  {OUT_DIR}/2_mismatch.png       - 赤=FN, 緑=FP, グレー=一致")
    print(f"  {OUT_DIR}/3_contours.png       - 輪郭線のみ")
    print(f"  {OUT_DIR}/4_target.png         - ターゲットシルエット")
    print(f"  {OUT_DIR}/5_model.png          - モデルシルエット")
    print(f"  {OUT_DIR}/6_original_overlay.png - 元画像との重ね合わせ")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
