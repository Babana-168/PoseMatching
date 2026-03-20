# overlay_color.py
# 元画像と3Dモデルのカラーオーバーレイ

import math
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).parent
INPUT_IMAGE = BASE_DIR / "Image0.png"
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit_low.obj"
OUT_DIR = BASE_DIR / "rotation_results_test53"

RENDER_SIZE = 512
CAMERA_FOV = 45.0
INITIAL_RX = -90.0
BBOX_MARGIN_RATIO = 0.12

# test53の最良結果
THETA = 94.754
PHI = -79.450
ROLL = 58.756
CAM_X = -0.4372
CAM_Y = -0.418
CAM_Z = 2.7199


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

    return sil


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(x0, y0, x1, y1, w, h, margin_ratio):
    bw, bh = x1 - x0, y1 - y0
    m = int(max(bw, bh) * margin_ratio)
    return max(0, x0 - m), max(0, y0 - m), min(w - 1, x1 + m), min(h - 1, y1 + m)


def main():
    # 元画像読み込み
    img = imread_unicode(INPUT_IMAGE)
    H, W = img.shape[:2]

    # ターゲットマスク作成
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # メッシュ
    v_np, faces_np = load_obj(MODEL_PATH)
    v_np = normalize_vertices(v_np)

    # レンダリング
    sil = render_silhouette(v_np, faces_np, THETA, PHI, ROLL, CAM_X, CAM_Y, CAM_Z, RENDER_SIZE)

    # モデルシルエットのバウンディングボックス
    bb_model = bbox_from_mask(sil)
    mx0, my0, mx1, my1 = expand_bbox(*bb_model, RENDER_SIZE, RENDER_SIZE, BBOX_MARGIN_RATIO)

    # モデルシルエットをターゲットのバウンディングボックスに合わせてリサイズ
    sil_crop = sil[my0:my1+1, mx0:mx1+1]
    target_h, target_w = y1 - y0 + 1, x1 - x0 + 1
    sil_resized = cv2.resize(sil_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # オーバーレイ作成（元画像サイズ）
    overlay = img.copy()

    # モデルのシルエット輪郭を緑で描画
    contours, _ = cv2.findContours(sil_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 輪郭の座標をオフセット
    for cnt in contours:
        cnt[:, 0, 0] += x0
        cnt[:, 0, 1] += y0

    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # 半透明オーバーレイも作成
    overlay_alpha = img.copy()
    model_region = np.zeros_like(img)
    model_region[y0:y1+1, x0:x1+1][sil_resized > 0] = [0, 255, 0]  # 緑
    overlay_alpha = cv2.addWeighted(overlay_alpha, 0.7, model_region, 0.3, 0)

    # 保存
    cv2.imwrite(str(OUT_DIR / "overlay_contour.png"), overlay)
    cv2.imwrite(str(OUT_DIR / "overlay_alpha.png"), overlay_alpha)

    # 並べて比較
    compare = np.hstack([img, overlay, overlay_alpha])
    cv2.imwrite(str(OUT_DIR / "compare_color.png"), compare)

    print(f"Saved to {OUT_DIR}")
    print("  overlay_contour.png - 輪郭線")
    print("  overlay_alpha.png - 半透明")
    print("  compare_color.png - 並べて比較")


if __name__ == "__main__":
    main()
