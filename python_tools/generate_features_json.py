"""
3Dモデルの全角度パターンの特徴をJSONに保存（マルチスレッド版）

各角度で以下の特徴を記録:
- バウンディングボックスの縦横比 (aspect_ratio)
- 正規化されたシルエット画像（小さいサイズ）
- Huモーメント（形状特徴）
- 重心の相対位置
"""

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from pathlib import Path
import warnings
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings('ignore')

# スレッド数（CPUコア数）
NUM_THREADS = multiprocessing.cpu_count()

# ===== 設定 =====
RENDER_SIZE = (128, 128)  # 正規化用のレンダリングサイズ
FEATURE_SIZE = (32, 32)   # 特徴として保存するサイズ
TARGET_FACES = 15000
INITIAL_RX = -90

# 角度の範囲（2度刻み）
THETA_RANGE = range(-180, 180, 2)   # 水平方向 180パターン
PHI_RANGE = range(-90, 91, 2)       # 垂直方向 91パターン
ROLL_RANGE = range(-60, 61, 5)      # ロール 25パターン
# 合計: 180 * 91 * 25 = 409,500 パターン


def create_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """ZYX順序の回転行列を作成（Z回転→Y回転→X回転の順に適用）"""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # ZYX順序: R = Rx @ Ry @ Rz
    R = np.array([
        [cy * cz, -cy * sz, sy],
        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
    ], dtype=np.float32)

    return R


def spherical_to_rotation(theta_deg, phi_deg, roll_deg=0):
    rx = INITIAL_RX + phi_deg
    ry = -theta_deg
    rz = roll_deg
    return rx, ry, rz


def render_silhouette(vertices, faces, rx, ry, rz, size):
    """シルエットをレンダリング（中心配置、自動スケール）"""
    width, height = size
    R = create_rotation_matrix(rx, ry, rz)
    rotated = vertices @ R.T

    # バウンディングボックスを取得
    min_xy = rotated[:, :2].min(axis=0)
    max_xy = rotated[:, :2].max(axis=0)

    # スケールを計算（マージンを持たせる）
    extent = max_xy - min_xy
    scale = min(width, height) * 0.8 / max(extent[0], extent[1]) if max(extent) > 0 else 1

    # 中心に配置
    center = (min_xy + max_xy) / 2
    proj_x = (rotated[:, 0] - center[0]) * scale + width / 2
    proj_y = height / 2 - (rotated[:, 1] - center[1]) * scale

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    for face in faces:
        polygon = [(proj_x[face[0]], proj_y[face[0]]),
                   (proj_x[face[1]], proj_y[face[1]]),
                   (proj_x[face[2]], proj_y[face[2]])]
        draw.polygon(polygon, fill=255)

    return np.array(img, dtype=np.uint8) > 127


def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def compute_hu_moments(mask):
    """Huモーメントを計算（形状の特徴量）"""
    mask_float = mask.astype(np.float64)

    # 空間モーメント
    y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
    m00 = mask_float.sum()

    if m00 == 0:
        return [0] * 7

    m10 = (x * mask_float).sum()
    m01 = (y * mask_float).sum()

    # 重心
    cx = m10 / m00
    cy = m01 / m00

    # 中心モーメント
    xc = x - cx
    yc = y - cy

    mu20 = (xc**2 * mask_float).sum() / m00
    mu02 = (yc**2 * mask_float).sum() / m00
    mu11 = (xc * yc * mask_float).sum() / m00
    mu30 = (xc**3 * mask_float).sum() / m00
    mu03 = (yc**3 * mask_float).sum() / m00
    mu21 = (xc**2 * yc * mask_float).sum() / m00
    mu12 = (xc * yc**2 * mask_float).sum() / m00

    # 正規化中心モーメント
    n20 = mu20
    n02 = mu02
    n11 = mu11
    n30 = mu30
    n03 = mu03
    n21 = mu21
    n12 = mu12

    # Huモーメント（最初の4つを使用）
    hu1 = n20 + n02
    hu2 = (n20 - n02)**2 + 4*n11**2
    hu3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    hu4 = (n30 + n12)**2 + (n21 + n03)**2

    # 対数スケールに変換（より比較しやすくする）
    hu_moments = [hu1, hu2, hu3, hu4]
    hu_log = []
    for h in hu_moments:
        if h != 0:
            hu_log.append(float(-np.sign(h) * np.log10(abs(h) + 1e-10)))
        else:
            hu_log.append(0.0)

    return hu_log


def extract_features(mask):
    """シルエットから特徴を抽出"""
    rmin, rmax, cmin, cmax = get_bounding_box(mask)

    if rmax <= rmin or cmax <= cmin:
        return None

    height = rmax - rmin
    width = cmax - cmin

    # 特徴1: アスペクト比
    aspect_ratio = width / height if height > 0 else 1.0

    # 特徴2: 面積比（バウンディングボックスに対する）
    area = mask.sum()
    bbox_area = height * width
    fill_ratio = area / bbox_area if bbox_area > 0 else 0

    # 特徴3: 重心の相対位置（バウンディングボックス内）
    coords = np.where(mask)
    if len(coords[0]) == 0:
        rel_cy, rel_cx = 0.5, 0.5
    else:
        cy = np.mean(coords[0])
        cx = np.mean(coords[1])
        rel_cy = (cy - rmin) / height if height > 0 else 0.5
        rel_cx = (cx - cmin) / width if width > 0 else 0.5

    # 特徴4: Huモーメント
    hu_moments = compute_hu_moments(mask)

    # 特徴5: 正規化シルエット（小さい画像として保存）
    cropped = mask[rmin:rmax+1, cmin:cmax+1]
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    normalized = cropped_img.resize(FEATURE_SIZE, Image.NEAREST)
    normalized_arr = np.array(normalized) > 127
    # ビット列として圧縮
    normalized_bits = normalized_arr.flatten().tolist()

    return {
        "aspect_ratio": round(aspect_ratio, 4),
        "fill_ratio": round(fill_ratio, 4),
        "centroid": [round(rel_cx, 4), round(rel_cy, 4)],
        "hu_moments": [round(h, 4) for h in hu_moments],
        "silhouette": normalized_bits  # 32x32 = 1024 bits
    }


def load_3d_model(model_path, target_faces=TARGET_FACES):
    mesh = trimesh.load(model_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)

    mesh.vertices -= mesh.centroid
    scale = 1.0 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale

    if len(mesh.faces) > target_faces:
        np.random.seed(42)
        indices = np.random.choice(len(mesh.faces), target_faces, replace=False)
        selected_faces = mesh.faces[indices]
        unique_vertices = np.unique(selected_faces.flatten())
        vertex_map = {old: new for new, old in enumerate(unique_vertices)}
        new_vertices = mesh.vertices[unique_vertices]
        new_faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    return mesh


def process_angle(args):
    """1つの角度パターンを処理（スレッドワーカー用）"""
    theta, phi, roll, vertices, faces = args
    rx, ry, rz = spherical_to_rotation(theta, phi, roll)
    silhouette = render_silhouette(vertices, faces, rx, ry, rz, RENDER_SIZE)
    features = extract_features(silhouette)

    if features is not None:
        features["theta"] = theta
        features["phi"] = phi
        features["roll"] = roll
        return features
    return None


def main():
    BASE_PATH = Path(__file__).resolve().parent
    MODEL_PATH = BASE_PATH / "models_rabit_dae" / "rabit.dae"
    OUTPUT_PATH = BASE_PATH / "model_features.json"

    print("=" * 60)
    print("  3Dモデル特徴量JSON生成（マルチスレッド版）")
    print("=" * 60)
    print(f"  使用スレッド数: {NUM_THREADS}")

    # モデル読み込み
    print("\n[1/3] モデル読み込み...")
    mesh = load_3d_model(MODEL_PATH)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces
    print(f"  面数: {len(faces):,}")

    # 角度パターンを生成
    angle_patterns = []
    for theta in THETA_RANGE:
        for phi in PHI_RANGE:
            for roll in ROLL_RANGE:
                angle_patterns.append((theta, phi, roll, vertices, faces))

    total = len(angle_patterns)
    print(f"\n[2/3] 特徴量計算... (全{total:,}パターン)")

    features_list = []
    count = 0
    start_time = time.time()

    # マルチスレッドで処理
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process_angle, args): args for args in angle_patterns}

        for future in as_completed(futures):
            count += 1
            result = future.result()
            if result is not None:
                features_list.append(result)

            if count % 5000 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / count * (total - count)
                print(f"  進捗: {count:,}/{total:,} ({count/total*100:.1f}%) - "
                      f"経過: {elapsed:.0f}秒, 残り: {eta:.0f}秒")

    print(f"\n  完了！ 有効パターン: {len(features_list):,}")

    # 角度順にソート
    features_list.sort(key=lambda x: (x["theta"], x["phi"], x["roll"]))

    # JSON保存
    print(f"\n[3/3] JSON保存...")

    output_data = {
        "model": str(MODEL_PATH),
        "render_size": list(RENDER_SIZE),
        "feature_size": list(FEATURE_SIZE),
        "rotation_order": "ZYX",  # 回転順序を記録
        "angle_step": {
            "theta": THETA_RANGE.step if hasattr(THETA_RANGE, 'step') else 2,
            "phi": PHI_RANGE.step if hasattr(PHI_RANGE, 'step') else 2,
            "roll": ROLL_RANGE.step if hasattr(ROLL_RANGE, 'step') else 5
        },
        "total_patterns": len(features_list),
        "features": features_list
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f)

    file_size = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"  保存完了: {OUTPUT_PATH}")
    print(f"  ファイルサイズ: {file_size:.1f} MB")

    total_time = time.time() - start_time
    print(f"\n総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print("=" * 60)


if __name__ == "__main__":
    main()
