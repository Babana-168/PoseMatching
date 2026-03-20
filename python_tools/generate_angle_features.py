"""
3Dモデルの各角度での特徴量を事前計算してJSONに保存
20度刻みで全方向をカバー
"""

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import json
from pathlib import Path
from scipy import ndimage
import time

# ===== 設定 =====
ANGLE_STEP = 20  # 20度刻み
RENDER_SIZE = (128, 128)  # 特徴計算用の解像度
TARGET_FACES = 10000


def create_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """回転行列を生成"""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    R = np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy]
    ], dtype=np.float32)

    return R


def render_silhouette(vertices, faces, rx, ry, rz, width, height, scale_factor=0.8):
    """シルエットレンダリング"""
    R = create_rotation_matrix(rx, ry, rz)
    rotated = vertices @ R.T

    min_dim = min(width, height)
    scale = min_dim * scale_factor

    proj_x = rotated[:, 0] * scale + width / 2
    proj_y = height / 2 - rotated[:, 1] * scale

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    for face in faces:
        polygon = [(proj_x[face[0]], proj_y[face[0]]),
                   (proj_x[face[1]], proj_y[face[1]]),
                   (proj_x[face[2]], proj_y[face[2]])]
        draw.polygon(polygon, fill=255)

    return np.array(img, dtype=np.uint8) > 127


def get_bounding_box(mask):
    """バウンディングボックスを取得"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(rmax), int(cmin), int(cmax)


def extract_features(mask):
    """マスクから特徴量を抽出"""
    rmin, rmax, cmin, cmax = get_bounding_box(mask)

    # 有効なマスクかチェック
    if rmax <= rmin or cmax <= cmin:
        return None

    height = rmax - rmin + 1
    width = cmax - cmin + 1

    # アスペクト比
    aspect_ratio = height / width if width > 0 else 0

    # 面積（正規化）
    area = mask.sum() / (mask.shape[0] * mask.shape[1])

    # 重心（正規化: 0-1）
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    centroid_y = np.mean(coords[0]) / mask.shape[0]
    centroid_x = np.mean(coords[1]) / mask.shape[1]

    # エッジヒストグラム（輪郭の形状を8方向で表現）
    sobel_x = ndimage.sobel(mask.astype(float), axis=1)
    sobel_y = ndimage.sobel(mask.astype(float), axis=0)

    # 8方向のエッジ強度
    edge_histogram = []
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        directional = sobel_x * np.cos(rad) + sobel_y * np.sin(rad)
        edge_histogram.append(float(np.abs(directional).sum()))

    # 正規化
    hist_sum = sum(edge_histogram)
    if hist_sum > 0:
        edge_histogram = [h / hist_sum for h in edge_histogram]

    # バウンディングボックスの位置（正規化）
    bbox_center_x = (cmin + cmax) / 2 / mask.shape[1]
    bbox_center_y = (rmin + rmax) / 2 / mask.shape[0]

    return {
        "aspect_ratio": round(aspect_ratio, 4),
        "area": round(area, 4),
        "centroid": [round(centroid_x, 4), round(centroid_y, 4)],
        "bbox_size": [width, height],
        "bbox_center": [round(bbox_center_x, 4), round(bbox_center_y, 4)],
        "edge_histogram": [round(h, 4) for h in edge_histogram]
    }


def load_3d_model(model_path, target_faces=TARGET_FACES):
    """3Dモデル読み込み"""
    mesh = trimesh.load(model_path, force='mesh')

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)

    mesh.vertices -= mesh.centroid
    scale = 1.0 / mesh.bounding_box.extents.max()
    mesh.vertices *= scale

    original_faces = len(mesh.faces)

    if len(mesh.faces) > target_faces:
        print(f"  メッシュ簡略化: {original_faces:,} → {target_faces:,} 面")
        np.random.seed(42)
        indices = np.random.choice(len(mesh.faces), target_faces, replace=False)
        selected_faces = mesh.faces[indices]
        unique_vertices = np.unique(selected_faces.flatten())
        vertex_map = {old: new for new, old in enumerate(unique_vertices)}
        new_vertices = mesh.vertices[unique_vertices]
        new_faces = np.array([[vertex_map[v] for v in face] for face in selected_faces])
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    return mesh


def generate_all_features(mesh, angle_step=ANGLE_STEP):
    """全角度の特徴量を生成"""
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces

    # 角度範囲
    rx_range = list(range(-180, 180, angle_step))
    ry_range = list(range(-180, 180, angle_step))
    rz_range = list(range(-180, 180, angle_step))

    total = len(rx_range) * len(ry_range) * len(rz_range)
    print(f"  角度刻み: {angle_step}度")
    print(f"  総パターン数: {total}")

    features_list = []
    count = 0
    start = time.time()

    for rx in rx_range:
        for ry in ry_range:
            for rz in rz_range:
                count += 1

                # レンダリング
                mask = render_silhouette(
                    vertices, faces, rx, ry, rz,
                    RENDER_SIZE[0], RENDER_SIZE[1]
                )

                # 特徴抽出
                features = extract_features(mask)

                if features is not None:
                    features_list.append({
                        "angles": {"rx": rx, "ry": ry, "rz": rz},
                        "features": features
                    })

                # 進捗表示
                if count % 1000 == 0:
                    elapsed = time.time() - start
                    eta = elapsed / count * (total - count)
                    print(f"    進捗: {count}/{total} ({count/total*100:.1f}%) - 残り{eta:.0f}秒")

    print(f"  完了: {len(features_list)}パターン生成 ({time.time() - start:.1f}秒)")

    return features_list


def main():
    BASE_PATH = Path(__file__).resolve().parent
    MODEL_PATH = BASE_PATH / "models_rabit_dae" / "rabit.dae"
    OUTPUT_PATH = BASE_PATH / "angle_features.json"

    print("=" * 60)
    print("  角度特徴量の事前計算")
    print("=" * 60)

    # モデル読み込み
    print("\n[1/2] 3Dモデル読み込み...")
    mesh = load_3d_model(MODEL_PATH)
    print(f"  頂点数: {len(mesh.vertices):,}")
    print(f"  面数: {len(mesh.faces):,}")

    # 特徴量生成
    print("\n[2/2] 特徴量生成...")
    features_list = generate_all_features(mesh)

    # JSON保存
    output_data = {
        "metadata": {
            "angle_step": ANGLE_STEP,
            "render_size": list(RENDER_SIZE),
            "total_patterns": len(features_list)
        },
        "patterns": features_list
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    file_size = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n保存完了: {OUTPUT_PATH}")
    print(f"ファイルサイズ: {file_size:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
