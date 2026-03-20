# render_textured.py
# テクスチャ付き3Dモデルをレンダリング
#
# WSL2で実行:
# conda activate pytorch3d
# cd /mnt/c/nagano/3Dnagano
# python render_textured.py

import math
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    PointLights,
)
from pytorch3d.io import load_obj

BASE_DIR = Path("/mnt/c/nagano/3Dnagano")
MODEL_PATH = BASE_DIR / "models_rabit_obj" / "rabit.obj"  # フルメッシュ（UV座標あり）
TEXTURE_PATH = BASE_DIR / "models_rabit_obj" / "rabit01.jpg"
OUT_DIR = BASE_DIR / "rotation_results_test53"

RENDER_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")

    # OBJファイルを読み込み
    verts, faces, aux = load_obj(str(MODEL_PATH))

    # テクスチャ画像を読み込み
    print(f"Loading texture: {TEXTURE_PATH}")
    texture_image = Image.open(str(TEXTURE_PATH))
    texture_image = np.array(texture_image) / 255.0
    texture_image = torch.from_numpy(texture_image).float().to(device)
    texture_image = texture_image.unsqueeze(0)  # [1, H, W, 3]

    # UV座標とテクスチャを設定
    verts_uvs = aux.verts_uvs.to(device)  # [V, 2]
    faces_uvs = faces.textures_idx.to(device)  # [F, 3]

    # TexturesUV作成
    textures = TexturesUV(
        maps=texture_image,
        faces_uvs=[faces_uvs],
        verts_uvs=[verts_uvs]
    )

    # メッシュ作成
    verts = verts.to(device)
    faces_verts = faces.verts_idx.to(device)

    mesh = Meshes(
        verts=[verts],
        faces=[faces_verts],
        textures=textures
    )

    # 頂点を正規化
    verts = mesh.verts_packed()
    center = verts.mean(0)
    verts = verts - center
    scale = verts.abs().max()
    verts = verts / (scale * 2)
    mesh = mesh.update_padded(verts.unsqueeze(0))

    # レンダラー設定
    raster_settings = RasterizationSettings(
        image_size=RENDER_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    # ライティング（明るく）
    lights = PointLights(
        device=device,
        location=[[0.0, 3.0, 3.0]],
        ambient_color=[[0.7, 0.7, 0.7]],
        diffuse_color=[[0.5, 0.5, 0.5]],
    )

    # 広い範囲で角度を探索
    angles = [
        # 上から見下ろす角度
        (0, 0, 0, 2.5, "front"),
        (0, 90, 0, 2.5, "right"),
        (0, 180, 0, 2.5, "back"),
        (0, 270, 0, 2.5, "left"),
        # 斜め上から
        (30, 45, 0, 2.5, "diag1"),
        (30, 135, 0, 2.5, "diag2"),
        (30, 225, 0, 2.5, "diag3"),
        (30, 315, 0, 2.5, "diag4"),
        # test52で見つけた角度付近
        (-50, 260, -5, 2.0, "best1"),
        (-30, 255, 0, 2.0, "best2"),
        (20, 100, 60, 2.5, "test1"),
        (10, 95, 60, 2.5, "test2"),
    ]

    for elev, azim, roll, dist, name in angles:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

        # roll適用
        roll_rad = roll * np.pi / 180.0
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        roll_matrix = torch.tensor([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        R = R.to(device)
        R = torch.bmm(roll_matrix, R)

        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )

        images = renderer(mesh)
        img = images[0, ..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(OUT_DIR / f"textured_{name}.png"), img)
        print(f"Saved: textured_{name}.png (elev={elev}, azim={azim}, roll={roll}, dist={dist})")

    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
