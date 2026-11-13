"""Deterministic angular target; not a reconstructed scene."""
from pathlib import Path
import cv2
import numpy as np
from .geometry import pixel_to_ray
from .models import ProjectManifest


def make_demo(root: Path, model="opencv_fisheye"):
    root.mkdir(parents=True, exist_ok=True)
    cameras = []
    width, height = 320, 240
    focal = 100 if model == "opencv_fisheye" else 160
    K = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    D = [0., 0., 0., 0.]
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    rays = pixel_to_ray(np.column_stack([xx.ravel(), yy.ravel()]), K, D, model)
    for name, degrees in [("front", 0), ("right", 90), ("rear", 180), ("left", -90)]:
        angle = np.deg2rad(degrees)
        c, s = np.cos(angle), np.sin(angle)
        T = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        world_rays = rays @ T[:3, :3].T
        lon = np.arctan2(world_rays[:, 0], world_rays[:, 2])
        lat = np.arcsin(world_rays[:, 1])
        sector = np.mod(np.floor((lon + np.pi) / (np.pi / 2)).astype(int), 4)
        palette = np.array([[45, 132, 153], [206, 103, 90], [57, 145, 110], [194, 160, 60]])
        rgb = palette[sector].astype(float) + (np.cos(lat) * 30)[:, None]
        grid = (np.abs(np.sin(lon * 12)) < .04) | (np.abs(np.sin(lat * 12)) < .04)
        rgb[grid] = [236, 242, 241]
        rgb[np.abs(lat) < .015] = [30, 36, 40]
        image = np.clip(rgb, 0, 255).astype(np.uint8).reshape(height, width, 3)
        if not cv2.imwrite(str(root / f"{name}.png"), image[:, :, ::-1]):
            raise OSError("Could not write fixture image")
        cameras.append(dict(id=name, model=model, resolution=[width, height], K=K, D=D, T_rig_from_camera=T.tolist()))
    manifest = ProjectManifest.model_validate(dict(
        project="Synthetic angular target", cameras=cameras,
        frames=[dict(timestamp_ns=0, images={c["id"]: c["id"] + ".png" for c in cameras}, T_world_from_rig=np.eye(4).tolist())],
        fusion=dict(camera_ids=[c["id"] for c in cameras], resolution=[1024, 512]),
    ))
    path = root / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(make_demo(Path("examples/synthetic_rig")))
