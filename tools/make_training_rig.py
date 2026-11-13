"""Generate a tiny SYNTHETIC calibrated 4-fisheye rig for CUDA integration tests."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization


def main():
    out = Path("examples/synthetic_training_rig")
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(12)
    points = rng.normal(size=(512, 3))
    points = (points / np.linalg.norm(points, axis=1, keepdims=True) *
              rng.uniform(3, 6, size=(512, 1))).astype("float32")
    colors = rng.uniform(.1, .95, size=points.shape).astype("float32")
    np.savez_compressed(out / "points.npz", points=points, colors=colors)
    gpu = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cuda")
    K = np.array([[44., 0, 48], [0, 44., 32], [0, 0, 1]])
    D = [.01, -.001, 0., 0.]
    cameras, frames = [], []
    for index, name in enumerate(("front", "right", "rear", "left")):
        angle = index * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        transform = np.eye(4)
        transform[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        transform[:3, 3] = [.2 * s, 0, .2 * c]
        cameras.append({"id": name, "model": "opencv_fisheye", "resolution": [96, 64],
                        "K": K.tolist(), "D": D, "T_rig_from_camera": transform.tolist()})
    for frame_index in range(3):
        pose = np.eye(4)
        pose[0, 3] = frame_index * .15
        images = {}
        for camera in cameras:
            view = np.linalg.inv(pose @ np.array(camera["T_rig_from_camera"]))
            with torch.no_grad():
                image = rasterization(
                    gpu(points), gpu(np.tile([1., 0, 0, 0], (512, 1))),
                    gpu(np.full((512, 3), .16)), gpu(np.full(512, .9)), gpu(colors),
                    gpu(view)[None], gpu(K)[None], 96, 64, packed=False,
                    camera_model="fisheye", radial_coeffs=gpu([D]),
                    with_geer=True, with_eval3d=True,
                )[0][0]
            name = f"{frame_index}-{camera['id']}.png"
            pixels = (image.cpu().numpy().clip(0, 1) * 255).round().astype("uint8")
            if not cv2.imwrite(str(out / name), pixels[:, :, ::-1]):
                raise OSError("Image write failed")
            images[camera["id"]] = name
        frames.append({"timestamp_ns": frame_index * 100000000,
                       "images": images, "T_world_from_rig": pose.tolist()})
    manifest = {"schema_version": 1, "project": "SYNTHETIC GEER rig",
                "root": ".", "cameras": cameras, "frames": frames,
                "fusion": {"camera_ids": [c["id"] for c in cameras]},
                "training_points": "points.npz"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SYNTHETIC fixture written: {out}")


if __name__ == "__main__":
    main()
