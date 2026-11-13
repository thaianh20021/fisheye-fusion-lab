"""Render a fly-through video from a trained Gaussian PLY (gsplat 3DGUT, front pinhole camera).

Camera path = the CAM_FRONT poses of the training frames, smoothed with a cubic spline
(position) and slerp (rotation). Intrinsics are the original full-resolution CAM_FRONT ones,
so every pixel is inside the field of view the model was trained on.

Needs colab_cli_kitti360.py next to it, or FLY_TRAINER (reuses its pose/metadata loaders).
Env: FLY_PLY, FLY_OUT, FLY_FRAMES (100), FLY_SUB (10 interpolated frames per step), FLY_FPS (30),
     KITTI360_ARCHIVE (Drive archive holding data_poses.zip and the metadata pickles).
"""
import importlib.util
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

PLY = os.environ.get("FLY_PLY", "/content/run100_cap1m/gaussians.ply")
OUT = Path(os.environ.get("FLY_OUT", "/content/fly/fly_best.mp4"))
N_FRAMES = int(os.environ.get("FLY_FRAMES", "100"))
SUB = int(os.environ.get("FLY_SUB", "10"))
FPS = int(os.environ.get("FLY_FPS", "30"))
os.environ.setdefault("KITTI360_ARCHIVE", "/content/drive/MyDrive/datasets/kitti360_full")

TRAINER = os.environ.get("FLY_TRAINER", str(Path(__file__).resolve().with_name("colab_cli_kitti360.py")))
spec = importlib.util.spec_from_file_location("trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def load_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = int(next(h for h in header if h.startswith("element vertex")).split()[2])
        names = [h.split()[2] for h in header if h.startswith("property")]
        data = np.frombuffer(f.read(), dtype="<f4").reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    pick = lambda keys: data[:, [col[k] for k in keys]].copy()
    return dict(
        means=pick(["x", "y", "z"]), quats=pick([f"rot_{i}" for i in range(4)]),
        scales=pick([f"scale_{i}" for i in range(3)]), opacity=data[:, col["opacity"]].copy(),
        colors=np.clip(pick([f"f_dc_{i}" for i in range(3)]) * 0.28209479 + 0.5, 0, 1))


def front_c2w_path():
    """World-from-camera poses (same re-centred world as the trainer) and the front intrinsics."""
    items = []
    for split in ("val", "train"):
        p = trainer.ensure_file(f"ImageSets-rosa-6class/kitti360_rosa_infos_{split}.pkl")
        items += pickle.load(p.open("rb"))["data_list"]
    official = trainer.read_official_poses()
    items = sorted([x for x in items if trainer.SEQ in x["lidar_path"] and x["frame_id"] in official],
                   key=lambda x: x["frame_id"])[:N_FRAMES]
    front0 = np.array(items[0]["images"]["CAM_FRONT"]["lidar2cam"], dtype=np.float64)
    origin = official[items[0]["frame_id"]][:3, 3].copy()
    c2w = []
    for it in items:
        wfl = official[it["frame_id"]] @ front0
        wfl[:3, 3] -= origin
        view = np.array(it["images"]["CAM_FRONT"]["lidar2cam"], dtype=np.float64) @ np.linalg.inv(wfl)
        c2w.append(np.linalg.inv(view))
    K = np.array(items[0]["images"]["CAM_FRONT"]["camera_intrinsic"], dtype=np.float64)
    return np.array(c2w), K, [it["frame_id"] for it in items]


def smooth(c2w, sub):
    t = np.arange(len(c2w))
    tq = np.linspace(0, len(c2w) - 1, (len(c2w) - 1) * sub + 1)
    pos = CubicSpline(t, c2w[:, :3, 3])(tq)
    rot = Slerp(t, Rotation.from_matrix(c2w[:, :3, :3]))(tq).as_matrix()
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3], out[:, :3, 3] = rot, pos
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g = load_ply(PLY)
    print(f"[FLY] {len(g['means'])} Gaussians from {PLY}", flush=True)
    c2w, K, frame_ids = front_c2w_path()
    path = smooth(c2w, SUB)
    W, H = 1408, 376
    print(f"[FLY] {len(frame_ids)} keyframes {frame_ids[0]}..{frame_ids[-1]} -> {len(path)} video frames "
          f"at {W}x{H}, {len(path) / FPS:.0f}s", flush=True)

    dev = "cuda"
    means = torch.tensor(g["means"], device=dev)
    quats = torch.nn.functional.normalize(torch.tensor(g["quats"], device=dev), dim=-1)
    scales = torch.tensor(g["scales"], device=dev).exp()
    opacity = torch.tensor(g["opacity"], device=dev).sigmoid()
    colors = torch.tensor(g["colors"], device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]

    raw = OUT.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, pose in enumerate(path):
            view = torch.tensor(np.linalg.inv(pose), dtype=torch.float32, device=dev)[None]
            img, _, _ = rasterization(means, quats, scales, opacity, colors, view, Kt, W, H, packed=False,
                                      camera_model="pinhole", with_ut=True, with_eval3d=True)
            frame = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % 100 == 0:
                print(f"[FLY] frame {i}/{len(path)} {(i + 1) / (time.perf_counter() - t0):.1f} fps", flush=True)
    writer.release()

    if shutil.which("ffmpeg"):   # re-encode to H.264 so the file plays in browsers and phones
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], check=True)
        raw.unlink()
    else:
        raw.replace(OUT)
    print(f"[DONE] {OUT} {OUT.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
