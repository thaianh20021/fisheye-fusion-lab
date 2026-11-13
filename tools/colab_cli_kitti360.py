import io
import os
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.optimizers import SelectiveAdam

BASE = "https://huggingface.co/datasets/danny042/KITTI-360-fisheye/resolve/main"
SEQ = "2013_05_28_drive_0003_sync"
ARCHIVE = Path(os.environ.get(
    "KITTI360_ARCHIVE", "/content/drive/MyDrive/datasets/kitti360_full"
))
ROOT = Path("/content/kitti360_cli")
OUT = Path("/content/kitti360_cli_out")
ROOT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
N_FRAMES = int(os.environ.get("KITTI360_N_FRAMES", "30"))
SCALE = 2
STEPS = int(os.environ.get("KITTI360_STEPS", "10000"))


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120)


def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return
    with fetch(url) as src, path.open("wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)


def stream_extract(url, suffix, wanted):
    out = {}
    rel = url.split("/resolve/main/", 1)[-1]
    local_tar = ARCHIVE / rel
    source = local_tar.open("rb") if local_tar.exists() else fetch(url)
    mode = "r:*" if local_tar.exists() else "r|*"
    with source, tarfile.open(fileobj=source, mode=mode) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(suffix):
                continue
            stem = Path(member.name).stem
            if stem not in wanted:
                continue
            out[stem] = tar.extractfile(member).read()
            if len(out) == len(wanted):
                break
    return out


def mei_to_fisheye(xi, k1, k2):
    theta = np.linspace(0.02, 1.45, 256)
    u = np.sin(theta) / (np.cos(theta) + xi)
    u2 = u * u
    r = (1 + xi) * u * (1 + k1 * u2 + k2 * u2 * u2)
    X = np.column_stack([theta**2, theta**4, theta**6, theta**8])
    k, *_ = np.linalg.lstsq(X, r / theta - 1, rcond=None)
    return k.astype(np.float32)

def read_official_poses():
    path = ARCHIVE / "data_poses.zip"
    with ZipFile(path) as archive, archive.open(f"{SEQ}/cam0_to_world.txt") as stream:
        poses = {}
        for line in stream:
            values = np.fromstring(line.decode(), sep=" ")
            if len(values) != 17:
                continue
            poses[int(values[0])] = values[1:].reshape(4, 4).astype(np.float32)
    return poses


def rigid(src, dst):
    tree = cKDTree(dst)
    cur = src.astype(np.float64).copy()
    total = np.eye(4)
    for threshold in (8.0, 3.0, 1.0, 0.4):
        for _ in range(15):
            d, idx = tree.query(cur)
            mask = d < threshold
            if mask.sum() < 100:
                break
            p, q = cur[mask], dst[idx[mask]].astype(np.float64)
            cp, cq = p.mean(0), q.mean(0)
            u, _, vt = np.linalg.svd((p - cp).T @ (q - cq))
            r = vt.T @ u.T
            if np.linalg.det(r) < 0:
                vt[-1] *= -1
                r = vt.T @ u.T
            t = cq - r @ cp
            cur = (r @ cur.T).T + t
            step = np.eye(4)
            step[:3, :3], step[:3, 3] = r, t
            total = step @ total
    return total.astype(np.float32)


def save_ply(path, means, quats, scales, opacity, colors):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1",
             "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(len(means), dtype=[(n, "<f4") for n in names])
    values = np.column_stack([
        means, np.zeros_like(means), (colors - 0.5) / 0.28209479,
        opacity, scales, quats,
    ])
    for i, name in enumerate(names):
        data[name] = values[:, i]
    with path.open("wb") as stream:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
        header += "".join(f"property float {n}\n" for n in names)
        stream.write((header + "end_header\n").encode("ascii"))
        stream.write(data.tobytes())


def main():
    meta_path = ARCHIVE / "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl"
    if not meta_path.exists():
        download(f"{BASE}/ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl", ROOT / "rosa.pkl")
        meta_path = ROOT / "rosa.pkl"
    with meta_path.open("rb") as stream:
        all_items = pickle.load(stream)["data_list"]
    official = read_official_poses()
    items = sorted(
        [x for x in all_items if SEQ in x["lidar_path"] and x["frame_id"] in official],
        key=lambda x: x["frame_id"],
    )[:N_FRAMES]
    frame_ids = [x["frame_id"] for x in items]
    wanted = {f"{x:010d}" for x in frame_ids}
    print("[DATA] frames:", frame_ids)

    images = {}
    for cam in ("image_00", "image_01", "image_02", "image_03"):
        url = f"{BASE}/data_2d_raw/{SEQ}_{cam}.tar"
        images[cam] = stream_extract(url, ".png", wanted)
        print("[DATA]", cam, len(images[cam]), "images")
    lidar = stream_extract(
        f"{BASE}/data_3d_raw/{SEQ}.tar", ".bin", wanted
    )
    print("[DATA] lidar", len(lidar), "scans")

    rng = np.random.default_rng(42)
    scans = {}
    for fid in frame_ids:
        raw = np.frombuffer(lidar[f"{fid:010d}"], dtype=np.float32).reshape(-1, 4)[:, :3]
        raw = raw[np.linalg.norm(raw, axis=1).clip(0, 100) > 2.5]
        scans[fid] = raw

    item_by_frame = {x["frame_id"]: x for x in items}
    front_lidar2cam = np.array(
        item_by_frame[frame_ids[0]]["images"]["CAM_FRONT"]["lidar2cam"],
        dtype=np.float32,
    )
    poses = {
        # `cam0_to_world` is world_from_cam0; metadata lidar2cam is cam0_from_lidar.
        fid: official[fid] @ front_lidar2cam
        for fid in frame_ids
    }
    views = []
    cameras = [
        ("CAM_FRONT", "image_00", "pinhole"),
        ("CAM_FRONT_RIGHT", "image_01", "pinhole"),
        ("CAM_BACK_RIGHT", "image_02", "fisheye"),
        ("CAM_BACK_LEFT", "image_03", "fisheye"),
    ]
    for fid in frame_ids:
        item = item_by_frame[fid]
        for key, cam, model in cameras:
            info = item["images"][key]
            raw = images[cam][f"{fid:010d}"]
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            h0, w0 = bgr.shape[:2]
            h, w = h0 // SCALE, w0 // SCALE
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            k = np.array(info["camera_intrinsic"], dtype=np.float32)
            if model == "fisheye":
                k1, k2, _, _, xi = info["distortion_coefficients"]
                k[0, 0] /= 1 + xi
                k[1, 1] /= 1 + xi
                radial = mei_to_fisheye(xi, k1, k2)
            else:
                radial = np.zeros(4, dtype=np.float32)
            k[0] /= SCALE
            k[1] /= SCALE
            view = np.array(info["lidar2cam"], dtype=np.float32) @ np.linalg.inv(poses[fid])
            views.append({
                "fid": fid, "cam": cam, "model": model, "h": h, "w": w,
                "image": torch.tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), device="cuda", dtype=torch.float32) / 255,
                "view": torch.tensor(view, dtype=torch.float32, device="cuda")[None],
                "K": torch.tensor(k, dtype=torch.float32, device="cuda")[None],
                "radial": torch.tensor(radial, dtype=torch.float32, device="cuda")[None],
            })

    # Put every scan in the first-frame world before merging Gaussians.
    points = np.concatenate([
        (poses[fid][:3, :3] @ scans[fid][
            rng.choice(len(scans[fid]), min(3000, len(scans[fid])), replace=False)
        ].T).T + poses[fid][:3, 3]
        for fid in frame_ids
    ])
    points = points[rng.choice(len(points), min(15000, len(points)), replace=False)].astype(np.float32)
    d = cKDTree(points).query(points, k=4)[0][:, 1:].mean(1).clip(0.02, 0.8).astype(np.float32)
    n = len(points)
    p = {
        "means": torch.nn.Parameter(torch.tensor(points, device="cuda")),
        "quats": torch.nn.Parameter(
            torch.tensor(
                np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
                dtype=torch.float32,
                device="cuda",
            )
        ),
        "scales": torch.nn.Parameter(torch.tensor(np.log(d[:, None].repeat(3, 1)), device="cuda")),
        "opacities": torch.nn.Parameter(torch.zeros(n, device="cuda")),
        "colors": torch.nn.Parameter(torch.zeros((n, 3), device="cuda")),
    }
    opt = SelectiveAdam(
        [{"params": [v], "lr": lr, "name": k} for (k, v), lr in zip(
            p.items(), (2.5e-3, 1.5e-3, 6e-3, 4e-2, 3e-2)
        )],
        eps=1e-15, betas=(0.9, 0.999),
    )

    def render(v):
        image, _, info = rasterization(
            p["means"], p["quats"], p["scales"].exp(), p["opacities"].sigmoid(),
            p["colors"].sigmoid(), v["view"], v["K"], v["w"], v["h"],
            packed=False, camera_model=v["model"],
            radial_coeffs=v["radial"] if v["model"] == "fisheye" else None,
            with_ut=True, with_eval3d=True,
        )
        return image[0], info

    train = [v for v in views if v["fid"] != frame_ids[-1]]
    test = [v for v in views if v["fid"] == frame_ids[-1]]
    target = next(v for v in test if v["cam"] == "image_00")
    before, _ = render(target)
    t0 = time.perf_counter()
    steps = STEPS
    for step in range(steps):
        v = train[step % len(train)]
        opt.zero_grad(set_to_none=True)
        pred, info = render(v)
        loss = (pred - v["image"]).abs().mean()
        loss.backward()
        radii = info["radii"]
        visible = (radii[0] > 0).all(-1) if radii.ndim == 3 else radii[0] > 0
        opt.step(visible)
        with torch.no_grad():
            p["scales"].clamp_(np.log(0.005), np.log(0.30))
        if step == 0 or (step + 1) % 500 == 0:
            print(f"step {step + 1}/{steps} loss={loss.item():.4f} active={int(visible.sum())}/{n}")
    after, _ = render(target)
    mse = (after - target["image"]).square().mean().clamp_min(1e-10)
    psnr = float((-10 * torch.log10(mse)).detach())
    comp = torch.cat(
        [target["image"].detach(), before.detach(), after.detach()], dim=1
    ).clamp(0, 1).cpu().numpy()
    cv2.imwrite(str(OUT / "compare_road.png"), comp[:, :, ::-1] * 255)
    save_ply(
        OUT / "gaussians.ply",
        p["means"].detach().cpu().numpy(),
        torch.nn.functional.normalize(p["quats"].detach(), dim=-1).cpu().numpy(),
        p["scales"].detach().cpu().numpy(),
        p["opacities"].detach().cpu().numpy(),
        p["colors"].detach().sigmoid().cpu().numpy(),
    )
    print(f"[DONE] {time.perf_counter() - t0:.1f}s | held-out front-road PSNR={psnr:.2f} dB")
    print("[OUT]", OUT / "compare_road.png")
    print("[OUT]", OUT / "gaussians.ply")


if __name__ == "__main__":
    main()
