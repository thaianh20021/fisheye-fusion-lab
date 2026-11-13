"""KITTI-360 4-camera 3DGUT + Fast3DGS-style trainer (gsplat).

- 2 pinhole (image_00/01) + 2 fisheye (image_02/03), real Velodyne LiDAR, official poses.
- Gaussians initialised from LiDAR, colours sampled from the camera that saw the scan.
- gsplat rasterization with 3DGUT (with_ut + with_eval3d) so fisheye is rendered exactly.
- Fast3DGS-style SelectiveAdam: only Gaussians visible in the current view are updated.
- gsplat MCMCStrategy: relocation of dead Gaussians and growth to a cap (no 2D gradients needed).

Env: KITTI360_N_FRAMES (100), KITTI360_STEPS (20000), KITTI360_MAX_GAUSSIANS (120000), KITTI360_CAP_MAX (1000000), KITTI360_SYNC_DIR, KITTI360_RUN_NAME, KITTI360_RESUME,
     KITTI360_TEST_EVERY (10), KITTI360_ARCHIVE, KITTI360_OUT (/content/kitti360_cli_out).
"""
import json
import os
import pickle
import random
import shutil
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from scipy.signal import medfilt
from scipy.spatial import cKDTree
from gsplat import rasterization
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import MCMCStrategy

BASE = "https://huggingface.co/datasets/danny042/KITTI-360-fisheye/resolve/main"
POSES_URL = ("https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
             "89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip")
SEQ = "2013_05_28_drive_0003_sync"
ARCHIVE = Path(os.environ.get("KITTI360_ARCHIVE", "/content/kitti360_min"))
OUT = Path(os.environ.get("KITTI360_OUT", "/content/kitti360_cli_out"))
ROOT = OUT / "work"   # scratch space inside the output directory: always writable, not Colab-specific
N_FRAMES = int(os.environ.get("KITTI360_N_FRAMES", "100"))
STEPS = int(os.environ.get("KITTI360_STEPS", "20000"))
MAX_GAUSSIANS = int(os.environ.get("KITTI360_MAX_GAUSSIANS", "120000"))
CAP_MAX = int(os.environ.get("KITTI360_CAP_MAX", "1000000"))
SEED = int(os.environ.get("KITTI360_SEED", "42"))
RUN_NAME = os.environ.get("KITTI360_RUN_NAME", time.strftime("run_%Y%m%d_%H%M%S"))
SYNC_DIR = os.environ.get("KITTI360_SYNC_DIR", "")      # e.g. a Drive folder; empty disables syncing
RESUME = os.environ.get("KITTI360_RESUME", "0") == "1"   # continue RUN_NAME from its last checkpoint
CKPT_EVERY = int(os.environ.get("KITTI360_CKPT_EVERY", "2500"))
USE_MASK = os.environ.get("KITTI360_MASK", "1") == "1"    # ignore the car hood / outside the fisheye circle
USE_SSIM = os.environ.get("KITTI360_SSIM", "0") == "1"    # loss = 0.8 L1 + 0.2 (1 - SSIM)
SSIM_IMPL = os.environ.get("KITTI360_SSIM_IMPL", "auto")  # auto | fused | torch (fused = fused-ssim CUDA kernel)
USE_SKY = os.environ.get("KITTI360_SKY", "0") == "1"      # seed large Gaussians on a far sky dome
MASK_ONLY = os.environ.get("KITTI360_MASK_ONLY", "0") == "1"
PERSIST = float(os.environ.get("KITTI360_PERSIST", "0.5"))   # edge persistence above which a lower-image edge is static
SKY_RADIUS, SKY_SCALE, SKY_PER_FRAME = 200.0, 12.0, 400
MAX_SCALE = 40.0 if USE_SKY else 0.5
LOG_PATH = os.environ.get("KITTI360_LOG", "/content/train.log")
TEST_EVERY = int(os.environ.get("KITTI360_TEST_EVERY", "10"))
SCALE = 2
CAMERAS = [
    ("CAM_FRONT", "image_00", "pinhole"),
    ("CAM_FRONT_RIGHT", "image_01", "pinhole"),
    ("CAM_BACK_RIGHT", "image_02", "fisheye"),
    ("CAM_BACK_LEFT", "image_03", "fisheye"),
]
DEV = "cuda"


# ------------------------------------------------------------------ data access
def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=120)


def ensure_file(rel, url=None, min_size=1):
    dst = ARCHIVE / rel
    if dst.exists() and dst.stat().st_size >= min_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fetch(url or f"{BASE}/{rel}") as src, dst.open("wb") as out:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            out.write(block)
    return dst


def stream_extract(rel, suffix, wanted, attempts=3):
    """Read wanted members from a local tar or stream it from Hugging Face (with retries)."""
    local = ARCHIVE / rel
    for attempt in range(attempts):
        out = {}
        try:
            source = local.open("rb") if local.exists() else fetch(f"{BASE}/{rel}")
            mode = "r:*" if local.exists() else "r|*"
            with source, tarfile.open(fileobj=source, mode=mode) as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(suffix):
                        continue
                    stem = Path(member.name).stem
                    if stem in wanted:
                        out[stem] = tar.extractfile(member).read()
                        if len(out) == len(wanted):
                            break
            return out
        except Exception as exc:  # network hiccups while streaming multi-GB tars
            print(f"[RETRY] {rel} attempt {attempt + 1}: {exc}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"could not read {rel}")


def read_official_poses():
    path = ensure_file("data_poses.zip", POSES_URL, min_size=1_000_000)
    with ZipFile(path) as archive, archive.open(f"{SEQ}/cam0_to_world.txt") as stream:
        poses = {}
        for line in stream:
            values = np.array(line.split(), dtype=np.float64)
            if len(values) == 17:
                poses[int(values[0])] = values[1:].reshape(4, 4)
    return poses


# ------------------------------------------------------------------ camera models
def mei_to_fisheye(xi, k1, k2):
    """Fit gsplat's OpenCV-fisheye radial polynomial to the exact Mei model (<0.002 px error)."""
    theta = np.linspace(0.02, 1.45, 256)
    u = np.sin(theta) / (np.cos(theta) + xi)
    u2 = u * u
    r = (1 + xi) * u * (1 + k1 * u2 + k2 * u2 * u2)
    X = np.column_stack([theta**2, theta**4, theta**6, theta**8])
    k, *_ = np.linalg.lstsq(X, r / theta - 1, rcond=None)
    return k.astype(np.float32)


def project_for_color(view, pts_world):
    """Pixel coordinates of world points in a view (same camera model the renderer uses)."""
    pc = pts_world @ view["T"][:3, :3].T + view["T"][:3, 3]
    K = view["K_np"]
    if view["model"] == "pinhole":
        ok = pc[:, 2] > 0.5
        z = np.where(ok, pc[:, 2], 1.0)
        uv = np.column_stack([K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * pc[:, 1] / z + K[1, 2]])
        return uv, ok
    norm_xy = np.linalg.norm(pc[:, :2], axis=1)
    theta = np.arctan2(norm_xy, pc[:, 2])
    ok = (np.linalg.norm(pc, axis=1) > 0.5) & (theta < 1.40)  # renderer limit is pi/2
    t2 = theta * theta
    r = view["radial_np"]
    theta_d = theta * (1 + r[0] * t2 + r[1] * t2**2 + r[2] * t2**3 + r[3] * t2**4)
    scale = np.where(norm_xy > 1e-9, theta_d / np.maximum(norm_xy, 1e-9), 1.0)
    uv = np.column_stack([K[0, 0] * pc[:, 0] * scale + K[0, 2], K[1, 1] * pc[:, 1] * scale + K[1, 2]])
    return uv, ok


def sample_colors(views_of_frame, pts_world):
    """Colour from the first camera of the same frame that sees the point."""
    colors = np.zeros((len(pts_world), 3), np.float32)
    seen = np.zeros(len(pts_world), bool)
    for view in views_of_frame:
        uv, ok = project_for_color(view, pts_world)
        h, w = view["image_np"].shape[:2]
        inside = ok & ~seen & (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 2)
        if not inside.any():
            continue
        px = np.round(uv[inside]).astype(int)
        colors[inside] = view["image_np"][px[:, 1], px[:, 0]].astype(np.float32) / 255.0
        seen |= inside
    return colors, seen


# ------------------------------------------------------------------ masks, SSIM
def static_masks(views):
    """1 = supervise, 0 = ignore. The car hood and the black area outside the fisheye circle are
    fixed in the camera frame. The hood's colours change with the light, but its *edges* stay at the
    same pixels in every frame while scene edges move, so edge persistence finds it."""
    masks = {}
    for cam in ("image_02", "image_03"):
        gray = np.stack([cv2.cvtColor(v["image_np"], cv2.COLOR_RGB2GRAY) for v in views if v["cam"] == cam]
                        ).astype(np.float32)
        sob = lambda g: np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        mean_grad = np.mean([sob(g) for g in gray], axis=0)          # average edge strength per pixel
        persistence = sob(gray.mean(0)) / (mean_grad + 1.0)          # ~1 where edges never move
        h, w = mean_grad.shape
        rows = np.arange(h)[:, None]
        edges = ((persistence > PERSIST) & (mean_grad > 15) & (rows > 0.63 * h)).astype(np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        # Hood = everything below the lowest-reaching persistent edge in each column.
        top = np.array([np.flatnonzero(closed[:, x]).min() if closed[:, x].any() else h for x in range(w)], np.float32)
        top = medfilt(top, 61)                       # single-column noise must not cut into the scene
        hood = (np.arange(h)[:, None] >= top[None, :]).astype(np.uint8)
        outside = ((gray.mean(0) < 12) & (gray.std(0) < 6)).astype(np.uint8)
        bad = cv2.dilate(np.maximum(hood, outside), np.ones((9, 9), np.uint8))
        masks[cam] = (1 - bad).astype(np.float32)
        cv2.imwrite(str(OUT / f"mask_{cam}.png"), masks[cam] * 255)
        cv2.imwrite(str(OUT / f"persist_{cam}.png"), np.clip(persistence * 255, 0, 255).astype(np.uint8))
        sync(OUT / f"mask_{cam}.png")
        print(f"[MASK] {cam}: supervising {masks[cam].mean() * 100:.0f}% of pixels", flush=True)
    return masks


def make_ssim():
    """Pick the SSIM kernel: the fused-ssim CUDA package when installed (accelerator), else conv2d."""
    if SSIM_IMPL in ("auto", "fused"):
        try:
            from fused_ssim import fused_ssim
            print("[ACCEL] SSIM: fused-ssim", flush=True)
            return lambda a, b: fused_ssim(a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None])
        except ImportError:
            if SSIM_IMPL == "fused":
                raise
    print("[ACCEL] SSIM: torch conv2d", flush=True)
    return ssim_fn


def ssim_fn(a, b, win=11, sigma=1.5):
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(win, device=x.device).float() - win // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda t: torch.nn.functional.conv2d(t, k, padding=win // 2, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))).mean()


# ------------------------------------------------------------------ durable output
def sync(path):
    """Copy a finished local file into SYNC_DIR/RUN_NAME atomically (tmp file, size check, rename)."""
    if not SYNC_DIR:
        return
    path = Path(path)
    dst_dir = Path(SYNC_DIR) / RUN_NAME
    dst, tmp = dst_dir / path.name, dst_dir / (path.name + ".tmp")
    for attempt in range(3):
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, tmp)
            if tmp.stat().st_size != path.stat().st_size:
                raise IOError("size mismatch after copy")
            os.replace(tmp, dst)
            return
        except Exception as exc:  # Drive FUSE can hiccup; training must not die because of it
            print(f"[SYNC-RETRY] {path.name}: {exc}", flush=True)
            time.sleep(5)
    print(f"[SYNC-FAIL] {path.name} (local copy kept in {path.parent})", flush=True)


# ------------------------------------------------------------------ export
def save_ply(path, means, quats, scales, opacity, colors):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(len(means), dtype=[(n, "<f4") for n in names])
    values = np.column_stack([means, np.zeros_like(means), (colors - 0.5) / 0.28209479,
                              opacity, scales, quats])
    for i, name in enumerate(names):
        data[name] = values[:, i]
    with path.open("wb") as stream:
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
        header += "".join(f"property float {n}\n" for n in names)
        stream.write((header + "end_header\n").encode("ascii"))
        stream.write(data.tobytes())


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    import gsplat
    config = dict(mask=USE_MASK, ssim=USE_SSIM, sky=USE_SKY, run=RUN_NAME, seed=SEED, frames=N_FRAMES, steps=STEPS, cap_max=CAP_MAX,
                  init_gaussians=MAX_GAUSSIANS, test_every=TEST_EVERY, image_scale=SCALE,
                  torch=torch.__version__, gsplat=gsplat.__version__, resume=RESUME,
                  gpu=torch.cuda.get_device_name(0))
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    sync(OUT / "config.json")
    all_items = []
    for split in ("val", "train"):   # val holds frames 12-282 of this sequence, train the rest
        meta_path = ensure_file(f"ImageSets-rosa-6class/kitti360_rosa_infos_{split}.pkl")
        with meta_path.open("rb") as stream:
            all_items += pickle.load(stream)["data_list"]
    official = read_official_poses()
    items = sorted([x for x in all_items if SEQ in x["lidar_path"] and x["frame_id"] in official],
                   key=lambda x: x["frame_id"])[:N_FRAMES]
    frame_ids = [x["frame_id"] for x in items]
    item_by_frame = {x["frame_id"]: x for x in items}
    wanted = {f"{x:010d}" for x in frame_ids}
    print(f"[DATA] {len(frame_ids)} frames: {frame_ids[0]}..{frame_ids[-1]}", flush=True)

    t0 = time.perf_counter()
    jobs = {cam: (f"data_2d_raw/{SEQ}_{cam}.tar", ".png") for _, cam, _ in CAMERAS}
    jobs["lidar"] = (f"data_3d_raw/{SEQ}.tar", ".bin")
    with ThreadPoolExecutor(len(jobs)) as pool:
        futures = {k: pool.submit(stream_extract, rel, suf, wanted) for k, (rel, suf) in jobs.items()}
        raw = {k: f.result() for k, f in futures.items()}
    print(f"[DATA] downloaded in {time.perf_counter() - t0:.0f}s:",
          {k: len(v) for k, v in raw.items()}, flush=True)

    # World frame = official world, re-centred on the first frame for float32 precision.
    front_lidar2cam = np.array(item_by_frame[frame_ids[0]]["images"]["CAM_FRONT"]["lidar2cam"], dtype=np.float64)
    origin = official[frame_ids[0]][:3, 3].copy()
    world_from_lidar = {}
    for fid in frame_ids:
        m = official[fid] @ front_lidar2cam     # cam0_to_world @ cam0_from_lidar (checked by projection_check)
        m[:3, 3] -= origin
        world_from_lidar[fid] = m

    views = []
    for fid in frame_ids:
        item = item_by_frame[fid]
        for key, cam, model in CAMERAS:
            info = item["images"][key]
            bgr = cv2.imdecode(np.frombuffer(raw[cam][f"{fid:010d}"], np.uint8), cv2.IMREAD_COLOR)
            h, w = bgr.shape[0] // SCALE, bgr.shape[1] // SCALE
            rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            k = np.array(info["camera_intrinsic"], dtype=np.float64)
            radial = np.zeros(4, np.float32)
            if model == "fisheye":
                k1, k2, _, _, xi = info["distortion_coefficients"]
                k[0, 0] /= 1 + xi
                k[1, 1] /= 1 + xi
                radial = mei_to_fisheye(xi, k1, k2)
            k[0] /= SCALE
            k[1] /= SCALE
            lidar2cam = np.array(info["lidar2cam"], dtype=np.float64)
            T = lidar2cam @ np.linalg.inv(world_from_lidar[fid])
            views.append(dict(fid=fid, cam=cam, model=model, h=h, w=w, image_np=rgb, K_np=k,
                              radial_np=radial, T=T))

    masks = static_masks(views)     # always computed: evaluation uses it even when training ignores it
    if MASK_ONLY:
        print("[DONE] mask preview only", flush=True)
        return

    # ---- Gaussian initialisation: LiDAR geometry + colour from the camera that saw it.
    rng = np.random.default_rng(42)
    per_frame = max(500, int(MAX_GAUSSIANS * 1.5 / len(frame_ids)))
    views_by_frame = {}
    for v in views:
        views_by_frame.setdefault(v["fid"], []).append(v)
    pts_all, col_all = [], []
    for fid in frame_ids:
        scan = np.frombuffer(raw["lidar"][f"{fid:010d}"], np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        dist = np.linalg.norm(scan, axis=1)
        scan = scan[(dist > 2.5) & (dist < 80)]
        world = scan @ world_from_lidar[fid][:3, :3].T + world_from_lidar[fid][:3, 3]
        colors, seen = sample_colors(views_by_frame[fid], world)
        world, colors = world[seen], colors[seen]          # keep only points some camera observed
        pick = rng.choice(len(world), min(per_frame, len(world)), replace=False)
        pts_all.append(world[pick])
        col_all.append(colors[pick])
    del raw
    points = np.concatenate(pts_all).astype(np.float32)
    colors0 = np.concatenate(col_all).astype(np.float32)
    if len(points) > MAX_GAUSSIANS:
        pick = rng.choice(len(points), MAX_GAUSSIANS, replace=False)
        points, colors0 = points[pick], colors0[pick]
    n = len(points)
    knn = cKDTree(points).query(points, k=4)[0][:, 1:].mean(1).clip(0.03, 1.0).astype(np.float32)
    if USE_SKY:
        sky_pts, sky_col = [], []
        for fid in frame_ids:
            d = rng.normal(size=(SKY_PER_FRAME, 3))
            d = d / np.linalg.norm(d, axis=1, keepdims=True)
            d[:, 2] = np.abs(d[:, 2])
            d = d[d[:, 2] > np.sin(np.radians(12))]        # world z is up: keep the upper sky
            world = world_from_lidar[fid][:3, 3] + SKY_RADIUS * d
            colors, seen = sample_colors(views_by_frame[fid], world)
            sky_pts.append(world[seen])
            sky_col.append(colors[seen])
        sky_pts, sky_col = np.concatenate(sky_pts).astype(np.float32), np.concatenate(sky_col).astype(np.float32)
        points = np.concatenate([points, sky_pts])
        colors0 = np.concatenate([colors0, sky_col])
        knn = np.concatenate([knn, np.full(len(sky_pts), SKY_SCALE, np.float32)])
        n = len(points)
        print(f"[INIT] +{len(sky_pts)} sky Gaussians on a {SKY_RADIUS:.0f} m dome", flush=True)
    print(f"[INIT] {n} Gaussians from LiDAR, extent {np.ptp(points, axis=0).round(1)} m", flush=True)

    centers = np.array([-v["T"][:3, :3].T @ v["T"][:3, 3] for v in views if v["cam"] == "image_00"])
    # gsplat scales the position lr and the split/prune thresholds with scene_scale. A driving
    # sequence is hundreds of metres long but its Gaussians are centimetres to decimetres, so
    # the trajectory length (~370 m for 100 frames) must not be used directly.
    trajectory = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    scene_scale = float(np.clip(1.1 * trajectory, 5.0, 20.0))
    print(f"[INIT] scene_scale {scene_scale:.1f} m (trajectory radius {trajectory:.0f} m)", flush=True)

    def logit(x):
        return np.log(x / (1 - x))

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(points, device=DEV)),
        "quats": torch.nn.Parameter(torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1)), dtype=torch.float32, device=DEV)),
        "scales": torch.nn.Parameter(torch.tensor(np.log(knn[:, None].repeat(3, 1)), device=DEV)),
        "opacities": torch.nn.Parameter(torch.full((n,), float(logit(0.3)), device=DEV)),
        "colors": torch.nn.Parameter(torch.tensor(logit(colors0.clip(0.02, 0.98)), device=DEV)),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2, "colors": 2.5e-2}
    optimizers = {k: SelectiveAdam([{"params": [params[k]], "lr": lr, "name": k}], eps=1e-15, betas=(0.9, 0.999))
                  for k, lr in lrs.items()}
    start_step = 0
    ckpt_dir = (Path(SYNC_DIR) / RUN_NAME) if SYNC_DIR else OUT
    if RESUME and (ckpt_dir / "ckpt.pt").exists():
        ck = torch.load(ckpt_dir / "ckpt.pt", map_location=DEV)
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(DEV)) for k, v in ck["params"].items()})
        optimizers = {k: SelectiveAdam([{"params": [params[k]], "lr": lr, "name": k}], eps=1e-15,
                                       betas=(0.9, 0.999)) for k, lr in lrs.items()}
        for k, opt in optimizers.items():
            opt.load_state_dict(ck["opt"][k])
        start_step = int(ck["step"])
        print(f"[RESUME] step {start_step}, {len(params['means'])} Gaussians from {ckpt_dir / 'ckpt.pt'}", flush=True)
        del ck
    # 3DGUT (with_eval3d) never touches means2d, so gradient-based densification (DefaultStrategy)
    # has nothing to read. MCMC relocates dead Gaussians and grows towards cap_max instead.
    strategy = MCMCStrategy(cap_max=CAP_MAX, refine_start_iter=500, refine_stop_iter=int(STEPS * 0.8),
                            refine_every=100, min_opacity=0.005, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    # ---- Move images to the GPU as uint8 (400 views of 100 frames fit on a T4).
    for v in views:
        v["image"] = torch.from_numpy(v["image_np"]).to(DEV)
        v["view"] = torch.tensor(v["T"], dtype=torch.float32, device=DEV)[None]
        v["K"] = torch.tensor(v["K_np"], dtype=torch.float32, device=DEV)[None]
        v["radial"] = torch.tensor(v["radial_np"], dtype=torch.float32, device=DEV)[None]
        m = masks.get(v["cam"], np.ones((v["h"], v["w"]), np.float32))
        v["eval_mask"] = torch.from_numpy(m).to(DEV)[..., None]
        v["mask"] = v["eval_mask"] if USE_MASK else torch.ones_like(v["eval_mask"])   # loss mask
    test_fids = set(frame_ids[TEST_EVERY - 1::TEST_EVERY]) if TEST_EVERY else set()
    train = [v for v in views if v["fid"] not in test_fids]
    test = [v for v in views if v["fid"] in test_fids]
    print(f"[SPLIT] train views {len(train)}, held-out views {len(test)}", flush=True)

    def render(v):
        image, _, info = rasterization(
            params["means"], params["quats"], params["scales"].exp(), params["opacities"].sigmoid(),
            params["colors"].sigmoid(), v["view"], v["K"], v["w"], v["h"], packed=False,
            camera_model=v["model"], radial_coeffs=v["radial"] if v["model"] == "fisheye" else None,
            with_ut=True, with_eval3d=True)
        return image[0], info

    def gt(v):
        return v["image"].float() / 255

    def evaluate():
        stats = {}
        with torch.no_grad():
            for v in test:
                pred, _ = render(v)
                err = (pred.clamp(0, 1) - gt(v)).square()
                mse = err.mean().clamp_min(1e-10)
                mse_m = ((err * v["eval_mask"]).sum() / (v["eval_mask"].sum() * 3).clamp_min(1)).clamp_min(1e-10)
                stats.setdefault(v["cam"], []).append((float(-10 * torch.log10(mse)), float(-10 * torch.log10(mse_m))))
        return {cam: {"full": float(np.mean([a for a, _ in x])), "masked": float(np.mean([b for _, b in x]))}
                for cam, x in stats.items()}

    def export(tag):
        with torch.no_grad():
            save_ply(OUT / f"gaussians{tag}.ply", params["means"].cpu().numpy(),
                     torch.nn.functional.normalize(params["quats"], dim=-1).cpu().numpy(),
                     params["scales"].cpu().numpy(), params["opacities"].cpu().numpy(),
                     params["colors"].sigmoid().cpu().numpy())

    def save_ckpt(next_step):
        path, tmp = OUT / "ckpt.pt", OUT / "ckpt.pt.tmp"
        torch.save({"step": next_step, "params": {k: v.detach().cpu() for k, v in params.items()},
                    "opt": {k: o.state_dict() for k, o in optimizers.items()}}, tmp)
        os.replace(tmp, path)
        sync(path)

    ssim_op = make_ssim() if USE_SSIM else None
    t0 = time.perf_counter()
    log = []
    for step in range(start_step, STEPS):
        v = random.choice(train)
        progress = step / max(1, STEPS - 1)
        optimizers["means"].param_groups[0]["lr"] = lrs["means"] * (0.01 ** progress)
        pred, info = render(v)
        target, m = gt(v), v["mask"]
        loss = ((pred - target).abs() * m).sum() / (m.sum() * 3).clamp_min(1)
        if USE_SSIM:
            loss = 0.8 * loss + 0.2 * (1 - ssim_op(pred * m, target * m))
        loss = loss + 0.01 * params["opacities"].sigmoid().mean() + 0.01 * params["scales"].exp().mean()
        loss.backward()
        radii = info["radii"]
        visible = (radii[0] > 0).all(-1) if radii.ndim == 3 else radii[0] > 0
        for opt in optimizers.values():
            opt.step(visible)
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    lr=optimizers["means"].param_groups[0]["lr"])
        with torch.no_grad():
            params["scales"].clamp_(np.log(0.005), np.log(MAX_SCALE))
        if step == 0 or (step + 1) % 500 == 0:
            rate = (step + 1 - start_step) / (time.perf_counter() - t0)
            print(f"step {step + 1}/{STEPS} loss={loss.item():.4f} gaussians={len(params['means'])} "
                  f"visible={int(visible.sum())} {rate:.1f} it/s", flush=True)
            log.append(dict(step=step + 1, loss=loss.item(), gaussians=len(params["means"])))
        if CKPT_EVERY and (step + 1) % CKPT_EVERY == 0 and step + 1 < STEPS:
            save_ckpt(step + 1)
        if (step + 1) % 5000 == 0 and step + 1 < STEPS:
            export("_partial")
            sync(OUT / "gaussians_partial.ply")
            if test:
                print(f"[EVAL] step {step + 1} held-out PSNR {evaluate()}", flush=True)

    metrics = {"psnr_per_camera": evaluate() if test else {}, "gaussians": len(params["means"]),
               "frames": len(frame_ids), "steps": STEPS, "train_seconds": time.perf_counter() - t0, "log": log}
    export("")
    sync(OUT / "gaussians.ply")
    for cam in ("image_00", "image_02", "image_03"):
        v = next((x for x in test or train if x["cam"] == cam), None)
        if v is not None:
            with torch.no_grad():
                pred, _ = render(v)
            sheet = torch.cat([gt(v), pred.clamp(0, 1)], dim=1).cpu().numpy()
            cv2.imwrite(str(OUT / f"compare_{cam}.png"), sheet[:, :, ::-1] * 255)
            sync(OUT / f"compare_{cam}.png")
    (OUT / "metrics.json").write_text(json.dumps({**metrics, "config": config}, indent=2))
    sync(OUT / "metrics.json")
    if Path(LOG_PATH).exists():
        shutil.copyfile(LOG_PATH, OUT / "train.log")
        sync(OUT / "train.log")
    print(f"[DONE] {metrics['train_seconds']:.0f}s | held-out PSNR {metrics['psnr_per_camera']}", flush=True)
    print("[OUT]", OUT, flush=True)


if __name__ == "__main__":
    main()
