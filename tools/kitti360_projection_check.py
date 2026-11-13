"""KITTI-360 LiDAR -> camera projection sanity check (run before any long train).

Self-contained so it can be sent with `colab exec -f`. For one frame it projects
the Velodyne scan into all four cameras along three independent chains and saves
overlays plus a numeric edge-alignment score:

  A "official":  calib_cam_to_velo + calib_cam_to_pose + perspective.txt + Mei yaml
                 (KITTI-360 devkit convention, treated as ground truth).
  B "trainer":   rosa pkl `lidar2cam` + `camera_intrinsic`, exactly as
                 colab_cli_kitti360.py builds its view matrices.
  C "world":     scan of frame F is lifted to world with the trainer pose chain
                 (cam0_to_world @ CAM_FRONT.lidar2cam) and re-projected into
                 frame F+offset with the trainer view matrix. Only aligns if the
                 multi-frame pose convention is right.

Env:
  KITTI360_ARCHIVE   default /content/drive/MyDrive/datasets/kitti360_full
  KITTI360_SEQ       default 2013_05_28_drive_0003_sync
  KITTI360_FRAME     default: first frame present in both pkl and poses
  KITTI360_OFFSET    default 5 (frame offset for chain C)
  KITTI360_CHECK_OUT default /content/kitti360_projection_check
"""
import json
import os
import pickle
import re
import tarfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

ARCHIVE = Path(os.environ.get("KITTI360_ARCHIVE", "/content/drive/MyDrive/datasets/kitti360_full"))
HF = "https://huggingface.co/datasets/danny042/KITTI-360-fisheye/resolve/main"
POSES_URL = ("https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/"
             "89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip")
SEQ = os.environ.get("KITTI360_SEQ", "2013_05_28_drive_0003_sync")
OUT = Path(os.environ.get("KITTI360_CHECK_OUT", "/content/kitti360_projection_check"))
OFFSET = int(os.environ.get("KITTI360_OFFSET", "5"))
# (rosa pkl key, KITTI-360 camera folder, model) as used by the trainer.
CAMERAS = [
    ("CAM_FRONT", "image_00", "pinhole"),
    ("CAM_FRONT_RIGHT", "image_01", "pinhole"),
    ("CAM_BACK_RIGHT", "image_02", "fisheye"),
    ("CAM_BACK_LEFT", "image_03", "fisheye"),
]


# ---------------------------------------------------------------- calibration
def read_records(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip():
            try:
                out[key.strip()] = np.array([float(v) for v in value.split()])
            except ValueError:
                pass
    return out


def to4(values):
    m = np.eye(4)
    m[:3] = np.asarray(values, dtype=float).reshape(3, 4)
    return m


def read_mei_yaml(path):
    text = Path(path).read_text()

    def num(key):
        match = re.search(rf"\b{key}\s*:\s*([-+0-9.eE]+)", text)
        if not match:
            raise ValueError(f"{key} missing in {path}")
        return float(match.group(1))

    return dict(
        xi=num("xi"), k1=num("k1"), k2=num("k2"), p1=num("p1"), p2=num("p2"),
        K=np.array([[num("gamma1"), 0, num("u0")], [0, num("gamma2"), num("v0")], [0, 0, 1]]),
        size=(int(num("image_width")), int(num("image_height"))),
    )


def read_official_calibration(calib_dir):
    """Return {cam: dict(T=cam_from_velo 4x4, model, K, ...)} in devkit convention."""
    cam_to_pose = {k: to4(v) for k, v in read_records(calib_dir / "calib_cam_to_pose.txt").items()}
    cam0_from_velo = np.linalg.inv(to4(np.loadtxt(calib_dir / "calib_cam_to_velo.txt").ravel()))
    persp = read_records(calib_dir / "perspective.txt")
    result = {}
    for cam in ("image_00", "image_01", "image_02", "image_03"):
        camk_from_cam0 = np.linalg.inv(cam_to_pose[cam]) @ cam_to_pose["image_00"]
        unrect = camk_from_cam0 @ cam0_from_velo
        if cam in ("image_00", "image_01"):
            idx = cam[-2:]
            r = np.eye(4)
            r[:3, :3] = persp[f"R_rect_{idx}"].reshape(3, 3)
            P = persp[f"P_rect_{idx}"].reshape(3, 4)
            # P_rect_01 carries a baseline term in column 3; in devkit camera 1 is
            # modeled with its own camToPose so only the 3x3 part is the intrinsic.
            result[cam] = dict(model="pinhole", T=r @ unrect, K=P[:, :3],
                               size=tuple(int(v) for v in persp[f"S_rect_{idx}"]))
        else:
            result[cam] = dict(model="mei", T=unrect, **read_mei_yaml(calib_dir / f"{cam}.yaml"))
    rect0_from_velo = result["image_00"]["T"]
    return result, rect0_from_velo


# ----------------------------------------------------------------- projection
def project_pinhole(pts_cam, K):
    z = pts_cam[:, 2]
    valid = z > 0.5
    uv = np.full((len(pts_cam), 2), np.nan)
    uv[valid] = (pts_cam[valid, :2] / z[valid, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, valid


def project_mei(pts_cam, K, xi, k1, k2, p1=0.0, p2=0.0):
    norm = np.linalg.norm(pts_cam, axis=1)
    denom = pts_cam[:, 2] + xi * norm
    valid = (norm > 0.5) & (denom > 1e-6)
    uv = np.full((len(pts_cam), 2), np.nan)
    x = pts_cam[valid, 0] / denom[valid]
    y = pts_cam[valid, 1] / denom[valid]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv[valid] = np.column_stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]])
    return uv, valid


def project_equidistant(pts_cam, K, radial4):
    """The model gsplat renders with: OpenCV fisheye, theta_d = theta(1+k1 t^2+...)."""
    norm_xy = np.linalg.norm(pts_cam[:, :2], axis=1)
    theta = np.arctan2(norm_xy, pts_cam[:, 2])
    valid = (np.linalg.norm(pts_cam, axis=1) > 0.5) & (theta < np.pi / 2)
    t2 = theta * theta
    theta_d = theta * (1 + radial4[0] * t2 + radial4[1] * t2**2 + radial4[2] * t2**3 + radial4[3] * t2**4)
    scale = np.where(norm_xy > 1e-9, theta_d / np.maximum(norm_xy, 1e-9), 1.0)
    uv = np.column_stack([K[0, 0] * pts_cam[:, 0] * scale + K[0, 2], K[1, 1] * pts_cam[:, 1] * scale + K[1, 2]])
    uv[~valid] = np.nan
    return uv, valid


def mei_to_fisheye(xi, k1, k2):
    """Copied from colab_cli_kitti360.py so chain B reproduces the trainer."""
    theta = np.linspace(0.02, 1.45, 256)
    u = np.sin(theta) / (np.cos(theta) + xi)
    u2 = u * u
    r = (1 + xi) * u * (1 + k1 * u2 + k2 * u2 * u2)
    X = np.column_stack([theta**2, theta**4, theta**6, theta**8])
    k, *_ = np.linalg.lstsq(X, r / theta - 1, rcond=None)
    return k


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


# -------------------------------------------------------------------- scoring
def edge_score(gray, uv, valid, depth):
    """Mean edge strength under projected depth discontinuities vs. under all points.

    Points whose depth jumps relative to their image neighbours should land on image
    edges if alignment is good. Ratio ~1 means no correlation (misaligned); >1.3 is
    typical of a correct calibration.
    """
    h, w = gray.shape
    inside = valid & (uv[:, 0] >= 1) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 1)
    if inside.sum() < 200:
        return dict(inside=int(inside.sum()), ratio=float("nan"))
    edges = cv2.GaussianBlur(cv2.Canny(gray, 50, 150).astype(np.float32), (0, 0), 2.0)
    px = uv[inside].astype(int)
    d = depth[inside]
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (px[:, 1], px[:, 0]), d)
    # Depth range inside an 11x11 window over the sparse z-buffer.
    kernel = np.ones((11, 11), np.uint8)
    near = cv2.erode(zbuf, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=np.inf)
    far = cv2.dilate(np.where(np.isinf(zbuf), -np.inf, zbuf).astype(np.float32), kernel,
                     borderType=cv2.BORDER_CONSTANT, borderValue=-np.inf)
    jump = far - near
    disc = np.nan_to_num(jump[px[:, 1], px[:, 0]]) > 0.1 * d.clip(1, None) + 0.5
    all_mean = edges[px[:, 1], px[:, 0]].mean()
    disc_mean = edges[px[disc, 1], px[disc, 0]].mean() if disc.any() else np.nan
    return dict(inside=int(inside.sum()), discontinuities=int(disc.sum()),
                ratio=float(disc_mean / max(all_mean, 1e-6)))


def overlay(bgr, uv, valid, depth, title):
    img = bgr.copy()
    h, w = img.shape[:2]
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    order = np.argsort(-depth[inside])
    px = uv[inside][order].astype(int)
    hue = (np.clip(np.log(depth[inside][order]) / np.log(60), 0, 1) * 120).astype(np.uint8)
    colors = cv2.cvtColor(np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], -1)[None],
                          cv2.COLOR_HSV2BGR)[0]
    radius = max(1, round(w / 700))
    for (u, v), c in zip(px, colors):
        cv2.circle(img, (int(u), int(v)), radius, tuple(int(x) for x in c), -1)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return img


# ----------------------------------------------------------------------- data
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120)


def ensure_small_files():
    """Download calibration, metadata and poses into ARCHIVE when missing (no Drive needed)."""
    wanted = [f"calibration/{n}" for n in ("calib_cam_to_pose.txt", "calib_cam_to_velo.txt",
                                           "perspective.txt", "image_02.yaml", "image_03.yaml")]
    wanted.append("ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl")
    for rel in wanted:
        dst = ARCHIVE / rel
        if not (dst.exists() and dst.stat().st_size > 0):
            dst.parent.mkdir(parents=True, exist_ok=True)
            with fetch(f"{HF}/{rel}") as src, dst.open("wb") as out:
                out.write(src.read())
            print("[DL]", rel, dst.stat().st_size, "bytes")
    dst = ARCHIVE / "data_poses.zip"
    if not (dst.exists() and dst.stat().st_size > 0):
        with fetch(POSES_URL) as src, dst.open("wb") as out:
            out.write(src.read())
        print("[DL] data_poses.zip", dst.stat().st_size, "bytes")


def tar_members(tar_path, suffix, stems):
    """Read members from a local tar, or stream them from Hugging Face when it is missing."""
    out, names = {}, {}
    tar_path = Path(tar_path)
    if tar_path.exists():
        source, mode = tar_path.open("rb"), "r:*"
    else:
        source, mode = fetch(f"{HF}/{tar_path.relative_to(ARCHIVE).as_posix()}"), "r|*"
        print("[STREAM]", tar_path.name)
    with source, tarfile.open(fileobj=source, mode=mode) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(suffix) and Path(m.name).stem in stems:
                out.setdefault(Path(m.name).stem, tar.extractfile(m).read())
                names.setdefault(Path(m.name).stem, []).append(m.name)
                if len(out) == len(stems):
                    break
    return out, names


def read_cam0_to_world():
    with ZipFile(ARCHIVE / "data_poses.zip") as z, z.open(f"{SEQ}/cam0_to_world.txt") as f:
        poses = {}
        for line in f:
            v = np.array(line.split(), dtype=float)
            if len(v) == 17:
                poses[int(v[0])] = v[1:].reshape(4, 4)
    return poses


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_small_files()
    report = {"seq": SEQ}
    official_cal, rect0_from_velo = read_official_calibration(ARCHIVE / "calibration")

    with (ARCHIVE / "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl").open("rb") as f:
        items = {x["frame_id"]: x for x in pickle.load(f)["data_list"] if SEQ in x["lidar_path"]}
    poses = read_cam0_to_world()
    common = sorted(set(items) & set(poses))
    frame = int(os.environ.get("KITTI360_FRAME", common[0]))
    later = next((f for f in common if f >= frame + OFFSET), None)
    report.update(frame=frame, later_frame=later)
    print(f"[DATA] frame {frame}, chain-C target frame {later}")

    stems = {f"{frame:010d}"} | ({f"{later:010d}"} if later is not None else set())
    images = {}
    for _, cam, _ in CAMERAS:
        data, names = tar_members(ARCHIVE / f"data_2d_raw/{SEQ}_{cam}.tar", ".png", stems)
        images[cam] = {k: cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) for k, v in data.items()}
        report[f"{cam}_tar_members"] = names
        print(f"[DATA] {cam}: {names}")
    scans, names = tar_members(ARCHIVE / f"data_3d_raw/{SEQ}.tar", ".bin", {f"{frame:010d}"})
    print(f"[DATA] lidar: {names}")
    velo = np.frombuffer(scans[f"{frame:010d}"], np.float32).reshape(-1, 4)[:, :3].astype(float)
    velo = velo[np.linalg.norm(velo, axis=1) > 2.5]

    item = items[frame]
    meta_front = np.array(item["images"]["CAM_FRONT"]["lidar2cam"], dtype=float)
    report["pkl_img_paths"] = {key: item["images"][key].get("img_path") for key, _, _ in CAMERAS}
    print("[META] pkl img_path per key:", json.dumps(report["pkl_img_paths"], indent=1))

    rows, cam_rows = [], {}
    for key, cam, model in CAMERAS:
        bgr = images[cam][f"{frame:010d}"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        info = item["images"][key]
        cal = official_cal[cam]
        res = {"image_size": [bgr.shape[1], bgr.shape[0]], "official_size": list(cal["size"])}

        # A: official devkit chain.
        pts = transform(cal["T"], velo)
        uv_a, ok_a = (project_pinhole(pts, cal["K"]) if model == "pinhole"
                      else project_mei(pts, cal["K"], cal["xi"], cal["k1"], cal["k2"], cal["p1"], cal["p2"]))
        res["A_official"] = edge_score(gray, uv_a, ok_a, np.linalg.norm(pts, axis=1))

        # B: trainer chain (pkl lidar2cam/intrinsics, equidistant fit for fisheye).
        T_b = np.array(info["lidar2cam"], dtype=float)
        K_b = np.array(info["camera_intrinsic"], dtype=float)
        pts_b = transform(T_b, velo)
        if model == "pinhole":
            uv_b, ok_b = project_pinhole(pts_b, K_b)
        else:
            k1, k2, _, _, xi = info["distortion_coefficients"]
            Kf = K_b.copy()
            Kf[0, 0] /= 1 + xi
            Kf[1, 1] /= 1 + xi
            uv_b, ok_b = project_equidistant(pts_b, Kf, mei_to_fisheye(xi, k1, k2))
            # B': same pkl extrinsic but exact Mei, isolates the equidistant fit error.
            uv_bm, ok_bm = project_mei(pts_b, K_b, xi, k1, k2)
            both = ok_b & ok_bm
            res["B_fit_vs_mei_px"] = float(np.nanmedian(np.linalg.norm(uv_b[both] - uv_bm[both], axis=1)))
        res["B_trainer"] = edge_score(gray, uv_b, ok_b, np.linalg.norm(pts_b, axis=1))
        res["T_pkl_minus_official_maxabs"] = float(np.abs(T_b - cal["T"]).max())
        res["K_pkl"], res["K_official"] = K_b.round(3).tolist(), np.asarray(cal["K"]).round(3).tolist()
        both = ok_a & ok_b
        res["A_vs_B_median_px"] = float(np.nanmedian(np.linalg.norm(uv_a[both] - uv_b[both], axis=1))) if both.any() else None

        panels = [overlay(bgr, uv_a, ok_a, np.linalg.norm(pts, axis=1), f"A official {cam}"),
                  overlay(bgr, uv_b, ok_b, np.linalg.norm(pts_b, axis=1), f"B trainer {key}->{cam}")]

        # C: multi-frame world consistency with the trainer pose chain.
        if later is not None:
            world_from_lidar = poses[frame] @ meta_front          # trainer: official @ front_lidar2cam
            later_world_from_lidar = poses[later] @ np.array(items[later]["images"]["CAM_FRONT"]["lidar2cam"], dtype=float)
            view = np.array(items[later]["images"][key]["lidar2cam"], dtype=float) @ np.linalg.inv(later_world_from_lidar)
            pts_c = transform(view @ world_from_lidar, velo)
            later_bgr = images[cam][f"{later:010d}"]
            if model == "pinhole":
                uv_c, ok_c = project_pinhole(pts_c, K_b)
            else:
                uv_c, ok_c = project_mei(pts_c, K_b, xi, k1, k2)
            res["C_world_trainer"] = edge_score(cv2.cvtColor(later_bgr, cv2.COLOR_BGR2GRAY), uv_c, ok_c,
                                                np.linalg.norm(pts_c, axis=1))
            # C*: same with the documented convention world_from_velo = cam0_to_world @ rect0_from_velo.
            view_ref = official_cal[cam]["T"] @ np.linalg.inv(poses[later] @ rect0_from_velo)
            pts_r = transform(view_ref @ poses[frame] @ rect0_from_velo, velo)
            uv_r, ok_r = (project_pinhole(pts_r, cal["K"]) if model == "pinhole"
                          else project_mei(pts_r, cal["K"], cal["xi"], cal["k1"], cal["k2"], cal["p1"], cal["p2"]))
            res["C_world_official"] = edge_score(cv2.cvtColor(later_bgr, cv2.COLOR_BGR2GRAY), uv_r, ok_r,
                                                 np.linalg.norm(pts_r, axis=1))
            panels += [overlay(later_bgr, uv_c, ok_c, np.linalg.norm(pts_c, axis=1), f"C trainer world {frame}->{later}"),
                       overlay(later_bgr, uv_r, ok_r, np.linalg.norm(pts_r, axis=1), f"C official world {frame}->{later}")]

        sheet = np.concatenate(panels, axis=0 if model == "pinhole" else 1)
        if model == "fisheye":
            sheet = cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"overlay_{cam}.png"), sheet)
        cam_rows[cam] = res
        print(f"[{cam}] " + json.dumps({k: v for k, v in res.items() if not k.startswith("K_")}))
        rows.append(sheet)

    report["cameras"] = cam_rows
    (OUT / "projection_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nEdge ratio ~1.0 = misaligned, >1.3 = aligned. Compare A (ground truth) with B/C.")
    print("[OUT]", OUT)


if __name__ == "__main__":
    main()
