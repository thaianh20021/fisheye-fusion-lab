"""End-to-end run of tools/kitti360_projection_check.py on a tiny synthetic archive."""
import importlib.util
import io
import json
import pickle
import tarfile
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

SEQ = "2013_05_28_drive_0003_sync"
TOOL = Path(__file__).resolve().parents[1] / "tools" / "kitti360_projection_check.py"


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def flat(m):
    return " ".join(f"{v:.9f}" for v in np.asarray(m)[:3].ravel())


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build_archive(root, frames):
    cal = root / "calibration"
    cal.mkdir(parents=True)
    # velo: x forward, y left, z up; camera: x right, y down, z forward.
    cam0_to_velo = np.array([[0, 0, 1, 0.8], [-1, 0, 0, 0.3], [0, -1, 0, -0.1], [0, 0, 0, 1.0]])
    cam_to_pose = {"image_00": np.eye(4), "image_01": np.eye(4),
                   "image_02": rot_z(0) @ np.eye(4), "image_03": np.eye(4)}
    cam_to_pose["image_01"][0, 3] = 0.6
    cam_to_pose["image_02"][:3, :3] = cv2.Rodrigues(np.array([0, -np.pi / 2, 0]))[0]
    cam_to_pose["image_03"][:3, :3] = cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0]
    (cal / "calib_cam_to_pose.txt").write_text("".join(f"{k}: {flat(v)}\n" for k, v in cam_to_pose.items()))
    (cal / "calib_cam_to_velo.txt").write_text(flat(cam0_to_velo) + "\n")
    K = np.array([[300.0, 0, 200, 0], [0, 300, 100, 0], [0, 0, 1, 0]])
    eye3 = " ".join(map(str, np.eye(3).ravel()))
    (cal / "perspective.txt").write_text(
        f"S_rect_00: 400 200\nR_rect_00: {eye3}\nP_rect_00: {' '.join(map(str, K.ravel()))}\n"
        f"S_rect_01: 400 200\nR_rect_01: {eye3}\nP_rect_01: {' '.join(map(str, K.ravel()))}\n")
    mei = dict(xi=1.2, k1=0.02, k2=-0.001, gamma1=250.0, gamma2=250.0, u0=200.0, v0=200.0)
    for cam in ("image_02", "image_03"):
        (cal / f"{cam}.yaml").write_text(
            "%YAML:1.0\nmodel_type: MEI\nimage_width: 400\nimage_height: 400\n"
            f"mirror_parameters:\n   xi: {mei['xi']}\ndistortion_parameters:\n   k1: {mei['k1']}\n"
            f"   k2: {mei['k2']}\n   p1: 0.0\n   p2: 0.0\nprojection_parameters:\n"
            f"   gamma1: {mei['gamma1']}\n   gamma2: {mei['gamma2']}\n   u0: {mei['u0']}\n   v0: {mei['v0']}\n")

    velo_from_cam0 = cam0_to_velo
    lidar2cam = {}
    for cam in cam_to_pose:
        lidar2cam[cam] = np.linalg.inv(cam_to_pose[cam]) @ cam_to_pose["image_00"] @ np.linalg.inv(velo_from_cam0)
    keys = {"image_00": "CAM_FRONT", "image_01": "CAM_FRONT_RIGHT",
            "image_02": "CAM_BACK_RIGHT", "image_03": "CAM_BACK_LEFT"}
    data_list = []
    for f in frames:
        images = {}
        for cam, key in keys.items():
            fish = cam in ("image_02", "image_03")
            images[key] = dict(
                img_path=f"{SEQ}/{cam}/{f:010d}.png", lidar2cam=lidar2cam[cam].tolist(),
                camera_intrinsic=([[250, 0, 200], [0, 250, 200], [0, 0, 1]] if fish else K[:, :3].tolist()),
                **({"distortion_coefficients": [mei["k1"], mei["k2"], 0, 0, mei["xi"]]} if fish else {}))
        data_list.append(dict(frame_id=f, lidar_path=f"{SEQ}/velodyne/{f:010d}.bin", images=images))
    meta = root / "ImageSets-rosa-6class"
    meta.mkdir()
    (meta / "kitti360_rosa_infos_val.pkl").write_bytes(pickle.dumps({"data_list": data_list}))

    # Vehicle drives along world x; cam0 rect -> world.
    world_from_velo = {f: np.eye(4) for f in frames}
    for f in frames:
        world_from_velo[f][0, 3] = 0.5 * f
    lines = "".join(f"{f} " + " ".join(map(str, (world_from_velo[f] @ velo_from_cam0).ravel())) + "\n" for f in frames)
    with ZipFile(root / "data_poses.zip", "w") as z:
        z.writestr(f"{SEQ}/cam0_to_world.txt", lines)

    rng = np.random.default_rng(0)
    world = rng.uniform([-20, -15, -2], [40, 15, 6], (4000, 3))
    (root / "data_2d_raw").mkdir()
    (root / "data_3d_raw").mkdir()
    for cam in cam_to_pose:
        with tarfile.open(root / f"data_2d_raw/{SEQ}_{cam}.tar", "w") as tar:
            for f in frames:
                size = (400, 400) if cam in ("image_02", "image_03") else (200, 400)
                img = rng.integers(0, 255, (*size, 3), dtype=np.uint8)
                add(tar, f"{SEQ}/{cam}/data_rect/{f:010d}.png", cv2.imencode(".png", img)[1].tobytes())
    with tarfile.open(root / f"data_3d_raw/{SEQ}.tar", "w") as tar:
        for f in frames:
            local = world - world_from_velo[f][:3, 3]
            add(tar, f"{SEQ}/velodyne_points/data/{f:010d}.bin",
                np.column_stack([local, np.ones(len(local))]).astype(np.float32).tobytes())


def test_projection_check_runs_and_chains_agree(tmp_path, monkeypatch):
    archive, out = tmp_path / "archive", tmp_path / "out"
    build_archive(archive, frames=[0, 1, 2, 5, 6])
    monkeypatch.setenv("KITTI360_ARCHIVE", str(archive))
    monkeypatch.setenv("KITTI360_CHECK_OUT", str(out))
    spec = importlib.util.spec_from_file_location("projection_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    report = json.loads((out / "projection_report.json").read_text())
    assert report["frame"] == 0 and report["later_frame"] == 5
    for cam in ("image_00", "image_01", "image_02", "image_03"):
        res = report["cameras"][cam]
        assert (out / f"overlay_{cam}.png").is_file()
        assert res["T_pkl_minus_official_maxabs"] < 1e-6
        assert res["A_official"]["inside"] > 50
        if cam in ("image_00", "image_01"):
            assert res["A_vs_B_median_px"] < 1e-6
        else:
            # B uses the trainer's equidistant polynomial fit of the Mei model.
            assert res["B_fit_vs_mei_px"] < 2.0


def test_edge_score_separates_aligned_from_shifted():
    spec = importlib.util.spec_from_file_location("projection_check", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gray = np.full((200, 400), 40, np.uint8)
    gray[:, 200:] = 200                       # image edge at u=200
    v, u = np.mgrid[5:195:3, 5:395:3]
    uv = np.column_stack([u.ravel(), v.ravel()]).astype(float)
    depth = np.where(uv[:, 0] < 200, 5.0, 30.0)  # depth jump at the same column
    valid = np.ones(len(uv), bool)
    aligned = module.edge_score(gray, uv, valid, depth)["ratio"]
    shifted = module.edge_score(gray, uv + [60, 0], valid, depth)["ratio"]
    assert aligned > 1.5 and shifted < 0.5
