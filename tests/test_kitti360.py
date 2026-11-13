from pathlib import Path
import base64
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fisheye_fusion import server
from fisheye_fusion.kitti360 import build_kitti360_manifest
from fisheye_fusion.manifest import load_manifest


SEQUENCE = "2013_05_28_drive_0000_sync"
CAMERAS = ("image_00", "image_01", "image_02", "image_03")
POSE = "1 0 0 10 0 1 0 20 0 0 1 30"
EXTRINSIC = "1 0 0 1 0 1 0 2 0 0 1 3"


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path):
    write(tmp_path, "calibration/calib_cam_to_pose.txt",
          "\n".join(f"{camera}: {EXTRINSIC}" for camera in CAMERAS))
    write(tmp_path, "calibration/perspective.txt", "\n".join(
        f"P_rect_{i}: 300 0 320 -90 0 301 240 0 0 0 1 0\n"
        f"R_rect_{i}: 0 -1 0 1 0 0 0 0 1\nS_rect_{i}: 640 480"
        for i in ("00", "01")))
    for camera in CAMERAS:
        if camera in CAMERAS[2:]:
            write(tmp_path, f"calibration/{camera}.yaml", """%YAML:1.0
---
image_width: 640
image_height: 480
mirror_parameters:
   xi: 1.2
distortion_parameters:
   k1: 0.01
   k2: -0.02
projection_parameters:
   gamma1: 300.
   gamma2: 301.
   u0: 320.
   v0: 240.
""")
        base = f"data_2d_raw/{SEQUENCE}/{camera}"
        write(tmp_path, f"{base}/timestamps.txt",
              "2013-05-28 12:00:00.123456789\n2013-05-28 12:00:00.123456790\n")
        directory = "data_rect" if camera in CAMERAS[:2] else "data_rgb"
        for index in range(2):
            write(tmp_path, f"{base}/{directory}/{index:010d}.png", "rgb")
    write(tmp_path, f"data_poses/{SEQUENCE}/poses.txt", f"0 {POSE}\n1 {POSE}\n")
    return tmp_path


def test_fisheye_exact_timestamp_pose_and_relative_paths(dataset):
    before = set(dataset.rglob("*"))
    manifest = build_kitti360_manifest(dataset, SEQUENCE, [0, 1])
    assert set(dataset.rglob("*")) == before
    camera = manifest.cameras[0]
    assert camera.model == "mei" and camera.xi == 1.2
    assert camera.D == [0.01, -0.02, 0, 0]
    assert camera.K == [[300, 0, 320], [0, 301, 240], [0, 0, 1]]
    assert camera.resolution == (640, 480)
    assert np.asarray(camera.T_rig_from_camera)[:3, 3].tolist() == [1, 2, 3]
    assert manifest.frames[0].timestamp_ns == 1369742400123456789
    assert manifest.frames[1].timestamp_ns - manifest.frames[0].timestamp_ns == 1
    assert np.asarray(manifest.frames[0].T_world_from_rig)[:3, 3].tolist() == [10, 20, 30]
    assert manifest.root == "."
    for frame in manifest.frames:
        assert frame.lidar is None
        for path in frame.images.values():
            assert not Path(path).is_absolute()
            assert "/data_rgb/" in path and (dataset / path).is_file()


@pytest.mark.parametrize("cameras", [("image_00",), ("image_00", "image_01"), CAMERAS])
def test_rectification_and_camera_counts(dataset, cameras):
    manifest = build_kitti360_manifest(dataset, SEQUENCE, [1], cameras)
    camera = manifest.cameras[0]
    assert camera.model == "pinhole"
    assert "/data_rect/0000000001.png" in manifest.frames[0].images["image_00"]
    expected = np.array([[0, 1, 0, 1], [-1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    np.testing.assert_allclose(camera.T_rig_from_camera, expected)
    assert camera.K[0] == [300, 0, 320]


@pytest.mark.parametrize("sequence", ["../escape", "/abs", "C:\\data", ".", "x/y", "x:y"])
def test_unsafe_sequence(dataset, sequence):
    with pytest.raises(ValueError, match="safe identifier"):
        build_kitti360_manifest(dataset, sequence, [0])


@pytest.mark.parametrize("frames", [[], [1, 0], [0, 0], [-1], [True], [0.5], list(range(33))])
def test_invalid_frames(dataset, frames):
    with pytest.raises(ValueError, match="frame_ids"):
        build_kitti360_manifest(dataset, SEQUENCE, frames)


@pytest.mark.parametrize("cameras", [(), ("image_04",), ("image_02", "image_02"), CAMERAS[:3]])
def test_invalid_cameras(dataset, cameras):
    with pytest.raises(ValueError, match="cameras"):
        build_kitti360_manifest(dataset, SEQUENCE, [0], cameras)


def test_budget_exact_boundary(dataset):
    manifest = build_kitti360_manifest(dataset, SEQUENCE, [0], ("image_00",), max_bytes=3)
    assert len(manifest.frames) == 1
    with pytest.raises(ValueError, match="max_bytes"):
        build_kitti360_manifest(dataset, SEQUENCE, [0], ("image_00",), max_bytes=2)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, 200 * 1024**2 + 1])
def test_invalid_byte_budget(dataset, budget):
    with pytest.raises(ValueError, match="max_bytes"):
        build_kitti360_manifest(dataset, SEQUENCE, [0], max_bytes=budget)


def test_calibration_records_size_cap(dataset):
    path = dataset / "calibration/calib_cam_to_pose.txt"
    with path.open("wb") as stream:
        stream.truncate(4 * 1024**2 + 1)
    with pytest.raises(ValueError, match="4 MiB"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


@pytest.mark.parametrize("cameras", [CAMERAS[2:], CAMERAS])
def test_api_sample_render_export_roundtrip(dataset, monkeypatch, cameras):
    monkeypatch.setattr(server, "WORKSPACE", dataset.parent)
    for camera in cameras:
        directory = "data_rect" if camera in CAMERAS[:2] else "data_rgb"
        image = np.full((480, 640, 3), (45, 110, 190), dtype=np.uint8)
        for index in range(2):
            assert cv2.imwrite(str(dataset / f"data_2d_raw/{SEQUENCE}/{camera}/{directory}/{index:010d}.png"), image)
        # Distinct sub-microsecond camera offsets must survive sampling/export.
        offset = CAMERAS.index(camera)
        write(dataset, f"data_2d_raw/{SEQUENCE}/{camera}/timestamps.txt",
              f"2013-05-28 12:00:00.{123456789 + offset:09d}\n"
              f"2013-05-28 12:00:01.{123456789 + offset:09d}\n")
    output = dataset.name + "-sample"
    expected_times = {camera: 1369742400123456789 + CAMERAS.index(camera) for camera in cameras}
    with TestClient(server.create_app()) as client:
        body = dict(dataset_root=dataset.name, sequence=SEQUENCE, frame_ids=[0, 1],
                    camera_ids=list(cameras), output=output, max_mib=2)
        preview = client.post("/api/import/kitti360", json=body)
        assert preview.status_code == 200, preview.text
        assert not preview.json()["written"]
        assert not (dataset.parent / output).exists()
        imported = client.post("/api/import/kitti360", json={**body, "write": True})
        assert imported.status_code == 200, imported.text
        result = imported.json()
        assert result["written"] and result["frames"] == 2
        assert result["files"] == 2 * len(cameras)
        assert result["bytes"] <= 2 * 1024**2
        sampled = load_manifest(dataset.parent / result["manifest"])
        assert sampled.frames[0].image_timestamps_ns == expected_times
        assert set(sampled.frames[0].images) == set(cameras)
        job = {"manifest": result["manifest"], "width": 512}
        inspected = client.post("/api/inspect", json=job)
        assert inspected.status_code == 200, inspected.text
        assert len(inspected.json()["images"]) == len(cameras)
        rendered = client.post("/api/render", json=job)
        assert rendered.status_code == 200, rendered.text
        assert 0 < rendered.json()["coverage"] <= 1
        pixels = cv2.imdecode(np.frombuffer(base64.b64decode(rendered.json()["png"]), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
        assert pixels.shape == (256, 512, 3) and pixels.any()
        for platform in ("colab", "kaggle"):
            exported = client.post(f"/api/bundle/{platform}", json=job)
            assert exported.status_code == 200, exported.text
            with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
                packed = json.loads(archive.read("manifest.json"))
                assert packed["frames"][0]["image_timestamps_ns"] == expected_times
                assert len(packed["cameras"]) == len(cameras)
                for path in packed["frames"][0]["images"].values():
                    assert archive.read(path)


def test_missing_exact_pose(dataset):
    write(dataset, f"data_poses/{SEQUENCE}/poses.txt", f"1 {POSE}\n")
    with pytest.raises(ValueError, match="missing exact poses"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


@pytest.mark.parametrize("contents", ["", "bad\n", "2013-05-28 12:00:00.1234567891\n"])
def test_missing_or_invalid_timestamp(dataset, contents):
    write(dataset, f"data_2d_raw/{SEQUENCE}/image_03/timestamps.txt", contents)
    with pytest.raises(ValueError, match="timestamp"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


def test_missing_image(dataset):
    (dataset / f"data_2d_raw/{SEQUENCE}/image_02/data_rgb/0000000000.png").unlink()
    with pytest.raises(ValueError, match="missing file"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


def test_missing_yaml_field(dataset):
    path = dataset / "calibration/image_02.yaml"
    path.write_text(path.read_text().replace("   xi: 1.2", "   other: 1.2"), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


def test_malformed_yaml(dataset):
    write(dataset, "calibration/image_02.yaml", "%YAML:1.0\nbroken: [\n")
    with pytest.raises(ValueError, match="calibration"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


def test_duplicate_pose(dataset):
    write(dataset, f"data_poses/{SEQUENCE}/poses.txt", f"0 {POSE}\n0 {POSE}\n")
    with pytest.raises(ValueError, match="duplicate pose"):
        build_kitti360_manifest(dataset, SEQUENCE, [0])


def test_other_camera_nonmonotonic_timestamps(dataset):
    write(dataset, f"data_2d_raw/{SEQUENCE}/image_03/timestamps.txt",
          "2013-05-28 12:00:00.123456789\n2013-05-28 12:00:00.123456789\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        build_kitti360_manifest(dataset, SEQUENCE, [0, 1])


@pytest.mark.parametrize("relative", [
    "calibration/calib_cam_to_pose.txt",
    f"data_poses/{SEQUENCE}/poses.txt",
    f"data_2d_raw/{SEQUENCE}/image_02/timestamps.txt",
    f"data_2d_raw/{SEQUENCE}/image_02/data_rgb/0000000000.png",
])
def test_symlink_outside_root_rejected(dataset, tmp_path, relative):
    root = dataset / "subset"
    root.mkdir()
    # Use a separate dataset root so the link target remains outside that root.
    link = root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(dataset / relative)
    except OSError:
        pytest.skip("symlink creation unavailable")
    from fisheye_fusion.kitti360 import _file
    with pytest.raises(ValueError, match="outside dataset root"):
        _file(root, Path(relative))
