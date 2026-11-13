import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

from fisheye_fusion import training_data
from fisheye_fusion.training_data import inspect_dataset, prepare_dataset


def _arrays(count=4):
    return dict(
        images=np.zeros((count, 8, 12, 3), dtype=np.float32),
        Ks=np.tile([[10., 0, 6], [0, 10, 4], [0, 0, 1]], (count, 1, 1)),
        viewmats=np.tile(np.eye(4), (count, 1, 1)),
        points=np.ones((100, 3)), colors=np.full((100, 3), 0.5),
        train_indices=np.arange(count - 1), test_indices=np.array([count - 1]),
    )


def _rig(tmp_path, count=2, size=(800, 400)):
    cameras = []
    for index in range(count):
        extrinsic = np.eye(4)
        angle = index * np.pi / 2
        extrinsic[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]]
        extrinsic[:3, 3] = [index + 1, 0, 0.5]
        cameras.append(dict(
            id=f"cam{index}", model="opencv_fisheye", resolution=list(size),
            K=[[400., 0, size[0] / 2], [0, 400., size[1] / 2], [0, 0, 1]],
            D=[0.1, -0.02, 0.003, -0.0004],
            T_rig_from_camera=extrinsic.tolist(),
        ))
    frames = []
    for frame_index in range(3):
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [frame_index * 2, 3, 4]
        images = {}
        for camera in cameras:
            name = f"{frame_index}_{camera['id']}.png"
            image = np.full((size[1], size[0], 3), frame_index * 50, np.uint8)
            assert cv2.imwrite(str(tmp_path / name), image)
            images[camera["id"]] = name
        frames.append(dict(timestamp_ns=frame_index, images=images,
                           image_timestamps_ns={key: frame_index for key in images},
                           T_world_from_rig=pose.tolist()))
    np.savez(tmp_path / "points.npz", points=np.ones((100, 3)),
             colors=np.full((100, 3), 0.5))
    payload = dict(project="fixture", cameras=cameras, frames=frames,
                   fusion={"camera_ids": [c["id"] for c in cameras]},
                   training_points="points.npz")
    _write_manifest(tmp_path, payload)
    return payload


def _write_manifest(root, payload):
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("count", [2, 4])
def test_fisheye_extrinsics_and_whole_frame_holdout(tmp_path, count):
    payload = _rig(tmp_path, count)
    data = prepare_dataset(tmp_path, "manifest.json")
    assert data["images"].shape == (count * 3, 200, 400, 3)
    np.testing.assert_array_equal(data["train_indices"], np.arange(count * 2))
    np.testing.assert_array_equal(data["test_indices"], np.arange(count * 2, count * 3))
    assert set(data["camera_ids"][data["test_indices"]]) == {
        camera["id"] for camera in payload["cameras"]}
    for index, view in enumerate(data["viewmats"]):
        frame = payload["frames"][index // count]
        camera = payload["cameras"][index % count]
        world = np.array(frame["T_world_from_rig"]) @ camera["T_rig_from_camera"]
        np.testing.assert_allclose(view @ world, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(data["distortions"][index], camera["D"])
        expected_k = np.array(camera["K"])
        expected_k[:2] *= 0.5
        np.testing.assert_allclose(data["Ks"][index], expected_k)
    assert (data["camera_models"] == "fisheye").all()
    np.testing.assert_array_equal(data["points"], np.ones((100, 3)))
    summary = inspect_dataset(tmp_path, "manifest.json", ["cam1"])
    assert summary == dict(observations=3, observation_count=3, camera_ids=["cam1"],
                           cameras=[{"id": "cam1", "model": "fisheye"}],
                           camera_models=["fisheye"], train_count=2, test_count=1,
                           resolution=[400, 200], gaussians=100)
    json.dumps(summary)


def test_npz_defaults_and_subset_remapping(tmp_path):
    arrays = _arrays()
    np.savez(tmp_path / "data.npz", **arrays)
    data = prepare_dataset(tmp_path, "data.npz")
    assert (data["camera_models"] == "pinhole").all()
    assert not data["distortions"].any()
    with pytest.raises(ValueError, match="nonempty"):
        prepare_dataset(tmp_path, "data.npz", [])
    with pytest.raises(ValueError, match="no camera_ids"):
        prepare_dataset(tmp_path, "data.npz", ["front"])
    arrays.update(camera_ids=np.array(["a", "b", "a", "b"]),
                  camera_models=np.full(4, "fisheye"),
                  distortions=np.tile([0.1, 0.2, 0.3, 0.4], (4, 1)))
    np.savez(tmp_path / "data.npz", **arrays)
    data = prepare_dataset(tmp_path, "data.npz", ["b"])
    assert data["images"].shape[0] == 2
    np.testing.assert_array_equal(data["train_indices"], [0])
    np.testing.assert_array_equal(data["test_indices"], [1])
    np.testing.assert_allclose(data["distortions"], arrays["distortions"][[1, 3]])
    for selection in ([], ["unknown"], ["b", "b"], ["a"]):
        with pytest.raises(ValueError):
            prepare_dataset(tmp_path, "data.npz", selection)


@pytest.mark.parametrize("key,value", [
    ("images", np.zeros((33, 1, 1, 3))),
    ("images", np.zeros((4, 513, 1, 3))),
    ("images", np.zeros((4, 1, 513, 3))),
    ("images", np.zeros((4, 1, 1, 4))),
    ("images", np.full((4, 1, 1, 3), 1.1)),
    ("images", np.full((4, 1, 1, 3), np.nan)),
    ("points", np.ones((99, 3))),
    ("points", np.ones((30001, 3))),
    ("points", np.full((100, 3), np.inf)),
    ("colors", np.full((100, 3), -0.1)),
    ("colors", np.ones((99, 3))),
    ("Ks", np.zeros((4, 3, 3))),
    ("viewmats", np.tile(np.diag([2., 1, 1, 1]), (4, 1, 1))),
    ("train_indices", np.array([0., 1.])),
    ("train_indices", np.array([0, 0])),
    ("train_indices", np.array([-1])),
    ("train_indices", np.array([4])),
    ("train_indices", np.array([3])),
    ("test_indices", np.array([], dtype=int)),
    ("camera_models", np.full(4, "mei")),
    ("camera_models", np.array(["pinhole"])),
    ("distortions", np.zeros((4, 5))),
    ("distortions", np.ones((4, 4))),
    ("camera_ids", np.array(["a"])),
    ("images", np.array([object()], dtype=object)),
])
def test_invalid_npz(tmp_path, key, value):
    arrays = _arrays()
    arrays[key] = value
    np.savez(tmp_path / "data.npz", **arrays)
    with pytest.raises(ValueError):
        prepare_dataset(tmp_path, "data.npz")


def test_npz_header_limits_before_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("np.load must not run before archive/header validation")

    monkeypatch.setattr(np, "load", forbidden)
    path = tmp_path / "large.npz"
    # Real ZIP headers report >200 MiB, despite a small compressed file.
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("images.npy", "w") as member:
            for _ in range(201):
                member.write(bytes(1024**2))
    with pytest.raises(ValueError, match="200 MiB"):
        prepare_dataset(tmp_path, path.name)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="<f8", fortran_order=False, shape=(2**40,)))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images.npy", header.getvalue())
    with pytest.raises(ValueError, match="array size"):
        prepare_dataset(tmp_path, path.name)


@pytest.mark.parametrize("change,match", [
    ("no_points", "training_points"),
    ("points_escape", "unsafe|root"),
    ("image_escape", "root"),
    ("root_escape", "root"),
    ("offset", "synchronization"),
    ("mei", "unsupported"),
    ("opencv_pinhole", "unsupported"),
    ("pinhole_distortion", "distortion"),
    ("one_frame", "frames"),
    ("too_many", "observations"),
    ("bad_image", "decode"),
    ("wrong_resolution", "resolution"),
    ("different_shapes", "same shape"),
])
def test_rig_errors(tmp_path, change, match):
    payload = _rig(tmp_path, size=(20, 10))
    if change == "no_points":
        del payload["training_points"]
    elif change == "points_escape":
        payload["training_points"] = "../outside.npz"
    elif change == "image_escape":
        payload["frames"][0]["images"]["cam0"] = "../outside.png"
    elif change == "root_escape":
        payload["root"] = ".."
    elif change == "offset":
        payload["frames"][0]["image_timestamps_ns"]["cam1"] = 1
    elif change in ("mei", "opencv_pinhole"):
        payload["cameras"][0]["model"] = change
        if change == "mei":
            payload["cameras"][0]["xi"] = 1
    elif change == "pinhole_distortion":
        payload["cameras"][0]["model"] = "pinhole"
    elif change == "one_frame":
        payload["frames"] = payload["frames"][:1]
    elif change == "too_many":
        frame = payload["frames"][0]
        payload["frames"] = [dict(frame, timestamp_ns=i, image_timestamps_ns={})
                             for i in range(17)]
    elif change == "bad_image":
        (tmp_path / "0_cam0.png").write_bytes(b"invalid")
    elif change == "wrong_resolution":
        payload["cameras"][0]["resolution"] = [30, 10]
    elif change == "different_shapes":
        payload["cameras"][1]["resolution"] = [10, 10]
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"{i}_cam1.png"), np.zeros((10, 10, 3), np.uint8))
    _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        prepare_dataset(tmp_path, "manifest.json")


def test_pinhole_and_selection_errors(tmp_path):
    payload = _rig(tmp_path, 4, size=(20, 10))
    for camera in payload["cameras"]:
        camera.update(model="pinhole", D=[])
    for frame in payload["frames"]:
        frame.pop("image_timestamps_ns")
    _write_manifest(tmp_path, payload)
    data = prepare_dataset(tmp_path, "manifest.json", ["cam0", "cam3"])
    assert (data["camera_models"] == "pinhole").all()
    assert not data["distortions"].any()
    for selection in ([], ["cam0"] * 2, ["unknown"], ["cam0", "cam1", "cam2"]):
        with pytest.raises(ValueError):
            prepare_dataset(tmp_path, "manifest.json", selection)


def test_workspace_paths_and_missing_keys(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    np.savez(tmp_path / "outside.npz", **_arrays())
    for path in ("../outside.npz", str(tmp_path / "outside.npz"), "missing.npz"):
        with pytest.raises(ValueError):
            prepare_dataset(root, path)
    np.savez(root / "in.npz", **_arrays())
    assert len(prepare_dataset(root, str(root / "in.npz"))["images"]) == 4
    np.savez(root / "missing.npz", points=np.ones((100, 3)))
    with pytest.raises(ValueError, match="missing training"):
        prepare_dataset(root, "missing.npz")


def test_seed_symlink_rejected(tmp_path):
    payload = _rig(tmp_path, size=(20, 10))
    link = tmp_path / "link.npz"
    try:
        link.symlink_to(tmp_path / "points.npz")
    except OSError:
        pytest.skip("symlinks unavailable")
    payload["training_points"] = link.name
    _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="symlink"):
        prepare_dataset(tmp_path, "manifest.json")


def test_inspect_npz_ui_contract(tmp_path):
    arrays = _arrays()
    np.savez(tmp_path / "stump.npz", **arrays)
    summary = inspect_dataset(tmp_path, "stump.npz")
    assert summary["observations"] == summary["observation_count"] == 4
    assert type(summary["observations"]) is int
    assert summary["cameras"] == summary["camera_ids"] == []
    assert summary["resolution"] == [12, 8]
    assert summary["gaussians"] == 100
    json.dumps(summary)
    arrays.update(camera_ids=np.array(["rear", "front", "rear", "front"]),
                  camera_models=np.array(["fisheye", "pinhole", "fisheye", "pinhole"]))
    np.savez(tmp_path / "rig.npz", **arrays)
    summary = inspect_dataset(tmp_path, "rig.npz")
    assert summary["cameras"] == [{"id": "rear", "model": "fisheye"},
                                  {"id": "front", "model": "pinhole"}]
    summary = inspect_dataset(tmp_path, "rig.npz", ["front"])
    assert summary["cameras"] == [{"id": "front", "model": "pinhole"}]
    assert summary["observations"] == 2
    json.dumps(summary)


def test_rig_cumulative_image_size_checked_before_headers(tmp_path, monkeypatch):
    _rig(tmp_path, size=(20, 10))
    # Each source fits individually; only the cumulative size exceeds the limit.
    sizes = [source.stat().st_size for source in tmp_path.glob("*.png")]
    monkeypatch.setattr(training_data, "MAX_BYTES", sum(sizes) - 1)

    def forbidden(*args, **kwargs):
        pytest.fail("image header/decode must not run before cumulative size check")

    monkeypatch.setattr(Image, "open", forbidden)
    monkeypatch.setattr(cv2, "imread", forbidden)
    with pytest.raises(ValueError, match="cumulative 200 MiB"):
        prepare_dataset(tmp_path, "manifest.json")


@pytest.mark.parametrize("size", [(8193, 1), (1, 8193), (21, 10), None])
def test_all_image_headers_checked_before_decode(tmp_path, monkeypatch, size):
    _rig(tmp_path, size=(20, 10))
    source = tmp_path / "2_cam1.png"
    if size is None:
        source.write_bytes(b"invalid image header")
    else:
        Image.new("RGB", size).save(source)

    def forbidden(*args, **kwargs):
        pytest.fail("OpenCV must not decode before all image headers pass")

    monkeypatch.setattr(cv2, "imread", forbidden)
    with pytest.raises(ValueError, match="8192|resolution|header"):
        prepare_dataset(tmp_path, "manifest.json")


def test_decoded_dimensions_checked_again(tmp_path, monkeypatch):
    _rig(tmp_path, size=(20, 10))
    monkeypatch.setattr(cv2, "imread", lambda *args: np.zeros((1, 8193, 3), np.uint8))
    with pytest.raises(ValueError, match="decoded image dimensions"):
        prepare_dataset(tmp_path, "manifest.json")
