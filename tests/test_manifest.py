import json

import pytest

from fisheye_fusion.manifest import load_manifest, validate_manifest
from fisheye_fusion.models import ProjectManifest


def _transform():
    return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _camera(camera_id, model="opencv_pinhole"):
    return {
        "id": camera_id,
        "model": model,
        "resolution": [640, 480],
        "K": [[320, 0, 320], [0, 320, 240], [0, 0, 1]],
        "D": [0, 0, 0, 0],
        "T_rig_from_camera": _transform(),
    }


def _payload(camera_ids=("front",), model="opencv_pinhole"):
    cameras = [_camera(camera_id, model) for camera_id in camera_ids]
    return {
        "schema_version": 1,
        "project": "fixture",
        "cameras": cameras,
        "frames": [
            {
                "timestamp_ns": 0,
                "images": {camera_id: f"images/{camera_id}/000.png" for camera_id in camera_ids},
                "T_world_from_rig": _transform(),
            }
        ],
        "fusion": {
            "camera_ids": list(camera_ids),
            "output": "equirectangular",
            "resolution": [1024, 512],
        },
    }


def test_valid_pinhole_and_fisheye_manifests():
    assert ProjectManifest.model_validate(_payload()) .cameras[0].model == "opencv_pinhole"
    assert ProjectManifest.model_validate(_payload(("front",), "opencv_fisheye")).cameras[0].model == "opencv_fisheye"


@pytest.mark.parametrize("camera_ids", [("front", "rear"), ("front", "rear", "left", "right")])
def test_valid_multi_camera_fusion(tmp_path, camera_ids):
    for camera_id in camera_ids:
        image = tmp_path / "images" / camera_id / "000.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.touch()
    manifest = ProjectManifest.model_validate(_payload(camera_ids))
    assert validate_manifest(manifest, tmp_path) is manifest


def test_duplicate_camera_ids_rejected():
    payload = _payload(("front", "front"))
    with pytest.raises(ValueError, match="unique"):
        ProjectManifest.model_validate(payload)


def test_invalid_focal_lengths_rejected():
    payload = _payload()
    payload["cameras"][0]["K"][0][0] = 0
    with pytest.raises(ValueError, match="focal"):
        ProjectManifest.model_validate(payload)


def test_missing_image_reference_rejected(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        load_manifest(manifest_path)


def test_path_traversal_rejected(tmp_path):
    payload = _payload()
    payload["frames"][0]["images"]["front"] = "../outside.png"
    (tmp_path / "outside.png").touch()
    with pytest.raises(ValueError, match="escapes"):
        validate_manifest(ProjectManifest.model_validate(payload), tmp_path)
