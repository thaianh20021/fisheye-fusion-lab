import json
from pathlib import Path

import cv2
import numpy as np

from fisheye_fusion.manifest import load_manifest
from fisheye_fusion.fusion import fuse_images


def test_manifest_and_four_camera_fusion(tmp_path: Path):
    for camera_id, color in zip(["front", "rear", "left", "right"], [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        path = tmp_path / f"{camera_id}.png"
        cv2.imwrite(str(path), np.full((32, 64, 3), color[::-1], np.uint8))
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    camera = lambda name: {
        "id": name, "model": "pinhole", "resolution": [64, 32],
        "K": [[32, 0, 32], [0, 32, 16], [0, 0, 1]], "D": [], "T_rig_from_camera": identity,
    }
    data = {
        "project": "synthetic", "cameras": [camera(x) for x in ["front", "rear", "left", "right"]],
        "frames": [{"timestamp_ns": 0, "images": {x: f"{x}.png" for x in ["front", "rear", "left", "right"]}, "T_world_from_rig": identity}],
        "fusion": {"camera_ids": ["front", "rear", "left", "right"], "resolution": [64, 32]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    image, coverage = fuse_images(manifest, tmp_path)
    assert image.shape == (32, 64, 3)
    assert coverage.any()
