"""Offline checks: python -m unittest discover -s tests -p test_real_scene.py"""

import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.prepare_small_scene import (
    ROOT, LIMIT, DISCOVERY_ALLOWANCE, Downloads, intrinsics,
    read_cameras, read_images, read_points, viewmat,
)


class RealSceneTests(unittest.TestCase):
    def test_binary_and_camera_conventions(self):
        camera = struct.pack("<QiiQQ4d", 1, 7, 1, 800, 600, 500, 510, 400, 300)
        self.assertEqual(read_cameras(camera)[7][0], 1)
        K, distortion = intrinsics(read_cameras(camera)[7], 400, 300)
        np.testing.assert_allclose(K, [[250, 0, 200], [0, 255, 150], [0, 0, 1]])
        self.assertFalse(distortion.any())
        for model, params in [(2, (500, 400, 300, .1)),
                              (3, (500, 400, 300, .1, .2)),
                              (4, (500, 510, 400, 300, .1, .2, .03, .04))]:
            _, d = intrinsics((model, 800, 600, params), 400, 300)
            self.assertAlmostEqual(d[0], .1)
        image = (struct.pack("<Q", 1) +
                 struct.pack("<i7di", 42, np.sqrt(.5), 0, 0, np.sqrt(.5), 1, 2, 3, 7) +
                 b"photo.jpg\0" + struct.pack("<Qddq", 1, 10, 20, -1))
        parsed = read_images(image)["photo.jpg"]
        self.assertEqual(parsed["observations"]["id"][0], -1)
        np.testing.assert_allclose(viewmat(parsed) @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-6)
        point = (struct.pack("<Q", 1) + struct.pack("<Q3d3BdQii", 999, 1, 2, 3,
                                                  255, 128, 0, .5, 1, 42, 0))
        self.assertEqual(read_points(point, {42})[999], ((1, 2, 3), (255, 128, 0)))
        for reader, data in [(read_cameras, camera), (read_images, image),
                             (lambda x: read_points(x, {42}), point)]:
            with self.assertRaises(ValueError):
                reader(data[:-1])

    def test_budget_cleanup_and_cache(self):
        class Response(io.BytesIO):
            headers = {}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            downloads = Downloads(root)
            with patch("urllib.request.urlopen", return_value=Response(b"abc")) as request:
                self.assertEqual(downloads.fetch("https://example.org/a", "a", 3), b"abc")
                downloads.fetch("https://example.org/a", "a", 3)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(downloads.state["charged_bytes"], DISCOVERY_ALLOWANCE + 3)
            with patch("urllib.request.urlopen", return_value=Response(b"x")):
                with self.assertRaises(ValueError):
                    downloads.fetch("https://example.org/b", "b", 3)
            self.assertFalse((root / "b.part").exists())
            self.assertFalse((root / "b").exists())
            self.assertEqual(Downloads(root).state["charged_bytes"], DISCOVERY_ALLOWANCE + 6)
            downloads.state["charged_bytes"] = LIMIT
            with patch("urllib.request.urlopen") as request:
                with self.assertRaises(ValueError):
                    downloads.fetch("https://example.org/c", "c", 1)
                request.assert_not_called()

    @unittest.skipUnless((ROOT / "training.npz").exists(), "Run prepare_small_scene.py first")
    def test_real_artifact_and_reprojection(self):
        metadata = json.loads((ROOT / "provenance.json").read_text())
        self.assertLessEqual(metadata["download_budget_charged_bytes"], LIMIT)
        for name, record in metadata["files"].items():
            data = (ROOT / name).read_bytes()
            self.assertEqual(len(data), record["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
        self.assertEqual(hashlib.sha256((ROOT / "training.npz").read_bytes()).hexdigest(),
                         metadata["training_npz_sha256"])
        with np.load(ROOT / "training.npz", allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"images", "Ks", "viewmats", "points", "colors",
                                               "train_indices", "test_indices"})
            self.assertEqual(data["images"].shape[0], 12)
            self.assertEqual(data["images"].shape[-1], 3)
            self.assertLessEqual(data["images"].shape[2], 400)
            self.assertEqual(data["Ks"].shape, (12, 3, 3))
            self.assertEqual(data["viewmats"].shape, (12, 4, 4))
            self.assertEqual(len(data["train_indices"]), 10)
            self.assertEqual(len(data["test_indices"]), 2)
            self.assertEqual(sorted(np.concatenate([data["train_indices"], data["test_indices"]])),
                             list(range(12)))
            self.assertTrue(0 < len(data["points"]) <= 20000)
            self.assertEqual(data["points"].shape, data["colors"].shape)
            for key in ("images", "Ks", "viewmats", "points", "colors"):
                self.assertEqual(data[key].dtype, np.float32)
                self.assertTrue(np.isfinite(data[key]).all())
            for key in ("images", "colors"):
                self.assertGreaterEqual(data[key].min(), 0)
                self.assertLessEqual(data[key].max(), 1)
            poses = read_images((ROOT / "sparse/0/images.bin").read_bytes())
            cameras = read_cameras((ROOT / "sparse/0/cameras.bin").read_bytes())
            ids = {metadata["images"][i]["image_id"] for i in data["train_indices"]}
            points = read_points((ROOT / "sparse/0/points3D.bin").read_bytes(), ids)
            for i, detail in enumerate(metadata["images"]):
                pose = poses[detail["name"]]
                observations = pose["observations"]
                observed = [o for o in observations if int(o["id"]) in points][:1000]
                self.assertGreater(len(observed), 50)
                xyz = np.array([points[int(o["id"])][0] for o in observed])
                camera_xyz = xyz @ data["viewmats"][i, :3, :3].T + data["viewmats"][i, :3, 3]
                self.assertGreater(np.mean(camera_xyz[:, 2] > 0), .99)
                projected = camera_xyz @ data["Ks"][i].T
                projected = projected[:, :2] / projected[:, 2:]
                camera = cameras[pose["camera_id"]]
                K, d = intrinsics(camera, camera[1], camera[2])
                xy = np.array([o["xy"] for o in observed])
                if d.any():
                    xy = cv2.undistortPoints(xy[:, None], K, d, P=K)[:, 0]
                xy *= np.array(detail["output_size"]) / np.array(camera[1:3])
                self.assertLess(np.median(np.linalg.norm(projected - xy, axis=1)), 2)
                R = data["viewmats"][i, :3, :3]
                np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(np.linalg.det(R), 1, places=5)


if __name__ == "__main__":
    unittest.main()
