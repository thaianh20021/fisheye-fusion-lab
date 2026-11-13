import io
import json
import zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from fisheye_fusion import server
from fisheye_fusion.demo import make_demo
from fisheye_fusion.manifest import load_manifest
from fisheye_fusion.fusion import fuse_images


def test_rotation_changes_panorama_coverage(tmp_path):
    path = make_demo(tmp_path)
    manifest = load_manifest(path)
    counts = []
    for selected in [["front"], ["front", "rear"], ["front", "rear", "left", "right"]]:
        manifest.fusion.camera_ids = selected
        manifest.fusion.resolution = (256, 128)
        image, mask = fuse_images(manifest, tmp_path)
        assert image[mask].mean() > 15
        assert not image[~mask].any()
        counts.append(mask.sum())
    assert counts[0] < counts[1] < counts[2]


def test_web_demo_render_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    c = TestClient(server.create_app())
    path = c.post("/api/demo").json()["manifest"]
    body = {"manifest": path, "width": 512}
    info = c.post("/api/inspect", json=body)
    assert info.status_code == 200
    assert len(info.json()["images"]) == 4
    result = c.post("/api/render", json=body)
    assert result.status_code == 200
    assert .2 < result.json()["coverage"] < 1
    bad = c.post("/api/render", json={**body, "camera_ids": ["front", "left", "rear"]})
    assert bad.status_code == 400
    for platform in ["colab", "kaggle"]:
        response = c.post(f"/api/bundle/{platform}", json=body)
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            assert "manifest.json" in z.namelist()
            assert len(json.loads(z.read(f"run_{platform}.ipynb"))["cells"]) == 4
            assert not json.loads(z.read("job.json"))["neural_training"]
            out = tmp_path / platform
            z.extractall(out)
        packaged = load_manifest(out / "manifest.json")
        image, _ = fuse_images(packaged, out)
        assert image.shape == (256, 512, 3)


def test_api_rejects_remote_origin_and_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    c = TestClient(server.create_app())
    assert c.post("/api/demo", headers={"origin": "https://evil.example"}).status_code == 403
    assert c.post("/api/inspect", json={"manifest": "../outside.json"}).status_code == 400
    assert c.post("/api/demo", headers={"host": "evil.example"}).status_code == 400


def test_invalid_rotation_and_intrinsics(tmp_path):
    from fisheye_fusion.models import ProjectManifest
    path = make_demo(tmp_path)
    data = json.loads(path.read_text())
    data["cameras"][0]["T_rig_from_camera"][0][0] = 2
    with pytest.raises(ValueError, match="rotation"):
        ProjectManifest.model_validate(data)
