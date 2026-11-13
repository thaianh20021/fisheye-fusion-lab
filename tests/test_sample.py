import json

import pytest

from fisheye_fusion.demo import make_demo
from fisheye_fusion.manifest import load_manifest
from fisheye_fusion.sample import sample_dataset


def test_sample_writes_small_roundtrip(tmp_path):
    source = make_demo(tmp_path / "source")
    result = sample_dataset(source, tmp_path / "sample", max_frames=1, max_bytes=2_000_000, write=True)
    assert result["written"]
    sampled = load_manifest(tmp_path / "sample" / "manifest.json")
    assert len(sampled.frames) == 1
    assert all((tmp_path / "sample" / path).is_file() for path in sampled.frames[0].images.values())


def test_sample_dry_run_does_not_write(tmp_path):
    source = make_demo(tmp_path / "source")
    output = tmp_path / "sample"
    result = sample_dataset(source, output, max_frames=1, write=False)
    assert not result["written"]
    assert not output.exists()


def test_sample_rejects_budget_and_existing_output(tmp_path):
    source = make_demo(tmp_path / "source")
    with pytest.raises(ValueError, match="limit"):
        sample_dataset(source, tmp_path / "too-small", max_frames=1, max_bytes=1)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        sample_dataset(source, output)


def test_repeated_references_and_camera_subset(tmp_path):
    source = make_demo(tmp_path / "source")
    data = json.loads(source.read_text())
    second = dict(data["frames"][0])
    second["timestamp_ns"] += 1
    data["frames"].append(second)
    source.write_text(json.dumps(data))
    result = sample_dataset(source, tmp_path / "sample", camera_ids=["front"], write=True)
    manifest = load_manifest(result["manifest"])
    assert len(manifest.frames) == 2
    assert len(manifest.cameras) == 1
    assert manifest.fusion.camera_ids == ["front"]
    assert result["bytes"] == sum(p.stat().st_size for p in (tmp_path / "sample").rglob("*") if p.is_file())


def test_metadata_counts_and_insufficient_space(tmp_path, monkeypatch):
    from fisheye_fusion import sample
    from collections import namedtuple
    source = make_demo(tmp_path / "source")
    planned = sample_dataset(source, tmp_path / "sample")
    with pytest.raises(ValueError, match="manifest"):
        sample_dataset(source, tmp_path / "sample", max_bytes=planned["bytes"] - 1)
    monkeypatch.setattr(sample.shutil, "disk_usage", lambda _: namedtuple("Usage", "free")(0))
    with pytest.raises(ValueError, match="disk space"):
        sample_dataset(source, tmp_path / "sample", write=True)
    assert not (tmp_path / "sample").exists()


def test_api_sample_dry_run_write_and_containment(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from fisheye_fusion import server
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    client = TestClient(server.create_app())
    source = make_demo(tmp_path / "source")
    body = {"manifest": "source/manifest.json", "output": "samples/demo", "camera_ids": ["front"]}
    planned = client.post("/api/sample", json=body)
    assert planned.status_code == 200
    assert not (tmp_path / "samples").exists()
    written = client.post("/api/sample", json={**body, "write": True})
    assert written.status_code == 200
    assert written.json()["written"]
    assert client.post("/api/sample", json={**body, "write": True}).status_code == 400
    assert client.post("/api/sample", json={**body, "output": "../outside"}).status_code == 400
