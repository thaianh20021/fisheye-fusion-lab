import json

import pytest
from fastapi.testclient import TestClient

from fisheye_fusion import server


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    output = tmp_path / "outputs" / "real-gs-run"
    with TestClient(server.create_app()) as client:
        yield client, output


@pytest.fixture
def results(experiment):
    client, output = experiment
    output.mkdir(parents=True)
    metrics = {
        "backend": "fixture static 3DGS", "dataset": "fixture.npz", "gpu": "fixture GPU",
        "steps": 2, "gaussians": 100, "resolution": [32, 24],
        "heldout_psnr_before": {"1": 10.25}, "heldout_psnr_after": {"1": 12.5},
        "seconds": 1.25, "peak_vram_mib": 64.5,
        "trace": [{"step": 2, "l1": 0.125, "seconds": 1.25}],
        "limitations": "Test fixture, not a trained result.",
    }
    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return client, output, metrics


def test_experiment_absent(experiment):
    client, _ = experiment
    assert client.get("/api/experiment").status_code in (400, 404)


def test_experiment_metrics(results):
    client, _, metrics = results
    response = client.get("/api/experiment")
    assert response.status_code == 200
    payload = response.json()
    actual = payload.get("metrics", payload)
    for key, expected in metrics.items():
        assert actual[key] == expected


@pytest.mark.parametrize("name", [
    "metrics.json", "scene.ply", "reference.png", "before.png", "after.png",
    *[f"novel-{index:02d}.png" for index in range(9)],
])
def test_experiment_artifact_bytes(results, name):
    client, output, _ = results
    path = output / name
    if name != "metrics.json":
        path.write_bytes(b"\x00\xfffixture artifact\n" + name.encode("ascii"))
    response = client.get(f"/api/experiment/artifacts/{name}")
    assert response.status_code == 200
    assert response.content == path.read_bytes()


@pytest.mark.parametrize("name", ["private.txt", "novel-09.png", "scene.ply.bak"])
def test_experiment_artifact_allowlist(results, name):
    client, output, _ = results
    (output / name).write_bytes(b"must not be exposed")
    assert client.get(f"/api/experiment/artifacts/{name}").status_code in (400, 404)


@pytest.mark.parametrize("path", [
    "%2e%2e%2fsecret.txt", "%2e%2e%5csecret.txt",
    "%2e%2e%2fscene.ply", "%2e%2e%5cscene.ply",
    "%2fscene.ply", "nested%2fscene.ply", "%252e%252e%252fscene.ply",
])
def test_experiment_artifact_path_escape(results, path):
    client, output, _ = results
    (output.parent / "secret.txt").write_bytes(b"outside")
    (output.parent / "scene.ply").write_bytes(b"outside model")
    (output / "nested").mkdir()
    (output / "nested" / "scene.ply").write_bytes(b"not allowlisted")
    assert client.get(f"/api/experiment/artifacts/{path}").status_code in (400, 404)


def test_experiment_missing_artifact(results):
    client, _, _ = results
    assert client.get("/api/experiment/artifacts/after.png").status_code in (400, 404)


def test_experiment_artifact_symlink_escape(results):
    client, output, _ = results
    target = output.parent / "secret.ply"
    target.write_bytes(b"outside model")
    try:
        (output / "scene.ply").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert client.get("/api/experiment/artifacts/scene.ply").status_code in (400, 404)


@pytest.mark.parametrize("endpoint", ["/api/experiment", "/api/experiment/artifacts/metrics.json"])
@pytest.mark.parametrize("run", ["", "unknown", "GEER", "../real-gs-run", r"..\geer-run", "/tmp", "real-gs-run"])
def test_experiment_invalid_run(results, endpoint, run):
    client, _, _ = results
    response = client.get(endpoint, params={"run": run})
    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown experiment run"


@pytest.mark.parametrize("name", [
    "metrics.json", "scene.ply", "reference.png", "before.png", "after.png",
    *[f"novel-{index:02d}.png" for index in range(9)],
])
def test_experiment_run_isolation(results, name):
    client, baseline, metrics = results
    geer = baseline.parent / "geer-run"
    geer.mkdir()
    geer_metrics = {**metrics, "backend": "fixture GEER", "steps": 3}
    (geer / "metrics.json").write_text(json.dumps(geer_metrics), encoding="utf-8")
    if name != "metrics.json":
        (baseline / name).write_bytes(b"baseline fixture")
        (geer / name).write_bytes(b"GEER fixture")
    assert client.get("/api/experiment").json() == metrics
    for run, output, expected in [("baseline", baseline, metrics), ("geer", geer, geer_metrics)]:
        response = client.get("/api/experiment", params={"run": run})
        assert response.status_code == 200
        assert response.json() == expected
        response = client.get(f"/api/experiment/artifacts/{name}", params={"run": run})
        assert response.status_code == 200
        assert response.content == (output / name).read_bytes()
    assert client.get(f"/api/experiment/artifacts/{name}").content == (baseline / name).read_bytes()


@pytest.mark.parametrize("missing_run", ["baseline", "geer"])
def test_experiment_missing_run_does_not_fall_back(results, missing_run):
    client, baseline, _ = results
    if missing_run == "baseline":
        baseline.rename(baseline.parent / "geer-run")
    for endpoint in ["/api/experiment", "/api/experiment/artifacts/metrics.json"]:
        assert client.get(endpoint, params={"run": missing_run}).status_code == 400


@pytest.mark.parametrize("name", ["private.txt", "novel-09.png", "nested%2fscene.ply", "%2e%2e%2freal-gs-run%2fmetrics.json"])
def test_geer_artifact_allowlist(results, name):
    client, _, _ = results
    assert client.get(f"/api/experiment/artifacts/{name}", params={"run": "geer"}).status_code == 400


def test_geer_artifact_cannot_link_to_baseline(results):
    client, baseline, _ = results
    geer = baseline.parent / "geer-run"
    geer.mkdir()
    try:
        (geer / "metrics.json").symlink_to(baseline / "metrics.json")
    except OSError:
        pytest.skip("symlinks unavailable")
    for endpoint in ["/api/experiment", "/api/experiment/artifacts/metrics.json"]:
        assert client.get(endpoint, params={"run": "geer"}).status_code == 400
