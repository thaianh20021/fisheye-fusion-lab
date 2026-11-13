import io
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fisheye_fusion.training import TrainingManager


class Process:
    def __init__(self, *args, **kwargs):
        self.stdout = io.BytesIO(b'{"step": 3, "l1": 0.2}\n')
        self.pid = 123
    def wait(self, timeout=None):
        return 1
    def poll(self):
        return 1


def test_job_failure_progress_and_persistence(tmp_path):
    manager = TrainingManager(tmp_path)
    with patch("fisheye_fusion.training.prepare_dataset", return_value={}), \
         patch.object(manager, "_command", return_value=["python", "trusted.py"]), \
         patch("fisheye_fusion.training.subprocess.Popen", Process):
        state = manager.start("input.npz", steps=10)
        manager.jobs[state["job_id"]]["thread"].join(5)
    state = manager.status(state["job_id"])
    assert state["state"] == "failed" and state["step"] == 3
    assert TrainingManager(tmp_path).status(state["job_id"])["state"] == "failed"
    assert manager.cancel(state["job_id"])["state"] == "failed"
    manager.shutdown()
    with pytest.raises(ValueError, match="shut down"):
        manager.start("input.npz")


def test_validation_and_busy(tmp_path):
    manager = TrainingManager(tmp_path)
    for steps in (0, 10001, True, 1.5):
        with pytest.raises(ValueError):
            manager.start("input.npz", steps=steps)
    manager.jobs["fake"] = {"state": "running"}
    with pytest.raises(ValueError, match="already active"):
        manager.start("input.npz")
    for value in ("../escape", "x", "/" + "a" * 32):
        with pytest.raises(ValueError):
            manager.status(value)


def test_artifacts_and_restart(tmp_path):
    manager = TrainingManager(tmp_path)
    job_id = "a" * 32
    folder = manager._directory(job_id, create=True)
    (folder / "result").mkdir()
    (folder / "result/after.png").write_bytes(b"test")
    status = {"state": "running", "artifacts": ["after.png"]}
    (folder / "status.json").write_text(json.dumps(status))
    assert manager.status(job_id)["state"] == "interrupted"
    assert manager.artifact(job_id, "after.png").read_bytes() == b"test"
    with pytest.raises(ValueError):
        manager.artifact(job_id, "../status.json")


def test_api_validation(tmp_path, monkeypatch):
    from fisheye_fusion import server
    monkeypatch.setattr(server, "WORKSPACE", tmp_path)
    with TestClient(server.create_app()) as client:
        assert client.post("/api/training", json={"dataset": "a", "steps": 0}).status_code == 422
        assert client.post("/api/training/inspect", json={"dataset": "../a"}).status_code == 400
        assert client.get("/api/training/invalid").status_code == 400
        assert client.post("/api/training", json={"dataset": "a"},
                           headers={"origin": "https://evil.invalid"}).status_code == 403
