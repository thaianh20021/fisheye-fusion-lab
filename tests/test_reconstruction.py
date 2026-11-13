import io
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from fisheye_fusion import reconstruction as runner


class FakeProcess:
    def __init__(self, data=b"", code=0, blocked=False):
        self.stdout = io.BytesIO(data)
        self.code = code
        self.returncode = None
        self.done = threading.Event()
        if not blocked:
            self.done.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = self.code
        return self.code

    def terminate(self):
        self.code = -15
        self.done.set()

    def kill(self):
        self.code = -9
        self.done.set()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    for relative in ("repo/run_inference.py", "repo/instant_nurec/__init__.py",
                     "repo/.venv/Scripts/python.exe", "checkpoint.pth", "clip.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sequence_id":"test"}' if relative == "clip.json" else "fixture")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=runner.RESERVE))
    config = dict(repo="repo", python="repo/.venv/Scripts/python.exe",
                  checkpoint="checkpoint.pth", ncore_path="clip.json")
    calls = []
    runtime = dict(python=[3, 11, 9], cuda=True, torch="test", ninja=True, render=True)
    options = {"ply": True, "code": 0, "blocked": False}

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == "git":
            if "--show-toplevel" in argv:
                return FakeProcess(str(tmp_path / "repo").encode())
            if "HEAD" in argv:
                return FakeProcess(options.get("revision", runner.REVISION).encode())
            return FakeProcess(options.get("dirty", "").encode())
        if "-c" in argv:
            return FakeProcess(json.dumps(runtime).encode())
        output = Path(argv[argv.index("--output-dir") + 1])
        if options["ply"]:
            (output / "scene.ply").write_bytes(b"ply\n")
        (output / "ignored.txt").write_text("not exposed")
        (output / "empty.ply").touch()
        return FakeProcess(b"x" * (runner.LOG_LIMIT + 100), options["code"], options["blocked"])

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    return runner.ReconstructionManager(tmp_path), config, calls, runtime, options


def finished(manager, job_id):
    for _ in range(200):
        status = manager.status(job_id)
        if status["state"] not in ("running", "cancelling"):
            return status
        time.sleep(.005)
    pytest.fail("job did not finish")


def test_success_contract_and_offline_argv(setup):
    manager, config, calls, _, _ = setup
    report = manager.preflight(config)
    assert report["ok"] and report["revision"] == report["expected_revision"] == runner.REVISION
    assert report["dirty"] is False
    job = manager.start({**config, "render_preview": True, "include_dynamic": True})
    result = finished(manager, job["job_id"])
    assert result["state"] == "succeeded" and result["returncode"] == 0
    assert len(result["log"]) == runner.LOG_LIMIT
    assert result["artifacts"] == ["scene.ply"]
    assert manager.artifact(job["job_id"], "scene.ply").read_bytes() == b"ply\n"
    argv, kwargs = calls[-1]
    assert argv[1] == "-u" and argv[2].endswith("run_inference.py")
    assert "--merge" in argv and "--render-preview" in argv and "--include-dynamic" in argv
    assert argv[argv.index("--n-gaussians") + 1] == "200000"
    assert argv[argv.index("--max-chunks") + 1] == "1"
    assert kwargs["shell"] is False and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["env"]["MAX_JOBS"] == "2"
    assert Path(kwargs["env"]["INSTANT_NUREC_FULL_PT"]).is_absolute()
    assert manager.cancel(job["job_id"])["state"] == "succeeded"


@pytest.mark.parametrize("patch", [
    {"max_chunks": 0}, {"max_chunks": 3}, {"max_chunks": True}, {"n_gaussians": 99999},
    {"n_gaussians": 2000001}, {"render_preview": "true"}, {"include_dynamic": 1},
    {"model": "arbitrary"}, {"env": {}}, {"command": "echo"},
    {"repo": "../outside"}, {"python": "repo/run_inference.py"},
    {"checkpoint": "clip.json"}, {"ncore_path": "checkpoint.pth"}, {"python": "/bin/python"},
    {"model": "pq-front", "include_dynamic": True},
])
def test_invalid_config_does_not_spawn(setup, patch):
    manager, config, calls, _, _ = setup
    assert not manager.preflight({**config, **patch})["ok"]
    assert not calls


def test_reject_canonical_manifest(setup):
    manager, config, calls, _, _ = setup
    (manager.workspace / "clip.json").write_text('{"cameras": [], "frames": [], "fusion": {}}')
    assert not manager.preflight(config)["ok"]
    assert not calls


@pytest.mark.parametrize("change", [{"python": [3, 12, 0]}, {"cuda": False}, {"ninja": False}, {"render": False}])
def test_runtime_requirements(setup, change):
    manager, config, _, runtime, _ = setup
    runtime.update(change)
    assert not manager.preflight({**config, "render_preview": True})["ok"]


def test_disk_reserve(setup, monkeypatch):
    manager, config, calls, _, _ = setup
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=runner.RESERVE - 1))
    assert not manager.preflight(config)["ok"]
    assert not calls


@pytest.mark.parametrize("code,ply", [(1, True), (0, False)])
def test_failure_requires_exit_zero_and_ply(setup, code, ply):
    manager, config, _, _, options = setup
    options.update(code=code, ply=ply)
    job = manager.start(config)
    assert finished(manager, job["job_id"])["state"] == "failed"


def test_cancel_and_single_active_job(setup):
    manager, config, _, _, options = setup
    options["blocked"] = True
    job = manager.start(config)
    with pytest.raises(ValueError, match="already active"):
        manager.start(config)
    manager.cancel(job["job_id"])
    assert finished(manager, job["job_id"])["state"] == "cancelled"
    options["blocked"] = False
    assert finished(manager, manager.start(config)["job_id"])["state"] == "succeeded"


def test_artifact_scope_and_unknown_job(setup):
    manager, config, _, _, _ = setup
    job = manager.start(config)
    finished(manager, job["job_id"])
    for relative in ("../checkpoint.pth", "/scene.ply", "ignored.txt", "empty.ply"):
        with pytest.raises(ValueError):
            manager.artifact(job["job_id"], relative)
    with pytest.raises(ValueError, match="unknown"):
        manager.status("../outside")


def test_artifact_symlink_replacement(setup):
    manager, config, _, _, _ = setup
    job = manager.start(config)
    finished(manager, job["job_id"])
    path = manager.artifact(job["job_id"], "scene.ply")
    path.unlink()
    try:
        path.symlink_to(manager.workspace / "checkpoint.pth")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        manager.artifact(job["job_id"], "scene.ply")


def test_preflight_timeout(setup, monkeypatch):
    manager, config, _, _, _ = setup
    process = FakeProcess()
    original_wait = process.wait

    def wait(timeout=None):
        if timeout == 30:
            raise subprocess.TimeoutExpired("probe", 30)
        return original_wait(timeout)

    process.wait = wait
    original_popen = runner.subprocess.Popen
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda argv, **kw: original_popen(argv, **kw) if argv[0] == "git" else process)
    report = manager.preflight(config)
    assert not report["ok"] and "timed out" in report["errors"][0]
    assert process.returncode == -15


@pytest.mark.parametrize("name", ["python", "python3", "python3.11"])
def test_linux_venv_symlink_preserves_lexical_path(setup, tmp_path_factory, name):
    manager, config, calls, _, _ = setup
    outside = tmp_path_factory.mktemp("system-python") / "python3.11"
    outside.write_text("trusted interpreter fixture")
    python = manager.workspace / "repo" / ".venv" / "bin" / name
    python.parent.mkdir()
    try:
        python.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    config["python"] = python.relative_to(manager.workspace).as_posix()
    job = manager.start(config)
    assert finished(manager, job["job_id"])["state"] == "succeeded"
    python_calls = [(argv, kwargs) for argv, kwargs in calls if argv[0] != "git"]
    assert len(python_calls) == 2
    assert all(argv[0] == str(python) for argv, _ in python_calls)
    assert python.resolve() == outside
    assert all(kwargs["env"]["PATH"].split(runner.os.pathsep)[0] == str(python.parent)
               for _, kwargs in python_calls)


def test_python_parent_symlink_rejected(setup, tmp_path_factory):
    manager, config, calls, _, _ = setup
    outside = tmp_path_factory.mktemp("outside-bin")
    (outside / "python").write_text("fixture")
    link = manager.workspace / "repo/.venv/bin"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert not manager.preflight({**config, "python": "repo/.venv/bin/python"})["ok"]
    assert not calls


@pytest.mark.parametrize("patch", [
    {"revision": "0" * 40}, {"dirty": " M run_inference.py"},
])
def test_revision_and_dirty_gate(setup, patch):
    manager, config, calls, _, options = setup
    options.update(patch)
    report = manager.preflight(config)
    assert not report["ok"]
    assert report["revision"] == options.get("revision", runner.REVISION)
    assert report["dirty"] == bool(options.get("dirty"))
    assert all(argv[0] == "git" for argv, _ in calls)
    with pytest.raises(ValueError):
        manager.start(config)


def test_shutdown_stops_job_and_prevents_restart(setup):
    manager, config, _, _, options = setup
    options["blocked"] = True
    job = manager.start(config)
    manager.shutdown()
    result = manager.status(job["job_id"])
    assert result["state"] == "cancelled"
    assert not manager._jobs[job["job_id"]]["thread"].is_alive()
    manager.shutdown()
    with pytest.raises(ValueError, match="shut down"):
        manager.start(config)


def test_tail_uses_read1():
    class Stream(io.BytesIO):
        def read(self, *args):
            raise AssertionError("buffered read delays live logs")

        def read1(self, size):
            return super().read(3)

    tail = bytearray()
    runner._tail(Stream(b"live logs"), tail)
    assert tail == b"live logs"
