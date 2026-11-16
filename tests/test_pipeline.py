"""KITTI-360 -> NCore -> NuRec -> eval pipeline: discovery, validation and the read-only outputs route."""
import json
from pathlib import Path

import pytest

from fisheye_fusion.pipeline import Discovery, PipelineManager


@pytest.fixture
def ws(tmp_path):
    """A workspace whose NuRec venv python is a symlink to an interpreter outside it (as uv makes)."""
    outside = tmp_path / "uv-python" / "python3.11"
    outside.parent.mkdir()
    outside.write_text("")
    root = tmp_path / "ws"
    venv_bin = root / "backends" / "instant-nurec" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(outside)
    (root / "backends" / "instant-nurec" / "run_inference.py").write_text("")
    ckpt = root / "backends" / "instant-nurec" / "checkpoints" / "pth"
    ckpt.mkdir(parents=True)
    (ckpt / "instant_nurec_pa_multiview_1.1.0.pth").write_text("")
    data = root / "kitti360"
    (data / "data_2d_raw").mkdir(parents=True)
    (data / "data_poses.zip").write_text("")
    for cam in ("image_00", "image_02"):
        (data / "data_2d_raw" / f"2013_05_28_drive_0003_sync_{cam}.tar").write_text("")
    return root


def test_report_handles_symlinked_venv_python(ws):
    report = Discovery(ws).report()
    assert report["nurec"]["ok"]
    assert report["nurec"]["python"] == "backends/instant-nurec/.venv/bin/python"
    assert report["nurec"]["checkpoints"] == {
        "pa-multiview": "backends/instant-nurec/checkpoints/pth/instant_nurec_pa_multiview_1.1.0.pth"}
    (seq,) = report["datasets"][0]["sequences"].items()
    assert seq == ("2013_05_28_drive_0003_sync", {"cameras": ["image_00", "image_02"], "lidar": False})


def test_evaluations_are_listed_with_their_inputs(ws):
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"camera_front_left_00": {"psnr_mean": 16.0, "views": 3}}))
    (run / "job.json").write_text(json.dumps({"ply": "a.ply", "clip": "c.json"}))
    (run / "camera_front_left_00_0.png").write_bytes(b"png")
    (ev,) = Discovery(ws).evaluations()
    assert ev["path"] == "outputs/eval/abc" and ev["ply"] == "a.ply" and ev["clip"] == "c.json"
    assert ev["images"] == ["camera_front_left_00_0.png"]


@pytest.mark.parametrize("bad", ["../secret.png", "/etc/passwd", "eval\\x.png", "c:x.png", "eval/abc/notes.txt",
                                 "eval/abc/missing.png"])
def test_output_file_rejects_escapes_and_other_types(ws, bad):
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "notes.txt").write_text("x")
    (ws / "secret.png").write_bytes(b"x")
    with pytest.raises(ValueError):
        Discovery(ws).output_file(bad)


def test_output_file_serves_results(ws):
    run = ws / "outputs" / "reconstruction" / "r" / "ply"
    run.mkdir(parents=True)
    (run / "scene.ply").write_bytes(b"ply")
    assert Discovery(ws).output_file("reconstruction/r/ply/scene.ply") == (run / "scene.ply").resolve()


def test_output_file_rejects_symlink_out_of_outputs(ws, tmp_path):
    (ws / "outputs").mkdir()
    (tmp_path / "elsewhere.png").write_bytes(b"x")
    (ws / "outputs" / "link.png").symlink_to(tmp_path / "elsewhere.png")
    with pytest.raises(ValueError):
        Discovery(ws).output_file("link.png")


def _captured_convert(ws, monkeypatch, **cfg):
    seen = {}
    manager = PipelineManager(ws)
    monkeypatch.setattr(manager, "_start", lambda kind, out, steps, env, info: seen.update(steps=steps, info=info) or {})
    manager.convert({"dataset": "kitti360", "sequence": "2013_05_28_drive_0003_sync", **cfg})
    return seen


def test_convert_passes_fisheye_model_and_defaults_to_rectify120(ws, monkeypatch):
    seen = _captured_convert(ws, monkeypatch, frames=40)
    argv = seen["steps"][0][1]
    assert argv[argv.index("--fisheye-model") + 1] == "rectify120"
    assert seen["info"]["fisheye_model"] == "rectify120"
    assert [label for label, _ in seen["steps"]] == ["convert", "projection check"]
    seen = _captured_convert(ws, monkeypatch, frames=40, fisheye_model="opencv", lidar=False, name="kb")
    argv = seen["steps"][0][1]
    assert argv[argv.index("--fisheye-model") + 1] == "opencv" and "--no-lidar" in argv
    assert len(seen["steps"]) == 1


@pytest.mark.parametrize("cfg", [{"fisheye_model": "pinhole"}, {"sequence": "drive_3"}, {"frames": 1},
                                 {"cameras": ["image_09"]}, {"lidar": "yes"}, {"name": "../x"}, {"unknown": 1}])
def test_convert_rejects_bad_config_before_creating_output(ws, monkeypatch, cfg):
    manager = PipelineManager(ws)
    monkeypatch.setattr(manager, "_start", lambda *a: pytest.fail("must not start"))
    with pytest.raises(ValueError):
        manager.convert({"dataset": "kitti360", "sequence": "2013_05_28_drive_0003_sync", **cfg})
    assert not (ws / "outputs" / "ncore").exists() or not any((ws / "outputs" / "ncore").iterdir())


def test_outputs_route(ws, monkeypatch):
    from fastapi.testclient import TestClient

    import fisheye_fusion.server as server

    monkeypatch.setattr(server, "WORKSPACE", Path(ws).resolve())
    run = ws / "outputs" / "eval" / "abc"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("{}")
    with TestClient(server.create_app()) as client:
        assert client.get("/api/outputs/eval/abc/metrics.json").status_code == 200
        assert client.get("/api/outputs/eval/abc/nope.png").status_code == 400
        setup = client.get("/api/setup").json()
        assert setup["nurec"]["ok"] and "evaluations" in setup


# ------------------------------------------------------------------ CARLA bridge jobs
def _captured_carla(ws, monkeypatch, **cfg):
    import fisheye_fusion.pipeline as pl

    seen = {}
    manager = PipelineManager(ws)
    monkeypatch.setattr(type(manager.discovery), "gsplat_python", property(lambda self: ws / "gsplat-python"))
    monkeypatch.setattr(manager, "_start", lambda kind, out, steps, env, info: seen.update(kind=kind, out=out, steps=steps, info=info) or {})
    manager.carla({"ply": "scene/scene.ply", "clip": "clip/clip.json", **cfg})
    return seen


@pytest.fixture
def carla_ws(ws):
    for f in ("scene/scene.ply", "clip/clip.json", "gsplat-python"):
        (ws / f).parent.mkdir(parents=True, exist_ok=True)
        (ws / f).write_text("x")
    return ws


def test_carla_job_without_server_renders_neural_only(carla_ws, monkeypatch):
    seen = _captured_carla(carla_ws, monkeypatch, use_carla=False, duration=5, lateral=-2.5)
    labels = [label for label, _ in seen["steps"]]
    assert seen["kind"] == "carla" and labels == ["export trajectory", "render"]
    bridge = seen["steps"][1][1]
    assert "--carla" not in bridge and "--trainer-config" not in bridge
    assert bridge[bridge.index("--lateral") + 1] == "-2.5" and bridge[bridge.index("--duration") + 1] == "5.0"
    assert json.loads((seen["out"] / "job.json").read_text())["use_carla"] is False


def test_carla_job_needs_a_running_server(carla_ws, monkeypatch):
    import fisheye_fusion.pipeline as pl

    monkeypatch.setattr(pl, "_port_open", lambda port, host="127.0.0.1": False)
    with pytest.raises(ValueError, match="no CARLA server"):
        _captured_carla(carla_ws, monkeypatch, use_carla=True)
    monkeypatch.setattr(pl, "_port_open", lambda port, host="127.0.0.1": True)
    monkeypatch.setattr(pl.CarlaServer, "client_ok", lambda self: True)
    bridge = _captured_carla(carla_ws, monkeypatch, use_carla=True, npc=5)["steps"][1][1]
    assert bridge[bridge.index("--carla") + 1] == "127.0.0.1:2000" and bridge[bridge.index("--npc") + 1] == "5"


def test_carla_job_aligns_3dgut_scenes(carla_ws, monkeypatch):
    (carla_ws / "scene" / "config.json").write_text(json.dumps({"frame_ids": [167, 172]}))
    seen = _captured_carla(carla_ws, monkeypatch, use_carla=False)
    bridge = seen["steps"][1][1]
    assert bridge[bridge.index("--trainer-config") + 1].endswith("scene/config.json")
    assert bridge[bridge.index("--kitti-data") + 1].endswith("kitti360")
    assert seen["info"]["kind"] == "3dgut"


@pytest.mark.parametrize("cfg", [{"npc": 9}, {"npc": True}, {"fov": 10}, {"lateral": "1"}, {"duration": -1},
                                 {"use_carla": "yes"}, {"name": "../x"}, {"cmd": "x"}])
def test_carla_job_rejects_bad_config(carla_ws, monkeypatch, cfg):
    with pytest.raises(ValueError):
        _captured_carla(carla_ws, monkeypatch, **{"use_carla": False, **cfg})


def test_discovery_lists_3dgut_runs_and_carla_videos(ws):
    run = ws / "fflab-runs" / "r1"
    run.mkdir(parents=True)
    (run / "gaussians.ply").write_bytes(b"ply")
    (run / "config.json").write_text(json.dumps({"frame_ids": [1, 2, 3]}))
    (ws / "fflab-runs" / "no-config").mkdir()
    (ws / "fflab-runs" / "no-config" / "gaussians.ply").write_bytes(b"ply")
    vid = ws / "outputs" / "carla" / "v1"
    vid.mkdir(parents=True)
    (vid / "composite.mp4").write_bytes(b"mp4")
    (vid / "job.json").write_text(json.dumps({"use_carla": True, "npc": 2}))
    d = Discovery(ws)
    recs = [r for r in d.reconstructions() if r["kind"] == "3dgut"]
    assert [(r["job_id"], r["frames"]) for r in recs] == [("r1", 3)]
    (v,) = d.carla_runs()
    assert v["path"] == "outputs/carla/v1" and v["video"] == "composite.mp4" and v["npc"] == 2
    assert d.output_file("carla/v1/composite.mp4").name == "composite.mp4"


def test_landed_gpu_picks_the_gpu_that_grew(monkeypatch):
    import fisheye_fusion.pipeline as pl

    monkeypatch.setattr(pl, "_gpus", lambda: [{"index": 0, "memory_used_mib": 23700}, {"index": 1, "memory_used_mib": 9000},
                                              {"index": 2, "memory_used_mib": 23650}])
    assert pl._landed_gpu({0: 23600, 1: 3000, 2: 23600}) == 1
    assert pl._landed_gpu({0: 23600, 1: 8900, 2: 23600}) is None  # nothing grew by 400 MiB


def test_carla_server_refuses_when_not_installed(ws, monkeypatch, tmp_path):
    import fisheye_fusion.pipeline as pl

    monkeypatch.setenv("CARLA_ROOT", str(tmp_path / "no-carla"))
    monkeypatch.setattr(pl, "_port_open", lambda port, host="127.0.0.1": False)
    server = PipelineManager(ws).carla_server
    with pytest.raises(ValueError, match="CARLA not found"):
        server.start(None)
    with pytest.raises(ValueError, match="unknown GPU"):
        server.start(99)
