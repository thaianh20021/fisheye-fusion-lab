"""KITTI-360 -> NCore -> Instant NuRec -> evaluation: environment discovery and background jobs.

The heavy lifting runs in subprocesses with the interpreter that has the right packages:
- conversion / projection check / view export: the Instant NuRec venv (it ships `nvidia-ncore`);
- PLY rendering for evaluation: the fflab venv (gsplat 1.5.3, `fflab setup`).
All paths given by clients are relative to the workspace, like the reconstruction runner.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .reconstruction import _stop, _tail

TOOLS = Path(__file__).resolve().parent.parent / "tools"
KITTI_CAMERAS = {
    "image_00": "camera_front_left_00",
    "image_01": "camera_front_right_01",
    "image_02": "camera_fisheye_left_02",
    "image_03": "camera_fisheye_right_03",
}
SEQUENCE = re.compile(r"2013_05_28_drive_\d{4}_sync")
MODEL_FILES = {"pa-front": "pa_front", "pa-multiview": "pa_multiview", "pq-front": "pq_road"}
ARTIFACTS = {".png", ".json", ".ply", ".mp4"}
# rectify120: what Instant NuRec takes; ftheta / opencv keep the full fisheye (3DGUT, gsplat)
FISHEYE_MODELS = ("rectify120", "ftheta", "opencv")


# ------------------------------------------------------------------ discovery
def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines() if out.returncode == 0 else []:
        i, name, used, total, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "memory_used_mib": int(used),
                     "memory_total_mib": int(total), "utilization": int(util)})
    return gpus


def _cuda_home() -> str | None:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    candidates = sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True)
    return str(candidates[0].parent.parent) if candidates else None


class Discovery:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def rel(self, path: Path) -> str:
        # abspath, not resolve: a venv's python is a symlink to an interpreter outside the workspace
        return Path(os.path.abspath(path)).relative_to(self.workspace).as_posix()

    @property
    def nurec_repo(self) -> Path:
        return self.workspace / "backends" / "instant-nurec"

    @property
    def nurec_python(self) -> Path:
        return self.nurec_repo / ".venv" / "bin" / "python"

    @property
    def gsplat_python(self) -> Path | None:
        venv = os.environ.get("FFLAB_VENV") or str(Path.home() / ".fflab" / "venv")
        python = Path(venv) / "bin" / "python"
        return python if python.is_file() else None

    def checkpoints(self) -> dict[str, str]:
        found = {}
        for model, stem in MODEL_FILES.items():
            hits = sorted((self.nurec_repo / "checkpoints" / "pth").glob(f"instant_nurec_{stem}_*.pth"))
            if hits:
                found[model] = self.rel(hits[-1])
        return found

    def datasets(self) -> list[dict]:
        """KITTI-360 archive roots (folders with data_poses.zip) up to two levels below the workspace."""
        roots = []
        for poses in sorted(self.workspace.glob("data_poses.zip")) + sorted(self.workspace.glob("*/data_poses.zip")) \
                + sorted(self.workspace.glob("*/*/data_poses.zip")):
            root = poses.parent
            if "backends" in root.relative_to(self.workspace).parts:
                continue
            sequences = {}
            for tar in sorted((root / "data_2d_raw").glob("*_image_0[0-3].tar")):
                seq, cam = tar.name[: -len("_image_00.tar")], tar.name[-len("image_00.tar"):-4]
                if SEQUENCE.fullmatch(seq):
                    sequences.setdefault(seq, {"cameras": [], "lidar": (root / "data_3d_raw" / f"{seq}.tar").is_file()})
                    sequences[seq]["cameras"].append(cam)
            has_meta = all((root / f"ImageSets-rosa-6class/kitti360_rosa_infos_{s}.pkl").is_file() for s in ("train", "val"))
            roots.append({"path": self.rel(root), "sequences": sequences, "calibration_pickles": has_meta})
        return roots

    def clips(self) -> list[dict]:
        clips = []
        for path in sorted((self.workspace / "outputs" / "ncore").glob("*/*.json")):
            try:
                meta = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("version") != "v4":
                continue
            cams = [c for s in meta.get("component_stores", []) for c in s.get("components", {}).get("cameras", {})]
            gm = meta.get("generic_meta_data", {})
            clips.append({"path": self.rel(path), "sequence_id": meta.get("sequence_id"), "cameras": cams,
                          "frames": len(gm.get("frame_ids", [])), "source": gm.get("source_sequence", "")})
        return clips

    def reconstructions(self) -> list[dict]:
        out = []
        root = self.workspace / "outputs" / "reconstruction"
        for ply in sorted(root.glob("*/*/ply/*/*.ply"), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append({"path": self.rel(ply), "job_id": ply.relative_to(root).parts[0], "kind": "nurec",
                        "size_mib": round(ply.stat().st_size / 2**20, 1)})
        # 3DGUT runs (fflab train): final gaussians.ply next to the config.json that lists its frames
        for ply in sorted((self.workspace / "fflab-runs").glob("*/gaussians.ply"), key=lambda p: p.stat().st_mtime, reverse=True):
            cfg = trainer_config(ply)
            if cfg is not None:
                out.append({"path": self.rel(ply), "job_id": ply.parent.name, "kind": "3dgut",
                            "frames": len(json.loads(cfg.read_text())["frame_ids"]),
                            "size_mib": round(ply.stat().st_size / 2**20, 1)})
        return out

    def evaluations(self) -> list[dict]:
        out = []
        for metrics in sorted((self.workspace / "outputs" / "eval").glob("*/metrics.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                scores = json.loads(metrics.read_text())
                info = json.loads((metrics.parent / "job.json").read_text()) if (metrics.parent / "job.json").is_file() else {}
            except (OSError, ValueError):
                continue
            out.append({"path": self.rel(metrics.parent), "metrics": scores, **info,
                        "images": sorted(p.name for p in metrics.parent.glob("*.png"))})
        return out

    def carla_runs(self) -> list[dict]:
        out = []
        for run in sorted((self.workspace / "outputs" / "carla").glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
            videos = [v for v in ("composite.mp4", "neural.mp4") if (run / v).is_file() and (run / v).stat().st_size]
            if not videos:
                continue
            try:
                info = json.loads((run / "job.json").read_text()) if (run / "job.json").is_file() else {}
            except (OSError, ValueError):
                info = {}
            out.append({"path": self.rel(run), "video": videos[0], **info,
                        "frames": sorted(p.name for p in (run / "frames").glob("*.png"))[:12]})
        return out

    def output_file(self, relative: str) -> Path:
        """A .png/.json/.ply/.mp4 under outputs/, for showing or downloading earlier results (read-only)."""
        if (not isinstance(relative, str) or "\\" in relative or ":" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            raise ValueError("invalid output path")
        root = (self.workspace / "outputs").resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {".png", ".json", ".ply", ".mp4"} or not path.is_file():
            raise ValueError("unknown output file")
        return path

    def report(self) -> dict:
        nurec_ok = (self.nurec_repo / "run_inference.py").is_file() and self.nurec_python.is_file()
        return {
            "workspace": str(self.workspace),
            "nurec": {"ok": nurec_ok, "repo": self.rel(self.nurec_repo) if self.nurec_repo.exists() else None,
                      "python": self.rel(self.nurec_python) if self.nurec_python.is_file() else None,
                      "checkpoints": self.checkpoints() if nurec_ok else {}},
            "gsplat": {"ok": self.gsplat_python is not None, "cuda_home": _cuda_home()},
            "gpus": _gpus(),
            "datasets": self.datasets(),
            "clips": self.clips(),
            "reconstructions": self.reconstructions(),
            "evaluations": self.evaluations(),
            "carla_runs": self.carla_runs(),
            "cameras": KITTI_CAMERAS,
        }


# ------------------------------------------------------------------ jobs
def trainer_config(ply: Path) -> Path | None:
    """config.json of the 3DGUT run a PLY came from (KITTI-360 world, needs aligning to a clip), if any."""
    cfg = Path(ply).parent / "config.json"
    try:
        return cfg if json.loads(cfg.read_text()).get("frame_ids") else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ CARLA server
CARLA_PORT = 2000
CARLA_STATE = Path.home() / ".fflab" / "carla.json"


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _gpu_memory() -> dict[int, int]:
    return {g["index"]: g["memory_used_mib"] for g in _gpus()}


def _landed_gpu(before: dict[int, int], min_mib: int = 400) -> int | None:
    """The GPU whose memory grew most since `before` (a Vulkan process does not show in nvidia-smi's
    process table here); None when no GPU grew by min_mib."""
    after = _gpu_memory()
    growth = {i: after.get(i, 0) - before.get(i, 0) for i in after}
    best = max(growth, key=growth.get, default=None)
    return best if best is not None and growth[best] >= min_mib else None


class CarlaServer:
    """One CARLA server on this machine, offscreen, on localhost:2000 (tools/carla_server.sh).

    CARLA picks its GPU by Vulkan adapter index, which differs from nvidia-smi's numbering. A start
    for GPU N tries adapters until the server lands on N and remembers the mapping in ~/.fflab/carla.json."""

    def __init__(self, workspace: Path, gsplat_python):
        self.workspace = workspace
        self._gsplat_python = gsplat_python
        self._lock = threading.RLock()
        self.proc = None
        self.state = "stopped"
        self.gpu = None
        self.message = ""
        self.tail = bytearray()
        self._client = None

    @property
    def root(self) -> Path:
        return Path(os.environ.get("CARLA_ROOT") or Path.home() / "carla")

    def installed(self) -> bool:
        return (self.root / "CarlaUE4" / "Binaries" / "Linux" / "CarlaUE4-Linux-Shipping").is_file()

    def client_ok(self) -> bool:
        if self._client is None:
            py = self._gsplat_python()
            self._client = bool(py) and subprocess.run(
                [str(py), "-c", "import importlib.util,sys; sys.exit(importlib.util.find_spec('carla') is None)"],
                capture_output=True, timeout=30).returncode == 0
        return self._client

    def _mapping(self) -> dict:
        try:
            return json.loads(CARLA_STATE.read_text()).get("adapter_of_gpu", {})
        except (OSError, ValueError):
            return {}

    def _remember(self, adapter: int, gpu: int):
        m = self._mapping()
        m[str(gpu)] = adapter
        CARLA_STATE.parent.mkdir(parents=True, exist_ok=True)
        CARLA_STATE.write_text(json.dumps({"adapter_of_gpu": m}, indent=2))

    def status(self) -> dict:
        with self._lock:
            alive = self.proc is not None and self.proc.poll() is None
            if self.state == "running" and not alive:
                self.state, self.message = "stopped", "server exited"
            return {"root": str(self.root), "installed": self.installed(), "client": self.client_ok(),
                    "port": CARLA_PORT, "port_open": _port_open(CARLA_PORT), "state": self.state,
                    "managed": alive, "gpu": self.gpu, "message": self.message,
                    "log": bytes(self.tail[-4000:]).decode("utf-8", errors="replace"),
                    "adapter_of_gpu": self._mapping()}

    def start(self, gpu) -> dict:
        if gpu is not None and (type(gpu) is not int or gpu not in {g["index"] for g in _gpus()}):
            raise ValueError("unknown GPU index")
        with self._lock:
            if self.state in ("starting", "running") or _port_open(CARLA_PORT):
                raise ValueError(f"a CARLA server is already running on port {CARLA_PORT}")
            if not self.installed():
                raise ValueError(f"CARLA not found at {self.root} (extract CARLA 0.9.16 there or set CARLA_ROOT)")
            self.state, self.gpu, self.message, self.tail = "starting", None, "", bytearray()
            threading.Thread(target=self._start, args=(gpu,), daemon=True).start()
        return self.status()

    def _launch(self, adapter: int):
        before = _gpu_memory()
        self.proc = subprocess.Popen(["bash", str(TOOLS / "carla_server.sh"), str(adapter), str(CARLA_PORT)],
                                     cwd=str(self.root), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        threading.Thread(target=_tail, args=(self.proc.stdout, self.tail), daemon=True).start()
        for _ in range(180):
            if _port_open(CARLA_PORT) or self.proc.poll() is not None:
                break
            time.sleep(1)
        if not _port_open(CARLA_PORT):
            return None
        for _ in range(15):  # its memory keeps growing for a few seconds after the port opens
            time.sleep(1)
            landed = _landed_gpu(before)
            if landed is not None:
                return landed
        return -1

    def _start(self, gpu):
        try:
            mapping = self._mapping()
            tried = []
            first = mapping.get(str(gpu)) if gpu is not None else None
            candidates = ([first] if first is not None else []) + [a for a in range(max(len(_gpus()), 1)) if a != first]
            for adapter in candidates:
                tried.append(adapter)
                landed = self._launch(adapter)
                if landed is None:
                    self._kill()
                    self.state, self.message = "failed", "CARLA exited before opening its port (see log)"
                    return
                if landed >= 0:
                    self._remember(adapter, landed)
                if gpu is None or landed == gpu or landed < 0:
                    with self._lock:
                        self.state, self.gpu = "running", (landed if landed >= 0 else None)
                        self.message = f"running on GPU {landed}" if landed >= 0 else "running (GPU not identified)"
                    return
                self._kill()  # landed on another GPU: try the next adapter
            self.state, self.message = "failed", f"no Vulkan adapter put CARLA on GPU {gpu} (tried {tried})"
        except Exception as error:  # noqa: BLE001 - reported to the client
            self._kill()
            self.state, self.message = "failed", str(error)

    def _kill(self):
        if self.proc is not None and self.proc.poll() is None:
            _stop(self.proc)
        for _ in range(20):
            if not _port_open(CARLA_PORT):
                break
            time.sleep(0.5)

    def stop(self) -> dict:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                if _port_open(CARLA_PORT):
                    raise ValueError("the CARLA server on this port was not started here; stop it where it was started")
                self.state = "stopped"
                return self.status()
            self.state = "stopping"
        self._kill()
        with self._lock:
            self.state, self.gpu, self.message = "stopped", None, "stopped"
        return self.status()


class PipelineManager:
    """Runs conversion and evaluation jobs; one job of each kind at a time."""

    def __init__(self, workspace: Path):
        self.discovery = Discovery(workspace)
        self.workspace = self.discovery.workspace
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._closed = False
        self.carla_server = CarlaServer(self.workspace, lambda: self.discovery.gsplat_python)

    # ---- validation helpers
    def _path(self, value, *, directory=False, suffix=None) -> Path:
        if (not isinstance(value, str) or not value or "\\" in value or ":" in value
                or Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("paths must be relative workspace paths using forward slashes")
        path = (self.workspace / value).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("path escapes workspace")
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"incorrect path type: {value}")
        if suffix and path.suffix.lower() != suffix:
            raise ValueError(f"path must end in {suffix}")
        return path

    def _env(self, gpu, extra_path=()):
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
        env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        if gpu is not None:
            if type(gpu) is not int or gpu < 0 or gpu not in {g["index"] for g in _gpus()}:
                raise ValueError("unknown GPU index")
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if extra_path:
            env["PATH"] = os.pathsep.join([*map(str, extra_path), env.get("PATH", "")])
        return env

    def _new_output(self, kind, name):
        root = self.workspace / "outputs" / kind
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            raise ValueError(f"output already exists: outputs/{kind}/{name}")
        path.mkdir()
        return path

    # ---- job types
    def convert(self, cfg: dict) -> dict:
        allowed = {"dataset", "sequence", "start", "frames", "cameras", "lidar", "name", "fisheye_model"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown convert fields")
        if not self.discovery.nurec_python.is_file():
            raise ValueError("Instant NuRec venv not found at backends/instant-nurec/.venv (it provides nvidia-ncore)")
        data = self._path(cfg.get("dataset", ""), directory=True)
        seq = cfg.get("sequence", "")
        if not SEQUENCE.fullmatch(str(seq)):
            raise ValueError("sequence must look like 2013_05_28_drive_0003_sync")
        start, frames = cfg.get("start", 0), cfg.get("frames", 40)
        if type(start) is not int or start < 0 or type(frames) is not int or not 2 <= frames <= 2000:
            raise ValueError("start must be >= 0 and frames in 2..2000")
        cams = cfg.get("cameras") or list(KITTI_CAMERAS)
        if not isinstance(cams, list) or not cams or set(cams) - set(KITTI_CAMERAS):
            raise ValueError(f"cameras must be a subset of {list(KITTI_CAMERAS)}")
        fisheye_model = cfg.get("fisheye_model", "rectify120")
        if fisheye_model not in FISHEYE_MODELS:
            raise ValueError(f"fisheye_model must be one of {FISHEYE_MODELS}")
        lidar = cfg.get("lidar", True)
        if type(lidar) is not bool:
            raise ValueError("lidar must be boolean")
        default = f"kitti360_{seq.split('_drive_')[1][:4]}_{start:06d}_{frames}f"
        name = cfg.get("name") or default
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise ValueError("name: letters, digits, _ . - only")
        out = self._new_output("ncore", name)
        py = str(self.discovery.nurec_python)
        argv = [py, "-u", str(TOOLS / "kitti360_to_ncore.py"), "--data", str(data), "--out", str(out),
                "--sequence", seq, "--start", str(start), "--frames", str(frames), "--cameras", ",".join(cams),
                "--sequence-id", name, "--fisheye-model", fisheye_model]
        if not lidar:
            argv.append("--no-lidar")
        steps = [("convert", argv)]
        if lidar:
            steps.append(("projection check", [py, "-u", str(TOOLS / "ncore_projection_check.py"),
                                               str(out / f"{name}.json"), "--out", str(out / "check"), "--frame", "0"]))
        return self._start("convert", out, steps, self._env(None),
                           {"clip": self.discovery.rel(out / f"{name}.json"), "fisheye_model": fisheye_model})

    def evaluate(self, cfg: dict) -> dict:
        allowed = {"ply", "clip", "every", "gpu"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown evaluate fields")
        ply = self._path(cfg.get("ply", ""), suffix=".ply")
        clip = self._path(cfg.get("clip", ""), suffix=".json")
        every = cfg.get("every", 4)
        if type(every) is not int or not 1 <= every <= 100:
            raise ValueError("every must be in 1..100")
        gsplat = self.discovery.gsplat_python
        if gsplat is None:
            raise ValueError("gsplat venv not found; run `fflab setup` (or set FFLAB_VENV)")
        out = self._new_output("eval", uuid.uuid4().hex[:12])
        (out / "job.json").write_text(json.dumps({"ply": self.discovery.rel(ply), "clip": self.discovery.rel(clip)}))
        cuda = _cuda_home()
        env = self._env(cfg.get("gpu"), [gsplat.parent] + ([Path(cuda) / "bin"] if cuda else []))
        if cuda:
            env.setdefault("CUDA_HOME", cuda)
        steps = [
            ("export views", [str(self.discovery.nurec_python), "-u", str(TOOLS / "ncore_eval_export.py"),
                              str(clip), "--out", str(out / "views.npz"), "--every", str(every)]),
            ("render + PSNR", [str(gsplat), "-u", str(TOOLS / "ncore_eval_render.py"), str(ply),
                               str(out / "views.npz"), "--out", str(out)]),
        ]
        return self._start("evaluate", out, steps, env, {"ply": self.discovery.rel(ply), "clip": self.discovery.rel(clip)})

    def carla(self, cfg: dict) -> dict:
        allowed = {"ply", "clip", "use_carla", "npc", "start", "duration", "lateral", "lift", "fov", "gpu", "name"}
        if not isinstance(cfg, dict) or set(cfg) - allowed:
            raise ValueError("unknown carla fields")
        ply = self._path(cfg.get("ply", ""), suffix=".ply")
        clip = self._path(cfg.get("clip", ""), suffix=".json")
        use_carla = cfg.get("use_carla", True)
        if type(use_carla) is not bool:
            raise ValueError("use_carla must be boolean")
        npc = cfg.get("npc", 3)
        if type(npc) is not int or not 0 <= npc <= 8:
            raise ValueError("npc must be in 0..8")
        nums = {}
        for key, default, lo, hi in (("start", 0.0, 0, 3600), ("duration", 10.0, 0, 3600), ("lateral", 0.0, -10, 10),
                                     ("lift", 0.0, -1, 20), ("fov", 90.0, 30, 150)):
            v = cfg.get(key, default)
            if type(v) not in (int, float) or not lo <= v <= hi:
                raise ValueError(f"{key} must be a number in {lo}..{hi}")
            nums[key] = float(v)
        gsplat = self.discovery.gsplat_python
        if gsplat is None:
            raise ValueError("gsplat venv not found; run `fflab setup` (or set FFLAB_VENV)")
        if use_carla:
            if not _port_open(CARLA_PORT):
                raise ValueError("no CARLA server on localhost:2000: start it first (or render without CARLA)")
            if not self.carla_server.client_ok():
                raise ValueError("the gsplat venv lacks the CARLA client: pip install carla==0.9.16 into it")
        name = cfg.get("name") or f"{ply.stem}_{uuid.uuid4().hex[:6]}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise ValueError("name: letters, digits, _ . - only")
        out = self._new_output("carla", name)
        info = {"ply": self.discovery.rel(ply), "clip": self.discovery.rel(clip), "use_carla": use_carla,
                "npc": npc if use_carla else 0, **nums}
        (out / "job.json").write_text(json.dumps(info))
        cuda = _cuda_home()
        env = self._env(cfg.get("gpu"), [gsplat.parent] + ([Path(cuda) / "bin"] if cuda else []))
        if cuda:
            env.setdefault("CUDA_HOME", cuda)
        bridge = [str(gsplat), "-u", str(TOOLS / "carla_nurec_bridge.py"), str(ply), str(out / "traj.npz"),
                  "--out", str(out), "--start", str(nums["start"]), "--duration", str(nums["duration"]),
                  "--lateral", str(nums["lateral"]), "--lift", str(nums["lift"]), "--fov", str(nums["fov"])]
        if use_carla:
            bridge += ["--carla", f"127.0.0.1:{CARLA_PORT}", "--npc", str(npc)]
        cfg_json = trainer_config(ply)
        if cfg_json is not None:  # 3DGUT scene: align its KITTI-360 world to the clip's
            data = next((self.workspace / d["path"] for d in self.discovery.datasets()), None)
            if data is None:
                raise ValueError("aligning a 3DGUT scene needs the KITTI-360 archive (data_poses.zip) in the workspace")
            bridge += ["--trainer-config", str(cfg_json), "--kitti-data", str(data)]
            info["kind"] = "3dgut"
        steps = [
            ("export trajectory", [str(self.discovery.nurec_python), "-u", str(TOOLS / "ncore_trajectory_export.py"),
                                   str(clip), "--out", str(out / "traj.npz")]),
            ("render" + (" with CARLA" if use_carla else ""), bridge),
        ]
        return self._start("carla", out, steps, env, info)

    # ---- runner
    def _start(self, kind, out, steps, env, info):
        with self._lock:
            if self._closed:
                raise ValueError("pipeline manager is shut down")
            if any(j["kind"] == kind and j["state"] == "running" for j in self._jobs.values()):
                shutil.rmtree(out, ignore_errors=True)
                raise ValueError(f"a {kind} job is already running")
            job_id = uuid.uuid4().hex
            job = {"job_id": job_id, "kind": kind, "state": "running", "step": None, "error": None,
                   "tail": bytearray(), "process": None, "output": out, "info": info, "cancel": False}
            self._jobs[job_id] = job
            job["thread"] = threading.Thread(target=self._run, args=(job, steps, env), daemon=True)
            job["thread"].start()
            return self.status(job_id)

    def _run(self, job, steps, env):
        try:
            for label, argv in steps:
                with self._lock:
                    if job["cancel"]:
                        break
                    job["step"] = label
                    job["tail"].extend(f"\n=== {label} ===\n".encode())
                    job["process"] = subprocess.Popen(
                        argv, cwd=str(self.workspace), env=env, shell=False, stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                _tail(job["process"].stdout, job["tail"])
                code = job["process"].wait()
                if code:
                    raise RuntimeError(f"{label} exited {code}")
            with self._lock:
                job["state"] = "cancelled" if job["cancel"] else "succeeded"
        except Exception as error:  # noqa: BLE001 - reported to the client
            with self._lock:
                job["state"] = "cancelled" if job["cancel"] else "failed"
                job["error"] = str(error)

    def _job(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id not in self._jobs:
            raise ValueError("unknown pipeline job")
        return self._jobs[job_id]

    def _artifacts(self, job):
        out = job["output"]
        if not out.is_dir():
            return []
        return sorted(p.relative_to(out).as_posix() for p in out.rglob("*")
                      if p.is_file() and not p.is_symlink() and p.suffix.lower() in ARTIFACTS and p.stat().st_size)

    def status(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            result = {k: job[k] for k in ("job_id", "kind", "state", "step", "error", "info")}
            result["log"] = bytes(job["tail"]).decode("utf-8", errors="replace")
            result["output"] = self.discovery.rel(job["output"])
            result["artifacts"] = self._artifacts(job)
            metrics = job["output"] / "metrics.json"
            if job["kind"] == "evaluate" and metrics.is_file():
                result["metrics"] = json.loads(metrics.read_text())
            return result

    def cancel(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            job["cancel"] = True
            process = job["process"]
        if process is not None:
            _stop(process)
        return self.status(job_id)

    def artifact(self, job_id, relative) -> Path:
        with self._lock:
            job = self._job(job_id)
            if relative not in self._artifacts(job):
                raise ValueError("unknown pipeline artifact")
            path = (job["output"] / relative).resolve()
            if not path.is_relative_to(job["output"].resolve()):
                raise ValueError("artifact escapes output")
            return path

    def shutdown(self):
        if self.carla_server.proc is not None:
            self.carla_server._kill()
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
            for job in jobs:
                job["cancel"] = True
        for job in jobs:
            if job["process"] is not None:
                _stop(job["process"])
            if "thread" in job:
                job["thread"].join(timeout=5)
