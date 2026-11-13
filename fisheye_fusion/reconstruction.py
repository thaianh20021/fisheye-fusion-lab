"""Offline Instant NuRec runner for explicitly trusted local software.

CLI contract: nvidia/instant-nurec@f2ecafcdf28f2dc8f2cc17447a772571cde558be.
This is process isolation, not a sandbox for an untrusted repo or checkpoint.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid


REVISION = "f2ecafcdf28f2dc8f2cc17447a772571cde558be"
LOG_LIMIT = 1024**2
RESERVE = 3 * 1024**3
EXTENSIONS = {".ply", ".png", ".npz", ".mp4"}
_PROBE = """
import importlib.util, json, shutil, sys
import torch
print(json.dumps({
    "python": list(sys.version_info[:3]),
    "cuda": bool(torch.cuda.is_available()),
    "torch": str(torch.__version__),
    "ninja": shutil.which("ninja") is not None,
    "render": all(importlib.util.find_spec(m) is not None for m in ("gsplat", "nvdiffrast")),
}))
"""


def _tail(stream, target):
    try:
        while chunk := getattr(stream, "read1", stream.read)(65536):
            target.extend(chunk)
            if len(target) > LOG_LIMIT:
                del target[:-LOG_LIMIT]
    finally:
        stream.close()


def _stop(process):
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _capture(argv, *, cwd, env, timeout, label):
    process = subprocess.Popen(
        argv, cwd=str(cwd), env=env, shell=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    tail = bytearray()
    reader = threading.Thread(target=_tail, args=(process.stdout, tail), daemon=True)
    reader.start()
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop(process)
        raise ValueError(f"{label} timed out") from None
    finally:
        reader.join(timeout=3)
    if reader.is_alive():
        raise ValueError(f"{label} output did not close")
    text = tail.decode("utf-8", errors="replace").strip()
    if code:
        raise ValueError(f"{label} failed: {text[-2000:]}")
    return text


class ReconstructionManager:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        self._lock = threading.RLock()
        self._jobs = {}
        self._closed = False

    def _path(self, value, *, directory=False, suffix=None):
        if (not isinstance(value, str) or not value or "\\" in value
                or ":" in value or Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("paths must be relative workspace paths using forward slashes")
        path = (self.workspace / value).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("path escapes workspace")
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"incorrect path type: {value}")
        if suffix and path.suffix.lower() != suffix:
            raise ValueError(f"path must end in {suffix}")
        return path

    def _python(self, value, repo):
        if (not isinstance(value, str) or not value or "\\" in value or ":" in value
                or Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("python must be a relative repo venv interpreter")
        path = self.workspace / value
        try:
            parts = path.relative_to(repo).parts
        except ValueError:
            raise ValueError("python must be inside the configured repo") from None
        allowed = {("bin", "python"), ("bin", "python3"), ("bin", "python3.11"),
                   ("Scripts", "python.exe")}
        if len(parts) != 3 or parts[0] != ".venv" or parts[1:] not in allowed:
            raise ValueError("python must be repo/.venv/bin/python[3[.11]] or .venv/Scripts/python.exe")
        # Resolve parents, never the executable: venv Python commonly links to /usr/bin.
        if path.parent.resolve(strict=True) != path.parent or not path.is_file():
            raise ValueError("venv interpreter parent must not redirect outside its lexical path")
        return path

    def _config(self, config):
        allowed = {"repo", "python", "checkpoint", "ncore_path", "model", "max_chunks",
                   "n_gaussians", "render_preview", "include_dynamic"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("unknown reconstruction config fields")
        value = {"model": "pa-front", "max_chunks": 1, "n_gaussians": 200000,
                 "render_preview": False, "include_dynamic": False, **config}
        for key in ("repo", "python", "checkpoint", "ncore_path"):
            if key not in value:
                raise ValueError(f"missing config field: {key}")
        if value["model"] not in ("pa-front", "pa-multiview", "pq-front"):
            raise ValueError("unsupported model")
        for key, low, high in (("max_chunks", 1, 2), ("n_gaussians", 100000, 2000000)):
            if type(value[key]) is not int or not low <= value[key] <= high:
                raise ValueError(f"{key} must be an integer in {low}..{high}")
        for key in ("render_preview", "include_dynamic"):
            if type(value[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        if value["include_dynamic"] and value["model"] == "pq-front":
            raise ValueError("include_dynamic requires a PA model")
        repo = self._path(value["repo"], directory=True)
        script = self._path((repo.relative_to(self.workspace) / "run_inference.py").as_posix(), suffix=".py")
        if script.parent != repo or script.name != "run_inference.py":
            raise ValueError("repo must contain run_inference.py")
        python = self._python(value["python"], repo)
        checkpoint = self._path(value["checkpoint"], suffix=".pth")
        if not checkpoint.stat().st_size:
            raise ValueError("checkpoint is empty")
        ncore = self._path(value["ncore_path"], suffix=".json")
        with ncore.open("rb") as stream:
            raw = stream.read(4 * 1024**2 + 1)
        if len(raw) > 4 * 1024**2:
            raise ValueError("NCore JSON exceeds 4 MiB")
        metadata = json.loads(raw)
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("NCore JSON must be a nonempty object")
        if {"cameras", "frames", "fusion"}.intersection(metadata):
            raise ValueError("canonical manifests are not NCore input")
        return value, repo, script, python, checkpoint, ncore

    def _environment(self, python, checkpoint):
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("PYTHON", "GIT_"))}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   INSTANT_NUREC_FULL_PT=str(checkpoint), PYTHONUNBUFFERED="1",
                   PYTHONDONTWRITEBYTECODE="1", MAX_JOBS="2", GIT_OPTIONAL_LOCKS="0",
                   PATH=str(python.parent) + os.pathsep + env.get("PATH", ""))
        return env

    def preflight(self, config: dict) -> dict:
        report = {"ok": False, "errors": [], "checks": {}, "expected_revision": REVISION,
                  "revision": None, "dirty": None}
        try:
            value, repo, _, python, checkpoint, _ = self._config(config)
            checks = report["checks"]
            checks["disk_free_bytes"] = shutil.disk_usage(self.workspace).free
            checks["backend"] = (repo / "instant_nurec" / "__init__.py").is_file()
            if checks["disk_free_bytes"] < RESERVE:
                report["errors"].append("at least 3 GiB free disk space required")
            if not checks["backend"]:
                report["errors"].append("repo missing instant_nurec package")
            if report["errors"]:
                return report
            env = self._environment(python, checkpoint)

            def git(*args):
                return _capture(["git", "-C", str(repo), *args], cwd=repo, env=env,
                                timeout=10, label="git preflight")

            if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
                raise ValueError("configured repo must be the git repository root")
            head = git("rev-parse", "--verify", "HEAD")
            if not re.fullmatch(r"[0-9a-f]{40}", head):
                raise ValueError("invalid git HEAD")
            report["revision"] = head
            report["dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
            if head != REVISION:
                report["errors"].append(f"repo HEAD must match expected revision {REVISION}")
            if report["dirty"]:
                report["errors"].append("repo must have a clean working tree")
            if report["errors"]:
                return report
            text = _capture([str(python), "-I", "-u", "-c", _PROBE], cwd=repo,
                            env=env, timeout=30, label="interpreter/CUDA preflight")
            runtime = json.loads(text.splitlines()[-1])
            checks.update(runtime)
            if runtime.get("python", [])[:2] != [3, 11]:
                report["errors"].append("Python 3.11 required")
            if runtime.get("cuda") is not True:
                report["errors"].append("CUDA-enabled torch and an available GPU required")
            if value["render_preview"] and not (runtime.get("ninja") and runtime.get("render")):
                report["errors"].append("render_preview requires ninja, gsplat and nvdiffrast")
        except (ValueError, OSError, RuntimeError, IndexError, TypeError) as error:
            report["errors"].append(str(error))
        report["ok"] = not report["errors"]
        return report

    def _output(self, job_id, create=False):
        path = self.workspace
        for part in ("outputs", "reconstruction", job_id):
            path = path / part
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("symlink output directories are forbidden")
            if create:
                path.mkdir(exist_ok=part != job_id)
            if not path.resolve().is_relative_to(self.workspace):
                raise ValueError("output escapes workspace")
        return path

    def start(self, config: dict) -> dict:
        with self._lock:
            if self._closed:
                raise ValueError("reconstruction manager is shut down")
            if any(job["state"] in ("running", "cancelling") for job in self._jobs.values()):
                raise ValueError("a reconstruction job is already active")
            report = self.preflight(config)
            if not report["ok"]:
                raise ValueError("; ".join(report["errors"]))
            value, repo, script, python, checkpoint, ncore = self._config(config)
            job_id = uuid.uuid4().hex
            output = self._output(job_id, create=True)
            argv = [str(python), "-u", str(script), "--model", value["model"],
                    "--ncore-path", str(ncore), "--output-dir", str(output), "--merge",
                    "--n-gaussians", str(value["n_gaussians"]), "--max-chunks", str(value["max_chunks"])]
            for key in ("render_preview", "include_dynamic"):
                if value[key]:
                    argv.append("--" + key.replace("_", "-"))
            process = subprocess.Popen(
                argv, cwd=str(repo), env=self._environment(python, checkpoint),
                shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            job = {"job_id": job_id, "state": "running", "returncode": None,
                   "error": None, "tail": bytearray(), "process": process, "artifacts": []}
            self._jobs[job_id] = job
            job["thread"] = threading.Thread(target=self._monitor, args=(job,), daemon=True)
            job["thread"].start()
            return self.status(job_id)

    def _artifacts(self, job_id):
        output = self._output(job_id)
        artifacts = []
        for directory, dirs, files in os.walk(output, followlinks=False):
            dirs[:] = [name for name in dirs
                       if not (Path(directory) / name).is_symlink()
                       and not getattr(Path(directory) / name, "is_junction", lambda: False)()]
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() not in EXTENSIONS or path.is_symlink():
                    continue
                if path.resolve().is_relative_to(output) and path.is_file() and path.stat().st_size:
                    artifacts.append(path.relative_to(output).as_posix())
        return sorted(artifacts)

    def _monitor(self, job):
        try:
            _tail(job["process"].stdout, job["tail"])
            code = job["process"].wait()
            with self._lock:
                job["returncode"] = code
                job["artifacts"] = self._artifacts(job["job_id"])
                if job["state"] == "cancelling":
                    job["state"] = "cancelled"
                elif code == 0 and any(Path(p).suffix.lower() == ".ply" for p in job["artifacts"]):
                    job["state"] = "succeeded"
                else:
                    job["state"] = "failed"
                    job["error"] = f"process exited {code}" if code else "no nonempty PLY output"
        except Exception as error:
            _stop(job["process"])
            with self._lock:
                job["state"] = "cancelled" if job["state"] == "cancelling" else "failed"
                job["returncode"] = job["process"].poll()
                job["error"] = str(error)

    def _job(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id not in self._jobs:
            raise ValueError("unknown reconstruction job")
        return self._jobs[job_id]

    def status(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            return {key: job[key] for key in ("job_id", "state", "returncode", "error")} | {
                "log": bytes(job["tail"]).decode("utf-8", errors="replace"),
                "artifacts": list(job["artifacts"]),
                "output": f"outputs/reconstruction/{job_id}",
            }

    def cancel(self, job_id) -> dict:
        with self._lock:
            job = self._job(job_id)
            if job["state"] not in ("running", "cancelling"):
                return self.status(job_id)
            job["state"] = "cancelling"
        _stop(job["process"])
        return self.status(job_id)

    def shutdown(self) -> None:
        """Stop active work and finish monitors before application shutdown."""
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
            for job in jobs:
                if job["state"] in ("running", "cancelling"):
                    job["state"] = "cancelling"
        for job in jobs:
            if job["state"] == "cancelling":
                _stop(job["process"])
            job["thread"].join(timeout=5)

    def artifact(self, job_id, relative) -> Path:
        with self._lock:
            job = self._job(job_id)
            if not isinstance(relative, str) or relative not in job["artifacts"]:
                raise ValueError("unknown reconstruction artifact")
            output = self._output(job_id)
            path = output
            for part in Path(relative).parts:
                path /= part
                if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                    raise ValueError("symlink artifacts are forbidden")
            if not path.resolve().is_relative_to(output) or not path.is_file():
                raise ValueError("artifact escapes output or is missing")
            return path
