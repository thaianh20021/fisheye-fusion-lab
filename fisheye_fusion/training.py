"""Bounded, local GEER training jobs; one CUDA process at a time."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid

import numpy as np

from .reconstruction import _tail
from .training_data import prepare_dataset, inspect_dataset

ARTIFACTS = {"metrics.json", "scene.ply", "reference.png", "before.png", "after.png"}
ARTIFACTS.update(f"novel-{i:02d}.png" for i in range(9))
ACTIVE = {"running", "cancelling"}


class TrainingManager:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        self.lock = threading.RLock()
        self.jobs = {}
        self.closed = False

    def inspect(self, dataset, camera_ids=None):
        return inspect_dataset(self.root, dataset, camera_ids)

    def _directory(self, job_id, create=False):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Unknown training job")
        path = self.root
        for part in ("outputs", "training", job_id):
            path /= part
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("Training output cannot be a link")
            if not path.resolve().is_relative_to(self.root):
                raise ValueError("Training output escapes workspace")
            if create:
                path.mkdir(exist_ok=part != job_id)
        return path

    def _command(self, dataset, output, steps):
        script = self.root / "tools" / "train_small_gs.py"
        if os.name == "nt":
            python = self.root / ".venv-geer/Scripts/python.exe"
            command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                       "-File", str(self.root / "tools/run_gs.ps1"), "-UseGeer"]
        else:
            python = self.root / ".venv-geer/bin/python"
            command = [str(python), "-u"]
        if not python.is_file() or not script.is_file():
            raise ValueError("GEER environment is not installed; see docs/geer-experiment.md")
        return command + [str(script), str(dataset), "--output", str(output),
                          "--steps", str(steps), "--geer"]

    def start(self, dataset, camera_ids=None, steps=1200):
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError("steps must be 1..10000")
        with self.lock:
            if self.closed:
                raise ValueError("Training manager is shut down")
            if any(job["state"] in ACTIVE for job in self.jobs.values()):
                raise ValueError("A training job is already active")
            if shutil.disk_usage(self.root).free < 1024**3:
                raise ValueError("Training requires at least 1 GiB free")
            arrays = prepare_dataset(self.root, dataset, camera_ids)
            job_id = uuid.uuid4().hex
            folder = self._directory(job_id)
            command = self._command(folder / "input.npz", folder / "result", steps)
            self._directory(job_id, create=True)
            np.savez_compressed(folder / "input.npz", **arrays)
            (folder / "config.json").write_text(json.dumps(
                {"dataset": dataset, "camera_ids": camera_ids, "steps": steps}), encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
            env.update(PYTHONUNBUFFERED="1", MAX_JOBS="2",
                       TORCH_EXTENSIONS_DIR=str(self.root / "outputs/cuda-cache-geer"))
            process = subprocess.Popen(command, cwd=self.root, env=env, shell=False,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            job = {"job_id": job_id, "state": "running", "step": 0, "steps": steps,
                   "error": None, "returncode": None, "artifacts": [],
                   "tail": bytearray(), "process": process}
            self.jobs[job_id] = job
            self._persist(job)
            job["thread"] = threading.Thread(target=self._monitor, args=(job,), daemon=True)
            job["thread"].start()
            return self.status(job_id)

    def _snapshot(self, job):
        log = bytes(job["tail"]).decode("utf-8", errors="replace")
        for line in reversed(log.splitlines()):
            try:
                item = json.loads(line)
                if isinstance(item, dict) and type(item.get("step")) is int:
                    job["step"] = min(item["step"], job["steps"])
                    break
            except ValueError:
                pass
        return {k: job[k] for k in ("job_id", "state", "step", "steps", "error",
                                    "returncode", "artifacts")} | {
            "log": log, "output": f"outputs/training/{job['job_id']}/result"}

    def _persist(self, job):
        folder = self._directory(job["job_id"])
        tmp = folder / "status.tmp"
        tmp.write_text(json.dumps(self._snapshot(job)), encoding="utf-8")
        tmp.replace(folder / "status.json")

    def _monitor(self, job):
        try:
            _tail(job["process"].stdout, job["tail"])
            code = job["process"].wait()
            with self.lock:
                job["returncode"] = code
                output = self._directory(job["job_id"]) / "result"
                job["artifacts"] = sorted(name for name in ARTIFACTS
                    if (output / name).is_file() and (output / name).stat().st_size)
                if job["state"] == "cancelling":
                    job["state"] = "cancelled"
                elif code == 0 and {"metrics.json", "scene.ply"} <= set(job["artifacts"]):
                    job["state"] = "succeeded"
                    job["step"] = job["steps"]
                else:
                    job["state"] = "failed"
                    job["error"] = f"Training exited {code}; inspect log"
                self._persist(job)
        except Exception as error:
            with self.lock:
                job["state"], job["error"] = "failed", str(error)
                self._persist(job)

    def status(self, job_id):
        with self.lock:
            folder = self._directory(job_id)
            if job_id in self.jobs:
                return self._snapshot(self.jobs[job_id])
            file = folder / "status.json"
            if not file.is_file() or file.stat().st_size > 2 * 1024**2:
                raise ValueError("Unknown training job")
            result = json.loads(file.read_text(encoding="utf-8"))
            if result["state"] in ACTIVE:
                result.update(state="interrupted", error="Server restarted; job is not monitored")
            return result

    def cancel(self, job_id):
        with self.lock:
            self.status(job_id)
            job = self.jobs.get(job_id)
            if not job or job["state"] not in ACTIVE:
                return self.status(job_id)
            job["state"] = "cancelling"
            process = job["process"]
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False)
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        job["thread"].join(timeout=5)
        return self.status(job_id)

    def artifact(self, job_id, name):
        state = self.status(job_id)
        if name not in ARTIFACTS or name not in state["artifacts"]:
            raise ValueError("Unknown training artifact")
        folder = self._directory(job_id)
        path = folder / "result" / name
        if path.resolve().parent != folder / "result" or not path.is_file():
            raise ValueError("Artifact escaped output or is missing")
        return path

    def shutdown(self):
        self.closed = True
        for job_id in list(self.jobs):
            self.cancel(job_id)
