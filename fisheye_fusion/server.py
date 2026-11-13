import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .backends import BACKENDS
from .demo import make_demo
from .fusion import fuse_images
from .jobs import make_bundle
from .manifest import load_manifest
from .models import ProjectManifest
from .sample import sample_dataset, sample_project

WORKSPACE = Path(os.environ.get("FUSION_WORKSPACE", Path(__file__).resolve().parent.parent)).resolve()
STATIC = Path(__file__).parent / "static"


class Job(BaseModel):
    manifest: str
    camera_ids: list[str] | None = None
    frame: int = Field(default=0, ge=0)
    width: Literal[512, 1024, 2048] = 1024

class SampleJob(Job):
    output: str
    max_frames: int = Field(default=4, ge=1, le=32)
    max_mib: int = Field(default=50, ge=1, le=200)
    include_lidar: bool = False
    write: bool = False

class KittiJob(BaseModel):
    dataset_root: str
    sequence: str
    frame_ids: list[int] = Field(min_length=1, max_length=32)
    camera_ids: list[str] = Field(default_factory=lambda: ["image_02", "image_03"])
    output: str
    max_mib: int = Field(default=50, ge=1, le=200)
    write: bool = False

class ReconstructionJob(BaseModel):
    repo: str
    python: str
    checkpoint: str
    ncore_path: str
    model: Literal["pa-front", "pa-multiview", "pq-front"] = "pa-front"
    max_chunks: int = Field(default=1, ge=1, le=2)
    n_gaussians: int = Field(default=200000, ge=100000, le=2000000)
    render_preview: bool = False
    include_dynamic: bool = False

class TrainingInput(BaseModel):
    dataset: str = Field(min_length=1, max_length=512)
    camera_ids: list[str] | None = Field(default=None, max_length=4)

class TrainingJob(TrainingInput):
    steps: int = Field(default=1200, ge=1, le=10000, strict=True)


def png(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def project(job):
    path = (WORKSPACE / job.manifest).resolve()
    if not path.is_relative_to(WORKSPACE) or path.suffix.lower() != ".json":
        raise ValueError("Manifest must be a JSON file within FUSION_WORKSPACE")
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Manifest exceeds 4 MiB")
    manifest = load_manifest(path)
    payload = manifest.model_dump()
    if job.camera_ids is not None:
        payload["fusion"]["camera_ids"] = job.camera_ids
    payload["fusion"]["resolution"] = [job.width, job.width // 2]
    manifest = ProjectManifest.model_validate(payload)
    if job.frame >= len(manifest.frames):
        raise ValueError("Frame index out of range")
    return path, manifest


def create_app():
    from .reconstruction import ReconstructionManager
    from .training import TrainingManager
    reconstruction = ReconstructionManager(WORKSPACE)
    training = TrainingManager(WORKSPACE)

    @asynccontextmanager
    async def lifespan(app):
        yield
        training.shutdown()
        reconstruction.shutdown()

    app = FastAPI(title="Fisheye Fusion Lab", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    def health():
        return {"ok": True, "workspace": str(WORKSPACE)}

    @app.get("/api/backends")
    def backends():
        return BACKENDS

    @app.get("/api/runtime")
    def runtime():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            gpu = result.stdout.strip() if result.returncode == 0 else "Unavailable"
        except (OSError, subprocess.TimeoutExpired):
            gpu = "Unavailable"
        return {"cpu_preview": True, "gpu": gpu, "cuda_training": "GEER local job runner",
                "cuda_inference": "Instant NuRec runner; separate environment and local assets required",
                "cloud": "CPU bundle + Instant NuRec inference notebook"}

    @app.post("/api/training/inspect")
    def training_inspect(job: TrainingInput):
        return training.inspect(job.dataset, job.camera_ids)

    @app.post("/api/training")
    def training_start(job: TrainingJob):
        return training.start(job.dataset, job.camera_ids, job.steps)

    @app.get("/api/training/{job_id}")
    def training_status(job_id: str):
        return training.status(job_id)

    @app.post("/api/training/{job_id}/cancel")
    def training_cancel(job_id: str):
        return training.cancel(job_id)

    @app.get("/api/training/{job_id}/artifacts/{name}")
    def training_artifact(job_id: str, name: str):
        return FileResponse(training.artifact(job_id, name))

    def experiment_file(name, run):
        runs = {"baseline": "real-gs-run", "geer": "geer-run"}
        if run not in runs:
            raise ValueError("Unknown experiment run")
        allowed = {"metrics.json", "scene.ply", "reference.png", "before.png", "after.png"}
        allowed.update(f"novel-{i:02d}.png" for i in range(9))
        if name not in allowed:
            raise ValueError("Unknown experiment artifact")
        root = (WORKSPACE / "outputs" / runs[run]).resolve()
        path = (root / name).resolve()
        if not root.is_relative_to(WORKSPACE) or path.parent != root or not path.is_file():
            raise ValueError("Experiment has not completed or artifact is missing")
        return path

    @app.get("/api/experiment")
    def experiment(run: str = "baseline"):
        path = experiment_file("metrics.json", run)
        if path.stat().st_size > 1024**2:
            raise ValueError("Experiment metadata too large")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/experiment/artifacts/{name:path}")
    def experiment_artifact(name: str, run: str = "baseline"):
        return FileResponse(experiment_file(name, run))

    @app.post("/api/demo")
    def demo(model: Literal["opencv_fisheye", "pinhole"] = "opencv_fisheye"):
        path = make_demo(WORKSPACE / "examples" / f"synthetic_{model}", model)
        return {"manifest": str(path.relative_to(WORKSPACE))}

    @app.post("/api/inspect")
    def inspect(job: Job):
        path, manifest = project(job)
        images = []
        for camera in manifest.cameras:
            ref = manifest.frames[job.frame].images.get(camera.id)
            if not ref:
                continue
            image = cv2.imread(str(path.parent / ref))
            if image is None:
                raise ValueError(f"Cannot decode {ref}")
            if (image.shape[1], image.shape[0]) != camera.resolution:
                raise ValueError(f"Image/calibration resolution mismatch: {camera.id}")
            image = cv2.resize(image, (320, round(320 * image.shape[0] / image.shape[1])))
            images.append({"id": camera.id, "png": png(image)})
        return {"manifest": manifest.model_dump(), "images": images, "frame_count": len(manifest.frames)}

    @app.post("/api/render")
    def render(job: Job):
        path, manifest = project(job)
        image, coverage = fuse_images(manifest, path.parent, job.frame)
        return {"png": png(image[:, :, ::-1]), "mask": png(coverage.astype("uint8") * 255), "coverage": float(coverage.mean()),
                "operation": "rotation-only panorama", "translation_ignored": any(np.any(np.asarray(c.T_rig_from_camera)[:3, 3]) for c in manifest.cameras)}


    @app.post("/api/bundle/{platform}")
    def bundle(platform: Literal["colab", "kaggle"], job: Job):
        path, manifest = project(job)
        manifest.frames = [manifest.frames[job.frame]]
        content = make_bundle(manifest, path.parent, platform)
        return Response(content, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="fusion-{platform}.zip"'})

    @app.post("/api/sample")
    def sample(job: SampleJob):
        path, manifest = project(job)
        destination = (WORKSPACE / job.output).resolve()
        if not destination.is_relative_to(WORKSPACE) or destination == WORKSPACE:
            raise ValueError("Sample output must be within FUSION_WORKSPACE")
        result = sample_dataset(
            path, destination, start=job.frame, max_frames=job.max_frames,
            max_bytes=job.max_mib * 1024**2, camera_ids=job.camera_ids or manifest.fusion.camera_ids,
            include_lidar=job.include_lidar, write=job.write,
        )
        result["manifest"] = str(Path(result["manifest"]).relative_to(WORKSPACE))
        return result

    @app.post("/api/import/kitti360")
    def import_kitti360(job: KittiJob):
        from .kitti360 import build_kitti360_manifest
        root = (WORKSPACE / job.dataset_root).resolve()
        output = (WORKSPACE / job.output).resolve()
        if not root.is_relative_to(WORKSPACE) or not output.is_relative_to(WORKSPACE) or output == WORKSPACE:
            raise ValueError("Dataset and output must be within FUSION_WORKSPACE; use CLI for external datasets")
        manifest = build_kitti360_manifest(
            root, job.sequence, job.frame_ids, camera_ids=job.camera_ids,
            max_bytes=job.max_mib * 1024**2,
        )
        result = sample_project(
            manifest, root, output, max_frames=len(job.frame_ids),
            max_bytes=job.max_mib * 1024**2, write=job.write,
        )
        result["manifest"] = str(Path(result["manifest"]).relative_to(WORKSPACE))
        return result

    @app.post("/api/reconstruction/preflight")
    def reconstruction_preflight(job: ReconstructionJob):
        return reconstruction.preflight(job.model_dump())

    @app.post("/api/reconstruction")
    def reconstruction_start(job: ReconstructionJob):
        return reconstruction.start(job.model_dump())

    @app.get("/api/reconstruction/{job_id}")
    def reconstruction_status(job_id: str):
        return reconstruction.status(job_id)

    @app.post("/api/reconstruction/{job_id}/cancel")
    def reconstruction_cancel(job_id: str):
        return reconstruction.cancel(job_id)

    @app.get("/api/reconstruction/{job_id}/artifacts/{relative:path}")
    def reconstruction_artifact(job_id: str, relative: str):
        path = reconstruction.artifact(job_id, relative)
        return FileResponse(path)

    @app.get("/api/notebooks/instant-nurec/{platform}")
    def nurec_notebook(platform: Literal["colab", "kaggle"]):
        from .nurec_notebook import make_nurec_notebook
        return Response(
            make_nurec_notebook(platform), media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="instant-nurec-{platform}.ipynb"'},
        )

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
