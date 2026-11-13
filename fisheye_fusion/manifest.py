import json
from pathlib import Path

from .models import ProjectManifest


def load_manifest(path: str | Path) -> ProjectManifest:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = ProjectManifest.model_validate(data)
    validate_manifest(manifest, path.parent)
    return manifest


def validate_manifest(manifest: ProjectManifest, root: str | Path = ".") -> ProjectManifest:
    root = Path(root).resolve()
    ids = [camera.id for camera in manifest.cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("camera IDs must be unique")
    camera_ids = set(ids)
    if not manifest.fusion.camera_ids:
        raise ValueError("fusion.camera_ids must not be empty")
    unknown = set(manifest.fusion.camera_ids) - camera_ids
    if unknown:
        raise ValueError(f"fusion references unknown cameras: {sorted(unknown)}")
    for frame in manifest.frames:
        if not set(frame.images).issubset(camera_ids):
            raise ValueError("frame references an unknown camera")
        for value in list(frame.images.values()) + ([frame.lidar] if frame.lidar else []):
            if Path(value).is_absolute():
                raise ValueError(f"path escapes project root: {value}")
            candidate = (root / value).resolve()
            if root not in candidate.parents and candidate != root:
                raise ValueError(f"path escapes project root: {value}")
            if not candidate.is_file():
                raise ValueError(f"file does not exist: {value}")
    return manifest
