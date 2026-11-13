"""Create a small, portable subset from a validated project manifest."""
import shutil
import tempfile
from pathlib import Path

from .manifest import load_manifest, validate_manifest
from .models import ProjectManifest

DEFAULT_MAX_FRAMES = 4
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
HARD_MAX_FRAMES = 32
HARD_MAX_BYTES = 200 * 1024 * 1024


def sample_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    start: int = 0,
    stride: int = 1,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_bytes: int = DEFAULT_MAX_BYTES,
    include_lidar: bool = False,
    camera_ids: list[str] | None = None,
    write: bool = False,
) -> dict:
    source_manifest = Path(manifest_path).resolve()
    source_root = source_manifest.parent
    if source_manifest.stat().st_size > 4 * 1024**2:
        raise ValueError("Manifest exceeds 4 MiB")
    manifest = load_manifest(source_manifest)
    return sample_project(
        manifest, source_root, output_dir, start=start, stride=stride,
        max_frames=max_frames, max_bytes=max_bytes, include_lidar=include_lidar,
        camera_ids=camera_ids, write=write,
    )


def sample_project(
    manifest: ProjectManifest, source_root: str | Path, output_dir: str | Path,
    *, start=0, stride=1, max_frames=DEFAULT_MAX_FRAMES,
    max_bytes=DEFAULT_MAX_BYTES, include_lidar=False, camera_ids=None, write=False,
) -> dict:
    source_root = Path(source_root).resolve()
    validate_manifest(manifest, source_root)
    if start < 0 or stride < 1 or max_frames < 1 or max_frames > HARD_MAX_FRAMES:
        raise ValueError("start >= 0, stride >= 1, max_frames must be 1..32")
    if max_bytes < 1 or max_bytes > HARD_MAX_BYTES:
        raise ValueError("max_bytes must be 1..200 MiB")
    destination = Path(output_dir).resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("output must not be inside the source project")
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")

    selected = manifest.frames[start::stride][:max_frames]
    if not selected:
        raise ValueError("no frames selected")
    selected_ids = manifest.fusion.camera_ids if camera_ids is None else camera_ids
    selection = manifest.model_dump()
    selection["fusion"]["camera_ids"] = selected_ids
    manifest = ProjectManifest.model_validate(selection)
    files: list[tuple[Path, Path]] = []
    new_frames = []
    for index, frame in enumerate(selected):
        images = {}
        for camera_index, camera_id in enumerate(selected_ids):
            relative = Path(frame.images[camera_id])
            source = _source_file(source_root, relative)
            target = Path("images") / f"{index:06d}_camera{camera_index}{source.suffix.lower()}"
            images[camera_id] = target.as_posix()
            files.append((source, target))
        lidar = None
        if include_lidar:
            if not frame.lidar:
                raise ValueError(f"frame {index} has no lidar file")
            source = _source_file(source_root, Path(frame.lidar))
            target = Path("lidar") / f"{index:06d}{source.suffix.lower()}"
            lidar = target.as_posix()
            files.append((source, target))
        item = frame.model_dump()
        item["images"] = images
        item["image_timestamps_ns"] = {
            key: value for key, value in frame.image_timestamps_ns.items() if key in images
        }
        item["lidar"] = lidar
        new_frames.append(item)

    payload_bytes = sum(source.stat().st_size for source, _ in files)
    if payload_bytes > max_bytes:
        raise ValueError(f"sample is {payload_bytes / 1048576:.1f} MiB; limit is {max_bytes / 1048576:.1f} MiB")
    data = manifest.model_dump()
    data["root"] = "."
    data["frames"] = new_frames
    data["cameras"] = [c.model_dump() for c in manifest.cameras if c.id in selected_ids]
    sampled = ProjectManifest.model_validate(data)
    metadata = sampled.model_dump_json(indent=2).encode("utf-8")
    total_bytes = payload_bytes + len(metadata)
    if total_bytes > max_bytes:
        raise ValueError("sample including manifest exceeds byte limit")
    result = {
        "frames": len(new_frames),
        "files": len(files),
        "bytes": total_bytes,
        "output": str(destination),
        "manifest": str(destination / "manifest.json"),
        "written": False,
        "camera_ids": selected_ids,
        "include_lidar": include_lidar,
    }
    if not write:
        return result

    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < total_bytes + 64 * 1024**2:
        raise ValueError("insufficient disk space; keep 64 MiB free")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        copied_bytes = len(metadata)
        for source, target in files:
            target_path = staging / target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target_path.open("xb") as writer:
                while chunk := reader.read(1024 * 1024):
                    copied_bytes += len(chunk)
                    if copied_bytes > max_bytes:
                        raise ValueError("copied sample exceeds byte limit")
                    writer.write(chunk)
        (staging / "manifest.json").write_bytes(metadata)
        validate_manifest(sampled, staging)
        if destination.exists():
            raise ValueError(f"output already exists: {destination}")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result["written"] = True
    result["bytes"] = copied_bytes
    return result


def _source_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative.as_posix() or ":" in str(relative):
        raise ValueError(f"unsafe dataset path: {relative}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise ValueError(f"symlink dataset path is not allowed: {relative}")
    source = candidate.resolve()
    if source.parent == root or root in source.parents:
        if source.is_file() and not source.is_symlink():
            return source
    raise ValueError(f"dataset file is missing or unsafe: {relative}")
