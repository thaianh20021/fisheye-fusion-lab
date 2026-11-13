"""Validated, workspace-contained inputs for small GEER training runs."""

import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from .geometry import rigid_transform
from .manifest import load_manifest
from .sample import _source_file

MAX_BYTES = 200 * 1024**2
_REQUIRED = {"images", "Ks", "viewmats", "points", "colors",
             "train_indices", "test_indices"}
_OBSERVATION_KEYS = ("images", "Ks", "viewmats", "camera_models",
                     "distortions", "camera_ids")


def _load_npz(path: Path) -> dict:
    # Check NPY headers too: a small ZIP member can claim a huge array allocation.
    try:
        with path.open("rb") as source:
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if (not entries or len(entries) > 32 or len(set(names)) != len(names)
                        or sum(entry.file_size for entry in entries) > MAX_BYTES):
                    raise ValueError("NPZ exceeds 200 MiB or has invalid members")
                for entry in entries:
                    if (not entry.filename.endswith(".npy")
                            or Path(entry.filename).name != entry.filename
                            or "\\" in entry.filename or entry.flag_bits & 1):
                        raise ValueError("invalid NPZ member")
                    with archive.open(entry) as member:
                        version = np.lib.format.read_magic(member)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                        else:
                            raise ValueError("unsupported NPY version")
                        size = math.prod(shape) * dtype.itemsize
                        if (dtype.hasobject or any(n < 0 for n in shape)
                                or size > MAX_BYTES
                                or size != entry.file_size - member.tell()):
                            raise ValueError("invalid NPZ array size or object dtype")
            source.seek(0)
            with np.load(source, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"cannot read NPZ: {path.name}") from exc


def _numeric(value: np.ndarray, name: str, shape: tuple) -> None:
    if (value.shape != shape or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite numeric data with shape {shape}")


def _strings(value: np.ndarray, name: str, count: int) -> np.ndarray:
    if value.shape != (count,) or value.dtype.kind not in "US":
        raise ValueError(f"{name} must contain one string per observation")
    try:
        result = value.astype(str)
    except UnicodeError as exc:
        raise ValueError(f"invalid {name}") from exc
    if np.any(result == ""):
        raise ValueError(f"{name} must not contain empty strings")
    return result


def _points(data: dict) -> None:
    if not {"points", "colors"} <= data.keys():
        raise ValueError("seed NPZ requires points and colors")
    points = data["points"]
    if points.ndim != 2 or not 100 <= len(points) <= 30000:
        raise ValueError("require 100..30000 seed points")
    _numeric(points, "points", (len(points), 3))
    _numeric(data["colors"], "colors", points.shape)
    if np.any((data["colors"] < 0) | (data["colors"] > 1)):
        raise ValueError("colors must be in 0..1")


def _validate(data: dict) -> dict:
    if not _REQUIRED <= data.keys():
        raise ValueError(f"missing training arrays: {sorted(_REQUIRED - data.keys())}")
    images = data["images"]
    if (images.ndim != 4 or not 1 <= images.shape[0] <= 32
            or not all(1 <= n <= 512 for n in images.shape[1:3])
            or images.shape[3] != 3):
        raise ValueError("images must be [M,H,W,3], M <= 32, H/W <= 512")
    count = len(images)
    _numeric(images, "images", images.shape)
    if np.any((images < 0) | (images > 1)):
        raise ValueError("images must be in 0..1")
    _points(data)
    _numeric(data["Ks"], "Ks", (count, 3, 3))
    _numeric(data["viewmats"], "viewmats", (count, 4, 4))
    for k, view in zip(data["Ks"], data["viewmats"]):
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0
                or k[1, 0] != 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("invalid K: positive focal lengths, zero skew required")
        rigid_transform(view)
    data["camera_models"] = _strings(
        data.get("camera_models", np.full(count, "pinhole")), "camera_models", count)
    if not np.isin(data["camera_models"], ["pinhole", "fisheye"]).all():
        raise ValueError("camera_models must be pinhole or fisheye")
    data["distortions"] = data.get("distortions", np.zeros((count, 4)))
    _numeric(data["distortions"], "distortions", (count, 4))
    if np.any(data["distortions"][data["camera_models"] == "pinhole"] != 0):
        raise ValueError("pinhole distortion must be zero")
    if "camera_ids" in data:
        data["camera_ids"] = _strings(data["camera_ids"], "camera_ids", count)
    for key in ("train_indices", "test_indices"):
        indexes = data[key]
        if (indexes.ndim != 1 or indexes.dtype.kind not in "iu" or not indexes.size
                or np.any(indexes >= count) or np.any(indexes < 0)
                or len(np.unique(indexes)) != len(indexes)):
            raise ValueError(f"invalid {key}")
    if np.intersect1d(data["train_indices"], data["test_indices"]).size:
        raise ValueError("train/test indices must be disjoint")
    return {key: value for key, value in data.items()
            if key in _REQUIRED or key in _OBSERVATION_KEYS}


def _selection(camera_ids: list[str], available: list[str]) -> None:
    if (not camera_ids or any(not isinstance(c, str) for c in camera_ids)
            or len(set(camera_ids)) != len(camera_ids)
            or not set(camera_ids) <= set(available)):
        raise ValueError("camera_ids must be unique, nonempty, known IDs")


def _rig(path: Path, camera_ids: list[str] | None) -> dict:
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("manifest exceeds 4 MiB")
    manifest = load_manifest(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    declared_root = (root / manifest.root).resolve()
    if not declared_root.is_relative_to(root):
        raise ValueError("manifest root escapes manifest directory")
    # Validate even unselected references with the existing symlink-safe helper.
    for frame in manifest.frames:
        for value in list(frame.images.values()) + ([frame.lidar] if frame.lidar else []):
            _source_file(root, Path(value))
    selected = manifest.fusion.camera_ids if camera_ids is None else camera_ids
    cameras = {camera.id: camera for camera in manifest.cameras}
    _selection(selected, list(cameras))
    if len(selected) not in (1, 2, 4):
        raise ValueError("select 1, 2, or 4 cameras")
    if len(manifest.frames) < 2 or len(manifest.frames) * len(selected) > 32:
        raise ValueError("require >=2 frames and <=32 observations")
    sources = []
    for frame in manifest.frames:
        for camera_id in selected:
            if camera_id not in frame.images:
                raise ValueError(f"missing image for camera {camera_id}")
            sources.append((_source_file(root, Path(frame.images[camera_id])),
                            cameras[camera_id]))
    if sum(source.stat().st_size for source, _ in sources) > MAX_BYTES:
        raise ValueError("source images exceed cumulative 200 MiB limit")
    # Inspect every header before any pixel decode, including later observations.
    for source, camera in sources:
        try:
            with Image.open(source) as header:
                width, height = header.size
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"cannot decode image header: {source.name}") from exc
        if not (1 <= width <= 8192 and 1 <= height <= 8192):
            raise ValueError("image header dimensions must be 1..8192")
        if (width, height) != camera.resolution:
            raise ValueError("image dimensions do not match camera resolution")
    points_path = raw.get("training_points")
    if not isinstance(points_path, str) or not points_path:
        raise ValueError("training_points must reference a shared world-space seed NPZ")
    seed_path = _source_file(root, Path(points_path))
    if seed_path.suffix.lower() != ".npz":
        raise ValueError("training_points must be NPZ")
    seeds = _load_npz(seed_path)
    _points(seeds)
    data = {key: [] for key in _OBSERVATION_KEYS}
    expected_shape = None
    for frame in manifest.frames:
        if any(t != frame.timestamp_ns for t in frame.image_timestamps_ns.values()):
            raise ValueError("image timestamps require exact synchronization (zero offset)")
        for camera_id in selected:
            camera = cameras[camera_id]
            if camera.model not in ("pinhole", "opencv_fisheye"):
                raise ValueError(f"unsupported camera model: {camera.model}")
            if camera.model == "pinhole" and any(camera.D):
                raise ValueError("pinhole distortion must be zero")
            if camera_id not in frame.images:
                raise ValueError(f"missing image for camera {camera_id}")
            image_path = _source_file(root, Path(frame.images[camera_id]))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode image: {image_path.name}")
            height, width = image.shape[:2]
            if not (1 <= width <= 8192 and 1 <= height <= 8192):
                raise ValueError("decoded image dimensions must be 1..8192")
            if (width, height) != camera.resolution:
                raise ValueError("image dimensions do not match camera resolution")
            scale = min(1.0, 400 / max(width, height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            if expected_shape is not None and image.shape != expected_shape:
                raise ValueError("all resized images must have the same shape")
            expected_shape = image.shape
            k = np.asarray(camera.K, dtype=np.float64).copy()
            k[0] *= size[0] / width
            k[1] *= size[1] / height
            world_camera = (np.asarray(frame.T_world_from_rig)
                            @ np.asarray(camera.T_rig_from_camera))
            data["images"].append(image[:, :, ::-1].astype(np.float32) / 255)
            data["Ks"].append(k)
            data["viewmats"].append(np.linalg.inv(world_camera))
            data["camera_ids"].append(camera_id)
            data["camera_models"].append(
                "fisheye" if camera.model == "opencv_fisheye" else "pinhole")
            data["distortions"].append(
                camera.D if camera.model == "opencv_fisheye" else [0.0] * 4)
    result = {key: np.asarray(value) for key, value in data.items()}
    count = len(result["images"])
    result.update(points=seeds["points"], colors=seeds["colors"],
                  train_indices=np.arange(count - len(selected)),
                  test_indices=np.arange(count - len(selected), count))
    return _validate(result)


def prepare_dataset(workspace: Path, dataset: str,
                    camera_ids: list[str] | None = None) -> dict:
    """Read a small NPZ or canonical rig; never generate seed geometry."""
    workspace = Path(workspace).resolve()
    relative = Path(dataset)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("dataset escapes workspace") from exc
    path = _source_file(workspace, relative)
    if path.suffix.lower() == ".json":
        return _rig(path, camera_ids)
    if path.suffix.lower() != ".npz":
        raise ValueError("dataset must be an existing NPZ or JSON manifest")
    data = _validate(_load_npz(path))
    if camera_ids is not None:
        if not camera_ids:
            raise ValueError("camera_ids must be nonempty")
        if "camera_ids" not in data:
            raise ValueError("NPZ has no camera_ids for selection")
        _selection(camera_ids, list(data["camera_ids"]))
        keep = np.flatnonzero(np.isin(data["camera_ids"], camera_ids))
        mapping = np.full(len(data["images"]), -1, dtype=np.int64)
        mapping[keep] = np.arange(len(keep))
        for key in _OBSERVATION_KEYS:
            if key in data:
                data[key] = data[key][keep]
        for key in ("train_indices", "test_indices"):
            indexes = mapping[data[key]]
            data[key] = indexes[indexes >= 0]
        _validate(data)
    return data


def inspect_dataset(workspace: Path, dataset: str,
                    camera_ids: list[str] | None = None) -> dict:
    """Return a JSON-serializable summary of the validated selected dataset."""
    data = prepare_dataset(workspace, dataset, camera_ids)
    cameras = dict(zip(data["camera_ids"].tolist(), data["camera_models"].tolist())) \
        if "camera_ids" in data else {}
    return {
        "observations": len(data["images"]),
        "observation_count": len(data["images"]),
        "cameras": [{"id": camera_id, "model": model}
                    for camera_id, model in cameras.items()],
        "camera_ids": list(cameras),
        "camera_models": list(dict.fromkeys(data["camera_models"].tolist())),
        "resolution": [data["images"].shape[2], data["images"].shape[1]],
        "gaussians": len(data["points"]),
        "train_count": len(data["train_indices"]),
        "test_count": len(data["test_indices"]),
    }
