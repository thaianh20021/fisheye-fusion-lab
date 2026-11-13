"""Read a bounded KITTI-360 RGB subset without copying or downloading data."""
from datetime import datetime
from pathlib import Path
import re

import cv2
import numpy as np

from .geometry import rigid_transform
from .models import ProjectManifest


def _file(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing file or path outside dataset root: {relative}")
    return path


def _records(path: Path) -> dict[str, list[float]]:
    with path.open("rb") as stream:
        content = stream.read(4 * 1024**2 + 1)
    if len(content) > 4 * 1024**2:
        raise ValueError("calibration records exceed 4 MiB")
    records = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid calibration record in {path.name}")
        key = key.strip()
        if key in records:
            raise ValueError(f"duplicate calibration key: {key}")
        records[key] = [float(token) for token in value.split()]
    return records


def _array(records, key, shape):
    if key not in records or len(records[key]) != int(np.prod(shape)):
        raise ValueError(f"missing or malformed calibration: {key}")
    array = np.asarray(records[key], dtype=float).reshape(shape)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite calibration: {key}")
    return array


def _transform(values):
    transform = np.eye(4)
    transform[:3] = np.asarray(values, dtype=float).reshape(3, 4)
    rigid_transform(transform)
    return transform


def _resolution(values):
    if any(not np.isfinite(v) or v != int(v) or not 0 < v <= 8192 for v in values):
        raise ValueError("invalid image resolution")
    return [int(v) for v in values]


def _fisheye(path: Path) -> dict:
    try:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError) as error:
        raise ValueError(f"invalid calibration YAML: {path.name}") from error
    try:
        if not storage.isOpened():
            raise ValueError(f"cannot read calibration: {path.name}")

        def number(*keys):
            node = storage.getNode(keys[0])
            for key in keys[1:]:
                node = node.getNode(key)
            if node.empty() or not (node.isInt() or node.isReal()):
                raise ValueError(f"missing numeric calibration: {'.'.join(keys)}")
            value = node.real()
            if not np.isfinite(value):
                raise ValueError(f"non-finite calibration: {'.'.join(keys)}")
            return value

        fx, fy, cx, cy = [number("projection_parameters", key)
                          for key in ("gamma1", "gamma2", "u0", "v0")]
        return dict(
            model="mei",
            xi=number("mirror_parameters", "xi"),
            K=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            D=[number("distortion_parameters", "k1"),
               number("distortion_parameters", "k2"), 0, 0],
            resolution=_resolution([number("image_width"), number("image_height")]),
        )
    except cv2.error as error:
        raise ValueError(f"invalid calibration YAML: {path.name}") from error
    finally:
        storage.release()


def _timestamps(path: Path, frame_ids: list[int]) -> dict[int, int]:
    selected = set(frame_ids)
    result = {}
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index > frame_ids[-1]:
                break
            if index not in selected:
                continue
            match = re.fullmatch(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{1,9})", line.strip())
            if not match:
                raise ValueError(f"invalid timestamp for frame {index} in {path}")
            delta = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)
            result[index] = (delta.days * 86400 + delta.seconds) * 10**9 + int(match[2].ljust(9, "0"))
    if selected != result.keys():
        raise ValueError(f"missing camera timestamps: {sorted(selected - result.keys())}")
    if any(result[b] <= result[a] for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("camera timestamps must be strictly increasing")
    return result


def build_kitti360_manifest(
    root, sequence, frame_ids, camera_ids=("image_02", "image_03"),
    max_bytes=50 * 1024**2,
) -> ProjectManifest:
    """Reference RGB files relative to root; max_bytes bounds selected image bytes.

    The rig is the GPS/IMU frame. Frame timestamps use the first selected camera,
    parsed as timezone-neutral epoch nanoseconds without float conversion.
    Each camera's original timestamp is also retained in image_timestamps_ns.
    """
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sequence):
        raise ValueError("sequence must be a single safe identifier")
    frame_ids = list(frame_ids)
    camera_ids = list(camera_ids)
    if (not 1 <= len(frame_ids) <= 32
            or any(type(i) is not int or not 0 <= i < 10**10 for i in frame_ids)
            or any(b <= a for a, b in zip(frame_ids, frame_ids[1:]))):
        raise ValueError("frame_ids must contain 1..32 strictly increasing nonnegative integers")
    known = ("image_00", "image_01", "image_02", "image_03")
    if (len(camera_ids) not in (1, 2, 4)
            or any(c not in known for c in camera_ids)
            or len(set(camera_ids)) != len(camera_ids)):
        raise ValueError("select 1, 2 or 4 unique known cameras")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 200 * 1024**2:
        raise ValueError("max_bytes must be an integer in 1..200 MiB")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("dataset root must be a directory")
    calibration = Path("calibration")
    extrinsics = _records(_file(root, calibration / "calib_cam_to_pose.txt"))
    perspective = (_records(_file(root, calibration / "perspective.txt"))
                   if any(c in known[:2] for c in camera_ids) else {})
    cameras, times = [], {}
    for camera in camera_ids:
        transform = _transform(_array(extrinsics, camera, (3, 4)))
        if camera in known[2:]:
            config = _fisheye(_file(root, calibration / f"{camera}.yaml"))
        else:
            suffix = camera[-2:]
            rectification = np.eye(4)
            rectification[:3, :3] = _array(perspective, f"R_rect_{suffix}", (3, 3))
            rigid_transform(rectification)
            transform = transform @ np.linalg.inv(rectification)
            config = dict(
                model="pinhole", D=[],
                K=_array(perspective, f"P_rect_{suffix}", (3, 4))[:, :3].tolist(),
                resolution=_resolution(_array(perspective, f"S_rect_{suffix}", (2,))),
            )
        cameras.append(dict(id=camera, T_rig_from_camera=transform.tolist(), **config))
        times[camera] = _timestamps(
            _file(root, Path("data_2d_raw") / sequence / camera / "timestamps.txt"), frame_ids)

    poses = {}
    wanted = set(frame_ids)
    with _file(root, Path("data_poses") / sequence / "poses.txt").open(encoding="utf-8") as stream:
        for line in stream:
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != 13 or not tokens[0].isdigit():
                raise ValueError("invalid poses.txt record")
            frame_id = int(tokens[0])
            if frame_id in wanted:
                if frame_id in poses:
                    raise ValueError(f"duplicate pose for frame {frame_id}")
                poses[frame_id] = _transform([float(v) for v in tokens[1:]]).tolist()
    if wanted != poses.keys():
        raise ValueError(f"missing exact poses for frames: {sorted(wanted - poses.keys())}")

    frames, seen, total = [], set(), 0
    for frame_id in frame_ids:
        images = {}
        for camera in camera_ids:
            directory = "data_rect" if camera in known[:2] else "data_rgb"
            relative = Path("data_2d_raw") / sequence / camera / directory / f"{frame_id:010d}.png"
            source = _file(root, relative)
            if source not in seen:
                total += source.stat().st_size
                seen.add(source)
            if total > max_bytes:
                raise ValueError(f"selected images exceed max_bytes ({total} > {max_bytes})")
            images[camera] = relative.as_posix()
        frames.append(dict(timestamp_ns=times[camera_ids[0]][frame_id],
                           image_timestamps_ns={camera: times[camera][frame_id] for camera in camera_ids},
                           images=images, T_world_from_rig=poses[frame_id]))
    return ProjectManifest.model_validate(dict(
        project=sequence, root=".", cameras=cameras, frames=frames,
        fusion=dict(camera_ids=camera_ids),
    ))
