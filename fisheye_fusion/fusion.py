from pathlib import Path

import cv2
import numpy as np

from .geometry import project_points, transform_points
from .models import CameraConfig, ProjectManifest


def _camera_map(camera: CameraConfig, output_size: tuple[int, int]):
    width, height = output_size
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    lon = (xs / width - 0.5) * 2 * np.pi
    lat = (ys / height - 0.5) * np.pi
    rays = np.stack(
        [np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)],
        axis=-1,
    ).reshape(-1, 3)
    # ponytail: rotation-only panorama; translated rigs need depth-aware warping.
    rays = rays @ np.asarray(camera.T_rig_from_camera)[:3, :3]
    uv, valid = project_points(rays, camera.K, camera.D, camera.model, xi=camera.xi)
    map_x = uv[:, 0].reshape(height, width).astype(np.float32)
    map_y = uv[:, 1].reshape(height, width).astype(np.float32)
    valid = valid.reshape(height, width) & np.isfinite(map_x) & np.isfinite(map_y)
    w, h = camera.resolution
    valid &= (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
    return np.nan_to_num(map_x), np.nan_to_num(map_y), valid


def fuse_images(manifest: ProjectManifest, root: str | Path, frame_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root)
    frame = manifest.frames[frame_index]
    cameras = {camera.id: camera for camera in manifest.cameras}
    width, height = manifest.fusion.resolution
    output = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    for camera_id in manifest.fusion.camera_ids:
        camera = cameras[camera_id]
        image = cv2.imread(str(root / frame.images[camera_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(frame.images[camera_id])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (image.shape[1], image.shape[0]) != camera.resolution:
            raise ValueError(f"{camera_id}: image resolution differs from calibration")
        map_x, map_y, valid = _camera_map(camera, (width, height))
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        w, h = camera.resolution
        edge = np.minimum.reduce([map_x, w - 1 - map_x, map_y, h - 1 - map_y])
        weight = (valid * np.clip(edge / 20, 0.01, 1)).astype(np.float32)[..., None]
        output += warped.astype(np.float32) * weight
        weights += weight
    return np.clip(output / np.maximum(weights, 1e-8), 0, 255).astype(np.uint8), weights[..., 0] > 0


def overlay_lidar(image: np.ndarray, points_camera: np.ndarray, camera: CameraConfig) -> np.ndarray:
    uv, valid = project_points(points_camera, camera.K, camera.D, camera.model, xi=camera.xi)
    result = image.copy()
    height, width = result.shape[:2]
    for x, y in uv[valid].astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(result, (x, y), 2, (255, 40, 40), -1)
    return result


def lidar_to_camera(points_lidar: np.ndarray, T_camera_from_lidar: np.ndarray) -> np.ndarray:
    return transform_points(T_camera_from_lidar, points_lidar)
