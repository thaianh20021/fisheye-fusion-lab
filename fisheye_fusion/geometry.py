import cv2
import numpy as np


def rigid_transform(matrix: list[list[float]] | np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("transform last row must be [0, 0, 0, 1]")
    if not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(value[:3, :3]), 1):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return value


def is_rigid_transform(matrix):
    try:
        rigid_transform(matrix)
        return True
    except ValueError:
        return False


def invert_transform(matrix):
    return np.linalg.inv(rigid_transform(matrix))


def pixel_to_ray(uv, K, D=(), model="pinhole", xi=None):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if model in ("pinhole", "opencv_pinhole"):
        xy = cv2.undistortPoints(uv, K, D if D.size else None).reshape(-1, 2)
        rays = np.column_stack([xy, np.ones(len(xy))])
    elif model == "mei":
        _check_xi(xi)
        xy = cv2.undistortPointsIter(
            uv, K, D, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
        ).reshape(-1, 2)
        r2 = np.sum(xy * xy, axis=1)
        radicand = 1 + (1 - xi * xi) * r2
        if np.any(radicand < 0):
            raise ValueError("pixel outside Mei inverse projection domain")
        scale = (xi + np.sqrt(radicand)) / (1 + r2)
        rays = np.column_stack([xy * scale[:, None], scale - xi])
    elif model == "opencv_fisheye":
        rays = cv2.fisheye.undistortPoints(uv, K, D).reshape(-1, 2)
        rays = np.column_stack([rays, np.ones(len(rays))])
    else:
        raise ValueError(f"unsupported camera model: {model}")
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ rigid_transform(transform).T)[:, :3]


def _check_xi(xi):
    if xi is None or not np.isfinite(xi) or xi < 0:
        raise ValueError("Mei camera requires finite nonnegative xi")


def project_points(points, K, D=(), model="pinhole", xi=None):
    points = np.asarray(points, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    valid = (points[:, 2] > 1e-8) & np.isfinite(points).all(axis=1)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if model not in ("pinhole", "opencv_pinhole", "opencv_fisheye", "mei"):
        raise ValueError(f"unsupported camera model: {model}")
    if model == "mei":
        _check_xi(xi)
        finite = np.isfinite(points).all(axis=1)
        norm = np.linalg.norm(np.where(finite[:, None], points, 0), axis=1)
        # Mei's injective domain can include rays behind the optical plane.
        bound = xi if xi <= 1 else 1 / xi
        valid = finite & (norm > 1e-8) & (points[:, 2] > -bound * norm + 1e-8)
        if valid.any():
            normalized = points[valid, :2] / (points[valid, 2] + xi * norm[valid])[:, None]
            plane = np.column_stack([normalized, np.ones(len(normalized))])
            projected, _ = cv2.projectPoints(plane, np.zeros(3), np.zeros(3), K, D)
            uv[valid] = projected.reshape(-1, 2)
        return uv, valid
    if not valid.any():
        return uv, valid
    if model in ("pinhole", "opencv_pinhole"):
        projected, _ = cv2.projectPoints(points[valid], np.zeros(3), np.zeros(3), K, D if D.size else None)
        uv[valid] = projected.reshape(-1, 2)
    elif model == "opencv_fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            points[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D
        )
        uv[valid] = projected.reshape(-1, 2)
    else:
        raise ValueError(f"unsupported camera model: {model}")
    return uv, valid
