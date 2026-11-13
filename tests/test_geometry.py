import numpy as np
import pytest

from fisheye_fusion.geometry import (
    invert_transform,
    is_rigid_transform,
    pixel_to_ray,
    project_points,
    transform_points,
)


K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])


def test_identity_projection_and_principal_point_ray():
    points = np.array([[0, 0, 1], [1, 2, 2], [-1, 1, 3]], dtype=float)
    pixels, valid = project_points(points, K)
    np.testing.assert_allclose(pixels, [[50, 40], [100, 140], [50 - 100 / 3, 40 + 100 / 3]])
    assert valid.tolist() == [True, True, True]
    np.testing.assert_allclose(pixel_to_ray([[50, 40]], K), [[0, 0, 1]])


def test_positive_depth_filtering():
    pixels, valid = project_points([[0, 0, 1], [0, 0, 0], [0, 0, -1]], K)
    assert valid.tolist() == [True, False, False]
    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1:]).all()


def test_ninety_degree_rig_rotation():
    rotation = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=float,
    )
    transform = np.vstack((rotation, [0, 0, 0, 1]))
    assert is_rigid_transform(transform)
    np.testing.assert_allclose(transform_points(transform, [[1, 0, 2]]), [[0, 1, 2]])


def test_transform_inverse():
    transform = np.array(
        [[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]],
        dtype=float,
    )
    points = np.array([[1, 2, 3], [-4, 2, 1]], dtype=float)
    np.testing.assert_allclose(transform_points(invert_transform(transform), transform_points(transform, points)), points)


def test_pinhole_round_trip():
    pixels = np.array([[0, 0], [50, 40], [100, 80], [27, 61]], dtype=float)
    rays = pixel_to_ray(pixels, K)
    projected, valid = project_points(rays, K)
    assert valid.all()
    np.testing.assert_allclose(projected, pixels, atol=1e-10)


def test_fisheye_round_trip():
    cv2 = pytest.importorskip("cv2")
    distortion = np.array([0.01, -0.002, 0.0005, 0.0001])
    pixels = np.array([[10, 10], [50, 40], [90, 70]], dtype=float)
    rays = pixel_to_ray(pixels, K, distortion, "opencv_fisheye")
    projected, valid = project_points(rays, K, distortion, "opencv_fisheye")
    assert valid.all()
    np.testing.assert_allclose(projected, pixels, atol=1e-5)
