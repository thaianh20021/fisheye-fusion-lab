import numpy as np
import pytest
from pydantic import ValidationError

from fisheye_fusion.geometry import pixel_to_ray, project_points
from fisheye_fusion.models import CameraConfig


K = np.array([[100.0, 0, 320], [0, 120.0, 240], [0, 0, 1]])
ZERO_D = [0.0, 0.0, 0.0, 0.0]


def camera_payload(**overrides):
    return {
        "id": "fisheye",
        "model": "mei",
        "resolution": [640, 480],
        "K": K.tolist(),
        "D": ZERO_D.copy(),
        "xi": 1.0,
        "T_rig_from_camera": np.eye(4).tolist(),
        **overrides,
    }


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0, 1.5])
def test_mei_principal_axis(xi):
    pixels, valid = project_points(
        [[0, 0, 1], [0, 0, 5]], K, ZERO_D, model="mei", xi=xi
    )
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(pixels, [[320, 240], [320, 240]])
    rays = pixel_to_ray([[320, 240]], K, ZERO_D, model="mei", xi=xi)
    np.testing.assert_allclose(rays, [[0, 0, 1]], atol=1e-12)


def test_mei_known_ray_with_brown_distortion():
    # Norm is exactly 3; xi=0.5 gives normalized coordinates (2/7, 4/7).
    x, y = 2 / 7, 4 / 7
    k1, k2, p1, p2 = distortion = [0.1, -0.02, 0.003, -0.004]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    pixels, valid = project_points(
        [[1, 2, 2], [3, 6, 6]], K, distortion, model="mei", xi=0.5
    )
    assert valid.all()
    expected = [100 * xd + 320, 120 * yd + 240]
    np.testing.assert_allclose(pixels, [expected, expected], atol=1e-10)


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("distortion", [ZERO_D, [0.01, -0.001, 0.0002, -0.0003]])
def test_mei_inverse_roundtrip(xi, distortion):
    points = np.array([[0, 0, 2], [0.3, -0.2, 1], [-0.4, 0.5, 1.5]])
    pixels, valid = project_points(points, K, distortion, model="mei", xi=xi)
    assert valid.all()
    rays = pixel_to_ray(pixels, K, distortion, model="mei", xi=xi)
    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(rays, expected, atol=1e-7)
    reprojection, valid = project_points(rays, K, distortion, model="mei", xi=xi)
    assert valid.all()
    np.testing.assert_allclose(reprojection, pixels, atol=1e-5)


def test_mei_xi_one_sees_beyond_ninety_degrees():
    points = np.array([[np.sqrt(3), 0, -1], [1, 0, 0]])
    pixels, valid = project_points(points, K, ZERO_D, model="mei", xi=1.0)
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(
        pixels, [[320 + 100 * np.sqrt(3), 240], [420, 240]], atol=1e-10
    )
    rays = pixel_to_ray(pixels, K, ZERO_D, model="mei", xi=1.0)
    np.testing.assert_allclose(
        rays, points / np.linalg.norm(points, axis=1, keepdims=True), atol=1e-10
    )
    assert rays[0, 2] < 0


@pytest.mark.parametrize(
    ("xi", "outside"),
    [(0.0, [1, 0, -1]), (0.5, [0.1, 0, -1]), (1.0, [0, 0, -1])],
)
def test_mei_outside_projection_domain_is_invalid(xi, outside):
    points = [
        [0, 0, 1],
        outside,
        [0, 0, 0],
        [np.nan, 0, 1],
        [0, np.inf, 1],
    ]
    pixels, valid = project_points(points, K, ZERO_D, model="mei", xi=xi)
    assert valid.tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(pixels[0], [320, 240])
    assert np.isnan(pixels[1:]).all()


@pytest.mark.parametrize("xi", [0.0, 0.5, 1.0, 1.5])
def test_mei_camera_accepts_finite_nonnegative_xi(xi):
    camera = CameraConfig.model_validate(camera_payload(xi=xi))
    assert camera.model == "mei"
    assert camera.xi == xi
    assert len(camera.D) == 4


def test_mei_camera_requires_xi():
    payload = camera_payload()
    del payload["xi"]
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(payload)


@pytest.mark.parametrize("xi", [None, -0.001, float("nan"), float("inf"), -float("inf")])
def test_mei_camera_rejects_invalid_xi(xi):
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(camera_payload(xi=xi))


@pytest.mark.parametrize("length", [0, 1, 3, 5, 8])
def test_mei_camera_requires_four_distortion_coefficients(length):
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(camera_payload(D=[0.0] * length))


@pytest.mark.parametrize("model", ["pinhole", "opencv_pinhole", "opencv_fisheye"])
def test_non_mei_camera_rejects_xi(model):
    with pytest.raises(ValidationError):
        CameraConfig.model_validate(camera_payload(model=model, xi=0.5))
