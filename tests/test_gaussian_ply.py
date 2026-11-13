import struct

import numpy as np
import pytest

from tools.train_small_gs import save_ply


def test_binary_gaussian_ply_encoding(tmp_path):
    path = tmp_path / "scene.ply"
    save_ply(path, np.array([[1, 2, 3]]), np.array([[1, 0, 0, 0]]),
             np.array([[-2, -3, -4]]), np.array([0.]), np.array([[.5, .5, .5]]))
    header, body = path.read_bytes().split(b"end_header\n", 1)
    assert b"binary_little_endian" in header
    assert b"element vertex 1" in header
    assert b"property float scale_0" in header
    values = struct.unpack("<17f", body)
    assert values[:3] == (1, 2, 3)
    assert values[6:9] == (0, 0, 0)
    assert values[10:13] == (-2, -3, -4)
    assert values[13:] == (1, 0, 0, 0)
    with pytest.raises(FileExistsError):
        save_ply(path, np.array([[1, 2, 3]]), np.array([[1, 0, 0, 0]]),
                 np.array([[-2, -3, -4]]), np.array([0.]), np.array([[.5, .5, .5]]))
