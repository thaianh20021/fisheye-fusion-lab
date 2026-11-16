"""tools/carla_nurec_bridge.py pieces that need no CARLA server and no GPU.

Needs torch + scipy + gsplat (the fflab setup venv): ~/.fflab/venv/bin/python -m pytest tests/test_carla_bridge.py
"""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import carla_nurec_bridge as b  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def pose(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def test_trainer_world_maps_onto_the_clip_world(tmp_path):
    """Camera 0 of every frame, in the 3DGUT trainer's world, lands on the clip's camera after the transform."""
    seq, fids = "2013_05_28_drive_0003_sync", [167, 172, 177]
    cam0 = {f: pose([0.01 * i, 0.3 + 0.02 * i, -0.05], [1000 + 7 * i, 3000 - 2 * i, 115 + 0.1 * i]) for i, f in enumerate(fids)}
    with ZipFile(tmp_path / "data_poses.zip", "w") as z:
        z.writestr(f"{seq}/cam0_to_world.txt", "\n".join(" ".join([str(f)] + [repr(float(x)) for x in cam0[f].ravel()]) for f in fids))
    rig = {f: cam0[f] @ np.linalg.inv(b.R_CAM_TO_FLU) for f in fids}  # what tools/kitti360_to_ncore.py writes
    rig_world = np.stack([np.linalg.inv(rig[fids[0]]) @ rig[f] for f in fids])
    np.savez(tmp_path / "traj.npz", rig_to_world=rig_world, timestamps_us=np.arange(3) * 500000,
             frame_ids=np.array(fids), source_sequence=np.asarray(seq), T_rig_camera_front_left_00=b.R_CAM_TO_FLU)
    for first in (167, 172):  # trainer starting on the same frame, or a later one
        (tmp_path / "config.json").write_text(json.dumps({"sequence": seq, "frame_ids": [f for f in fids if f >= first]}))
        T = b.kitti_trainer_to_ncore(tmp_path / "config.json", tmp_path / "traj.npz", tmp_path)
        for i, f in enumerate(fids):
            trainer_cam = cam0[f].copy()
            trainer_cam[:3, 3] -= cam0[first][:3, 3]
            assert np.allclose(T @ trainer_cam, rig_world[i] @ b.R_CAM_TO_FLU, atol=1e-6)


def test_transform_scene_moves_points_and_orientations():
    T = pose([0.2, -0.4, 1.1], [3.0, -2.0, 0.5])
    q = Rotation.from_rotvec([[0.3, 0.1, -0.2], [0, 0, 1.0]]).as_quat()[:, [3, 0, 1, 2]]
    scene = {"means": torch.tensor([[1.0, 2.0, 3.0], [-4.0, 0.0, 1.0]]), "quats": torch.tensor(q, dtype=torch.float32)}
    out = b.transform_scene(scene, T)
    assert np.allclose(out["means"].numpy(), scene["means"].numpy() @ T[:3, :3].T + T[:3, 3], atol=1e-5)
    got = Rotation.from_quat(out["quats"].numpy()[:, [1, 2, 3, 0]]).as_matrix()
    want = T[:3, :3] @ Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    assert np.allclose(got, want, atol=1e-5)


def test_occlusion_never_hides_behind_unknown_depth():
    veh = np.ones((1, 3), bool)
    depth = np.array([[np.inf, 5.0, 30.0]])      # nothing / clearly nearer / farther
    c_depth = np.array([[12.0, 12.0, 12.0]])
    show = veh & ~(depth + 0.8 <= c_depth)
    assert show.tolist() == [[True, False, True]]
