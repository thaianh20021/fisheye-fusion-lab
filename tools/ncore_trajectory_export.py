"""Export an NCore clip's rig trajectory for tools/carla_nurec_bridge.py.

    <ncore-python> tools/ncore_trajectory_export.py clip.json --out traj.npz

Writes rig_to_world (N, 4, 4, NCore world: the frame Instant NuRec's PLY and sky cubemap live in),
timestamps_us (N,), and T_rig_<camera> (4, 4) for every camera, so the bridge (gsplat venv,
no `ncore` package) can place virtual cameras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from upath import UPath

from ncore.data.v4 import CameraSensorComponent, PosesComponent, SequenceComponentGroupsReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    meta = json.loads(Path(a.json).read_text())
    base = UPath(a.json).parent
    r = SequenceComponentGroupsReader([base / s["path"] for s in meta["component_stores"]])
    poses = r.open_component_readers(PosesComponent.Reader)["default"]
    rig_world, ts = poses.get_dynamic_pose("rig", "world")
    out = {"rig_to_world": np.asarray(rig_world, np.float64), "timestamps_us": np.asarray(ts, np.int64)}
    for cam_id in r.open_component_readers(CameraSensorComponent.Reader):
        out[f"T_rig_{cam_id}"] = np.asarray(poses.get_static_pose(cam_id, "rig"), np.float64)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(ts)} rig poses, cameras {[k[6:] for k in out if k.startswith('T_rig_')]}")


if __name__ == "__main__":
    main()
