"""Drive a virtual camera through an Instant NuRec reconstruction, optionally with CARLA vehicles in it.

    <gsplat-python> tools/carla_nurec_bridge.py scene.ply traj.npz --out out_dir \
        [--sky scene.sky.npz] [--carla 127.0.0.1:2000] [--npc 3] [--lateral 0] [--fps 20]

traj.npz comes from tools/ncore_trajectory_export.py (NuRec venv). The ego camera follows the
recorded rig trajectory (spline + slerp), shifted --lateral metres to the left / --lift up for a
novel view. Each frame:
  1. gsplat (3DGUT, as Instant NuRec renders) draws the Gaussians: RGB, alpha, expected depth;
  2. the NuRec sky cubemap fills what the Gaussians leave open, then the camera's colour affine;
  3. with --carla: CARLA renders the same camera (RGB + semantic segmentation + depth) in synchronous
     mode, its vehicles are placed along the recorded road, and every vehicle pixel nearer than the
     reconstruction replaces it (depth test), so the reconstruction still hides cars behind it.
CARLA's own map is unloaded and the scene is placed --carla-z metres up, so only the vehicles show.
Only localhost ports are used (CARLA server on the same machine). Without --carla: neural video only.

Outputs: out_dir/composite.mp4 (or neural.mp4), out_dir/frames/*.png every --png-every frames,
out_dir/alignment.json (CARLA runs: projected vehicle centre vs its segmentation mask, pixels).
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

SH_C0 = 0.28209479177387814
VEHICLE_TAGS = (14, 15, 16, 18, 19)  # CARLA semantic tags: car, truck, bus, motorcycle, bicycle
FLIP_Y = np.diag([1.0, -1.0, 1.0])  # NCore world (right-handed, z up) <-> Unreal (left-handed, z up)


def log(msg):
    print(f"[bridge] {msg}", flush=True)


# ------------------------------------------------------------------ scene
def load_ply(path: Path, device: str):
    with path.open("rb") as f:
        props, n = [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[0] == "element" and parts[1] == "vertex":
                n = int(parts[2])
            elif parts[0] == "property":
                props.append((parts[2], {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}[parts[1]]))
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    col = lambda *k: torch.from_numpy(np.stack([data[x].astype(np.float32) for x in k], 1)).to(device)
    return dict(
        means=col("x", "y", "z"),
        quats=torch.nn.functional.normalize(col("rot_0", "rot_1", "rot_2", "rot_3"), dim=1),
        scales=torch.exp(col("scale_0", "scale_1", "scale_2")),
        opacities=torch.sigmoid(col("opacity")[:, 0]),
        colors=(col("f_dc_0", "f_dc_1", "f_dc_2") * SH_C0 + 0.5).clamp(0, 1),
    )


class Sky:
    """Instant NuRec sky sidecar: world-aligned cubemap + per-camera colour affine (README, sky.npz)."""

    def __init__(self, path: Path, camera: str, device: str):
        d = np.load(path, allow_pickle=False)
        self.faces = torch.from_numpy(d["sky_cubemap"].astype(np.float32)).permute(0, 3, 1, 2).to(device)
        ids = [str(s) for s in d["affine_sensor_ids"]]
        row = ids.index(camera) if camera in ids else 0
        self.affine = torch.from_numpy(d["affine_matrix"][row]).to(device)
        log(f"sky: {d['sky_source']}, affine of {ids[row]}")

    def sample(self, dirs: torch.Tensor) -> torch.Tensor:
        """dirs (H, W, 3) world directions -> (H, W, 3). Same face/uv mapping as instant_nurec.utils.cubemap."""
        flat = dirs.reshape(-1, 3)
        ax = flat.abs().argmax(-1)
        dom = flat.abs().gather(-1, ax[:, None])[:, 0].clamp_min(1e-12)
        pos = flat.gather(-1, ax[:, None])[:, 0] > 0
        face = torch.where(ax == 0, torch.where(pos, 0, 1), torch.where(ax == 1, torch.where(pos, 3, 2), torch.where(pos, 4, 5)))
        x, y, z = (flat / dom[:, None]).unbind(-1)
        u = torch.stack([-z, z, x, x, x, -x], -1).gather(-1, face[:, None])[:, 0]
        v = torch.stack([y, y, z, -z, y, y], -1).gather(-1, face[:, None])[:, 0]
        out = torch.zeros(len(flat), 3, device=flat.device)
        for i in range(6):
            m = face == i
            if m.any():
                grid = torch.stack([u[m], v[m]], -1)[None, None]
                out[m] = torch.nn.functional.grid_sample(self.faces[i : i + 1], grid, mode="bilinear",
                                                         padding_mode="border", align_corners=False)[0, :, 0].T
        return out.reshape(dirs.shape)

    def finish(self, fg, alpha, dirs):
        rgb = fg + (1 - alpha[..., None]) * self.sample(dirs)
        return (rgb @ self.affine[:, :3].T + self.affine[:, 3]).clamp(0, 1)


# ------------------------------------------------------------------ trajectory
class Trajectory:
    """Recorded rig poses as a smooth path, addressable by time and by arc length."""

    def __init__(self, traj: Path, camera: str):
        d = np.load(traj)
        P, ts = d["rig_to_world"], d["timestamps_us"].astype(np.float64) / 1e6
        self.t0, self.t1 = ts[0], ts[-1]
        self.pos = CubicSpline(ts, P[:, :3, 3])
        self.rot = Slerp(ts, Rotation.from_matrix(P[:, :3, :3]))
        self.T_rig_cam = d[f"T_rig_{camera}"]
        dense = np.linspace(ts[0], ts[-1], 4000)
        p = self.pos(dense)
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        self.dense_t = dense

    def rig(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.t0, self.t1))
        T = np.eye(4)
        T[:3, :3] = self.rot(t).as_matrix()
        T[:3, 3] = self.pos(t)
        return T

    def arc(self, t: float) -> float:
        return float(np.interp(t, self.dense_t, self.s))

    def at_arc(self, s: float) -> np.ndarray:
        return self.rig(float(np.interp(s, self.s, self.dense_t)))


def clear_ego_corridor(scene, traj: Trajectory, cam_height: float, half_width: float = 1.1, above: float = 0.6):
    """Delete Gaussians inside the space the recording car itself drove through: it is empty by definition.

    Feed-forward reconstruction puts a sheet of floaters a few metres in front of the camera, well above
    the real road (cam_height below the rig); it blurs the near road and would hide the wheels of
    inserted vehicles. Corridor: |lateral| < half_width, from 0.15 m above the road to `above` over the rig."""
    from scipy.spatial import cKDTree

    dense_t = np.linspace(traj.t0, traj.t1, max(200, int(traj.s[-1] / 0.25)))
    poses = np.stack([traj.rig(t) for t in dense_t])
    means = scene["means"].cpu().numpy().astype(np.float64)
    _, idx = cKDTree(poses[:, :3, 3]).query(means, k=1)
    T = poses[idx]
    local = np.einsum("nji,nj->ni", T[:, :3, :3], means - T[:, :3, 3])  # world -> rig (FLU) of the nearest pose
    inside = (np.abs(local[:, 0]) < 0.6) & (np.abs(local[:, 1]) < half_width) & \
             (local[:, 2] > -cam_height + 0.15) & (local[:, 2] < above)
    keep = torch.from_numpy(~inside).to(scene["means"].device)
    log(f"ego corridor: removed {int(inside.sum()):,} floater Gaussians of {len(means):,}")
    return {k: v[keep] for k, v in scene.items()}


def offset(T: np.ndarray, left=0.0, up=0.0, forward=0.0) -> np.ndarray:
    D = np.eye(4)
    D[:3, 3] = [forward, left, up]  # rig frame is FLU
    return T @ D


# ------------------------------------------------------------------ rendering
def intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / (2 * math.tan(math.radians(fov_deg) / 2))  # CARLA's pinhole: horizontal FOV, centred
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def render(scene, sky, c2w: np.ndarray, K: np.ndarray, w: int, h: int, device: str, depth_scene=None):
    """RGB from `scene`; depth (for occlusion) from `depth_scene` if given, e.g. without ego-corridor floaters."""
    viewmat = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))[None].to(device)
    Kt = torch.from_numpy(K)[None].to(device)
    common = dict(viewmats=viewmat, Ks=Kt, width=w, height=h, near_plane=0.2, packed=False, with_ut=True, with_eval3d=True)
    with torch.no_grad():
        out, alpha, _ = rasterization(**scene, **common)
        # 3DGUT/eval3d renders exactly 3 channels: a second pass with camera-space z as the colour
        # gives alpha-weighted depth, the "expected depth" of RGB+ED
        ds = depth_scene if depth_scene is not None else scene
        z = ds["means"] @ viewmat[0, 2, :3] + viewmat[0, 2, 3]
        zd, d_alpha, _ = rasterization(**{**ds, "colors": z[:, None].expand(-1, 3).contiguous()}, **common)
    rgb, alpha, d_alpha = out[0], alpha[0, ..., 0], d_alpha[0, ..., 0]
    depth = torch.where(d_alpha > 0.5, zd[0, ..., 0] / d_alpha.clamp_min(1e-6), torch.full_like(d_alpha, float("inf")))
    if sky is not None:
        ys, xs = torch.meshgrid(torch.arange(h, device=device) + 0.5, torch.arange(w, device=device) + 0.5, indexing="ij")
        cam = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        R = torch.from_numpy(c2w[:3, :3].astype(np.float32)).to(device)
        dirs = torch.nn.functional.normalize(cam @ R.T, dim=-1)
        rgb = sky.finish(rgb, alpha, dirs)
    else:
        rgb = rgb.clamp(0, 1)
    return rgb.cpu().numpy(), depth.cpu().numpy(), alpha.cpu().numpy()


# ------------------------------------------------------------------ CARLA
def ue_transform(carla, T: np.ndarray, z_offset: float):
    """A pose in NCore world (FLU axes, right-handed) -> carla.Transform (Unreal: left-handed, z up)."""
    R = FLIP_Y @ T[:3, :3] @ FLIP_Y
    f, r, u = R[:, 0], -R[:, 1], R[:, 2]  # forward, right (= -left), up in Unreal coordinates
    pitch = math.asin(np.clip(f[2], -1, 1))
    yaw = math.atan2(f[1], f[0])
    cp = math.cos(pitch)
    roll = math.atan2(-r[2] / cp, u[2] / cp)
    p = FLIP_Y @ T[:3, 3]
    return carla.Transform(carla.Location(float(p[0]), float(p[1]), float(p[2] + z_offset)),
                           carla.Rotation(pitch=math.degrees(pitch), yaw=math.degrees(yaw), roll=math.degrees(roll)))


class CarlaSide:
    def __init__(self, address: str, w: int, h: int, fov: float, fps: float, z_offset: float, n_npc: int, seed: int):
        import carla

        self.carla, self.z = carla, z_offset
        host, port = address.rsplit(":", 1)
        self.client = carla.Client(host, int(port))
        self.client.set_timeout(60.0)
        self.world = self.client.load_world("Town10HD_Opt", map_layers=carla.MapLayer.NONE)
        self.world.unload_map_layer(carla.MapLayer.All)
        s = self.world.get_settings()
        self.restore = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, 1.0 / fps
        self.world.apply_settings(s)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        lib = self.world.get_blueprint_library()
        far = carla.Transform(carla.Location(0, 0, z_offset + 50))
        self.sensors, self.queues = [], {}
        for kind in ("rgb", "semantic_segmentation", "depth"):
            bp = lib.find(f"sensor.camera.{kind}")
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", str(fov))
            if kind == "rgb":
                bp.set_attribute("motion_blur_intensity", "0")
            actor = self.world.spawn_actor(bp, far)
            q = queue.Queue()
            actor.listen(q.put)
            self.sensors.append(actor)
            self.queues[kind] = q
        rng = np.random.default_rng(seed)
        cars = [b for b in lib.filter("vehicle.*") if b.get_attribute("base_type").as_str() == "car"]
        self.npcs = []
        for i in range(n_npc):
            bp = cars[int(rng.integers(len(cars)))]
            if bp.has_attribute("color"):
                bp.set_attribute("color", bp.get_attribute("color").recommended_values[int(rng.integers(3))])
            a = self.world.spawn_actor(bp, carla.Transform(carla.Location(10 * i, 0, z_offset + 80)))
            a.set_simulate_physics(False)
            self.npcs.append(a)
        # footprint half-sizes (forward, left) for contact shadows, from CARLA's bounding boxes
        self.footprints = [(a.bounding_box.extent.x, a.bounding_box.extent.y) for a in self.npcs]
        log(f"CARLA {self.client.get_server_version()}: {len(self.npcs)} vehicles "
            f"({', '.join(a.type_id for a in self.npcs)}), sync {fps} Hz")
        self.world.tick()
        self._drain()

    def _drain(self):
        for q in self.queues.values():
            while not q.empty():
                q.get_nowait()

    def step(self, cam_T: np.ndarray, npc_T: list[np.ndarray]):
        for s in self.sensors:
            s.set_transform(ue_transform(self.carla, cam_T, self.z))
        for a, T in zip(self.npcs, npc_T):
            a.set_transform(ue_transform(self.carla, T, self.z))
        frame = self.world.tick()
        imgs = {}
        for kind, q in self.queues.items():
            while True:
                img = q.get(timeout=30)
                if img.frame >= frame:
                    break
            imgs[kind] = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
        rgb = imgs["rgb"][..., [2, 1, 0]].astype(np.float32) / 255
        tag = imgs["semantic_segmentation"][..., 2]
        d = imgs["depth"].astype(np.float64)
        depth = 1000.0 * (d[..., 2] + d[..., 1] * 256 + d[..., 0] * 65536) / (256**3 - 1)
        return rgb, np.isin(tag, VEHICLE_TAGS), depth

    def close(self):
        for a in self.sensors:
            a.stop()
        for a in self.sensors + self.npcs:
            a.destroy()
        self.world.apply_settings(self.restore)


# ------------------------------------------------------------------ main
def contact_shadow(rgb, show, npc_T, footprints, c2w, K, strength=0.45):
    """Darken the road under each inserted vehicle: its footprint, slightly enlarged, projected and feathered."""
    h, w = show.shape
    mask = np.zeros((h, w), np.float32)
    w2c = np.linalg.inv(c2w)
    for T, (hx, hy) in zip(npc_T, footprints):
        corners = [T @ np.array([sx * hx * 1.1, sy * hy * 1.15, 0.02, 1]) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        cam = np.array([(w2c @ c)[:3] for c in corners])
        if (cam[:, 2] < 0.5).any():
            continue
        uv = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
        cv2.fillConvexPoly(mask, np.round(uv).astype(np.int32), 1.0, lineType=cv2.LINE_AA)
    if not mask.any():
        return rgb
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, w / 300))
    mask[show] = 0
    return rgb * (1 - strength * mask[..., None])


def npc_plan(n: int, lane: float, ego_speed: float):
    """Per vehicle: (start gap ahead of the ego [m], own speed along the road [m/s, < 0 = oncoming], lateral [m, + = left])."""
    # KITTI-360 roads here have one lane each way and the ego drives in the right one:
    # same-direction traffic ahead in the ego lane, oncoming traffic one lane to the left
    base = [(14.0, ego_speed - 1.0, 0.0), (90.0, -ego_speed, lane), (45.0, ego_speed + 1.0, 0.0), (220.0, -ego_speed, lane)]
    return [base[i % len(base)] for i in range(n)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply")
    p.add_argument("traj")
    p.add_argument("--out", required=True)
    p.add_argument("--sky", default="", help="NuRec .sky.npz (default: next to the PLY)")
    p.add_argument("--camera", default="camera_front_left_00", help="camera whose mount and colour affine to use")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--fov", type=float, default=90.0, help="horizontal FOV, degrees")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--start", type=float, default=0.0, help="seconds into the clip")
    p.add_argument("--duration", type=float, default=0.0, help="seconds (0 = to the end)")
    p.add_argument("--lateral", type=float, default=0.0, help="shift the camera left (m); negative = right")
    p.add_argument("--lift", type=float, default=0.0, help="raise the camera (m)")
    p.add_argument("--carla", default="", help="host:port of a running CARLA server, e.g. 127.0.0.1:2000")
    p.add_argument("--npc", type=int, default=3)
    p.add_argument("--lane", type=float, default=3.5, help="lane width for neighbouring vehicles (m)")
    p.add_argument("--camera-height", type=float, default=1.65, help="rig origin above the real road (KITTI-360 cam 0, from calibration)")
    p.add_argument("--keep-corridor", action="store_true", help="let Gaussians in the space the recording car drove through hide inserted vehicles")
    p.add_argument("--occlusion-margin", type=float, default=0.8, help="the reconstruction hides a vehicle only if this much nearer (m): fuzzy road splats must not swallow wheels")
    p.add_argument("--carla-z", type=float, default=300.0, help="lift the scene in CARLA above its map (m)")
    p.add_argument("--png-every", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda"
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    ply = Path(a.ply)
    sky_path = Path(a.sky) if a.sky else ply.with_suffix("").with_suffix(".sky.npz")
    scene = load_ply(ply, dev)
    log(f"{len(scene['means']):,} Gaussians")
    sky = Sky(sky_path, a.camera, dev) if sky_path.is_file() else None
    traj = Trajectory(Path(a.traj), a.camera)
    # the floaters stay in the picture (they are the only near road there is) but may not hide vehicles
    occluders = scene if a.keep_corridor or not a.carla else clear_ego_corridor(scene, traj, a.camera_height)
    t_start = traj.t0 + a.start
    t_end = traj.t1 if a.duration <= 0 else min(traj.t1, t_start + a.duration)
    times = np.arange(t_start, t_end, 1.0 / a.fps)
    K = intrinsics(a.width, a.height, a.fov)
    log(f"{len(times)} frames, {traj.arc(t_end) - traj.arc(t_start):.0f} m, {a.width}x{a.height}, fov {a.fov}")

    sim = CarlaSide(a.carla, a.width, a.height, a.fov, a.fps, a.carla_z, a.npc, a.seed) if a.carla else None
    ego_speed = (traj.arc(t_end) - traj.arc(t_start)) / max(t_end - t_start, 1e-6)
    plan = npc_plan(a.npc, a.lane, ego_speed) if sim else []
    name = "composite.mp4" if sim else "neural.mp4"
    writer = cv2.VideoWriter(str(out / name), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
    align, t_wall = [], time.time()
    try:
        for i, t in enumerate(times):
            rig = offset(traj.rig(t), left=a.lateral, up=a.lift)
            c2w = rig @ traj.T_rig_cam
            rgb, depth, alpha = render(scene, sky, c2w, K, a.width, a.height, dev, depth_scene=occluders)
            if sim:
                s0, el = traj.arc(t_start), t - t_start
                npc_T = []
                for gap, speed, lat in plan:
                    s_npc = s0 + gap + speed * el
                    T = offset(traj.at_arc(s_npc), left=lat, up=-a.camera_height)
                    if speed < 0:
                        T = T @ np.diag([-1.0, -1.0, 1.0, 1.0])  # oncoming: turned 180 degrees about z
                    npc_T.append(T)
                c_rgb, veh, c_depth = sim.step(c2w @ np.linalg.inv(traj.T_rig_cam), npc_T)
                show = veh & (c_depth < depth + a.occlusion_margin)
                rgb = contact_shadow(rgb, show, npc_T, sim.footprints, c2w, K)
                rgb = np.where(show[..., None], c_rgb, rgb)
                for k, T in enumerate(npc_T):  # alignment: projected vehicle centre vs its mask
                    pc = np.linalg.inv(c2w) @ np.append(T[:3, 3] + T[:3, 2] * 0.7, 1)
                    if pc[2] > 2:
                        uv = K @ (pc[:3] / pc[2])
                        if 0 <= uv[0] < a.width and 0 <= uv[1] < a.height and veh.any():
                            ys, xs = np.nonzero(veh)
                            d = np.hypot(xs - uv[0], ys - uv[1]).min()
                            align.append({"frame": i, "npc": k, "distance_m": float(pc[2]), "px_to_nearest_vehicle_pixel": float(d)})
            frame = (rgb * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % a.png_every == 0:
                cv2.imwrite(str(out / "frames" / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i % 50 == 0:
                log(f"frame {i}/{len(times)} ({(time.time() - t_wall) / (i + 1):.2f} s/frame)")
    finally:
        writer.release()
        if sim:
            sim.close()
    if align:
        (out / "alignment.json").write_text(json.dumps(align))
        d = np.array([x["px_to_nearest_vehicle_pixel"] for x in align])
        log(f"alignment: vehicle centre within {np.percentile(d, 50):.1f} px (median), {np.percentile(d, 95):.1f} px (p95) of its mask")
    log(f"wrote {out / name}")


if __name__ == "__main__":
    main()
