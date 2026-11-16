# Gaussian 3D reconstruction

## What is connected

The Local Web UI launches the official Instant NuRec inference CLI in a separate
Python environment. It tracks status/logs and exposes generated Gaussian PLY,
sky/dynamic sidecars, and optional calibrated `.render.png` files. No panorama
is converted to fake depth or labeled as a reconstructed scene.

This is **inference with pretrained weights**, not per-scene training. CPU
panorama fusion, KITTI-360 import, and Instant NuRec are separate workflows.
The canonical Fusion Lab manifest is not NCore and is rejected by this backend.
KITTI-360 sequences are converted to NCore by `tools/kitti360_to_ncore.py` (see
below); other fisheye rigs still need their own converter.

The upstream released profiles are front-camera or multiview (1/3/5 context
cameras). The renderer preview is on the source camera trajectory, not an
arbitrary free-camera novel view. A Gaussian-capable viewer is needed to
interactively inspect PLY; sky/dynamic sidecars are not displayed automatically
by ordinary PLY viewers.

## KITTI-360 -> NCore -> Instant NuRec -> PSNR

The start page of the Web UI (`/`) walks through this in four steps; the old page
is at `/legacy`. The same steps on the command line:

```bash
fflab convert --frames 40                      # outputs/ncore/kitti360_0003_000000_40f/
fflab nurec --clip outputs/ncore/kitti360_0003_000000_40f/kitti360_0003_000000_40f.json --gpu 1
fflab eval  --ply outputs/reconstruction/<run>/<id>/ply/<clip>/<clip>.ply \
            --clip outputs/ncore/kitti360_0003_000000_40f/kitti360_0003_000000_40f.json --gpu 1
```

One-time setup (Linux, CUDA GPU): clone NVIDIA/instant-nurec at revision
`f2ecafcdf28f2dc8f2cc17447a772571cde558be` into `backends/instant-nurec`, run
`uv sync` there (its own `.venv`, Python 3.11; it ships `nvidia-ncore`), accept
the terms of `nvidia/instant-nurec` on Hugging Face and place the checkpoints in
`backends/instant-nurec/checkpoints/pth/`. Evaluation renders with gsplat from
the `fflab setup` venv. `backends/` is git-ignored.

What the converter writes (`tools/kitti360_to_ncore.py --help`):

- 4 cameras, Velodyne LiDAR, rig poses (world = first frame), empty cuboids; 10 Hz
  timestamps synthesised from frame ids. The KITTI-360 archive used here keeps
  every 5th frame, so 40 frames are ~230 m of road.
- Pinhole `image_00/01` as OpenCV pinhole (principal point shifted by +0.5 px to
  NCore's pixel-corner convention).
- Fisheye `image_02/03`, `--fisheye-model`:
  - `rectify120` (default): resampled with the exact MEI projection into NVIDIA's
    120-degree F-theta cross camera (1920x1080), same pose. Instant NuRec turns
    every context camera into a pinhole from the ray angle at the image edges,
    which fails past 90 degrees (KITTI-360 fisheyes reach ~95), and the
    multiview model was trained on these cameras. The virtual camera also stays
    clear of the car body.
  - `ftheta` / `opencv`: the full fisheye as F-theta (fit < 0.6 px) or
    Kannala-Brandt (< 0.01 px) with an `ego` mask for the car body, for 3DGUT /
    gsplat; Instant NuRec rejects these.
- `tests/test_kitti360_ncore.py` checks all camera models against the MEI
  projection with NCore's own camera code; run it with the NuRec venv.

Measured on sequence 0003 (frames 167-358, 40 frames; PSNR over pixels covered
by Gaussians, sky excluded, on the input frames themselves, so this is not a
held-out score):

| model, context cameras | front left | front right | fisheye left | fisheye right |
| --- | --- | --- | --- | --- |
| pa-front, `front_left_00` (full fisheye clip) | 15.9 | 15.9 | 12.9 | 12.5 |
| pa-multiview, front + both fisheyes (`rectify120`) | 16.0 | 16.1 | 16.0 | 16.2 |

On 150 frames (~1.1 km) the 2 M-Gaussian cap spreads thinner: 15.9 / 15.9 / 15.5 / 13.7.

## Driving through a reconstruction with CARLA vehicles

Step 5 of the Web UI, or `fflab carla`. A virtual camera follows the clip's
recorded trajectory (optionally shifted sideways or up) through the Gaussian
scene; with a CARLA server on the same machine, CARLA vehicles are composited
in. Only localhost ports are used; no Docker, no root.

```bash
fflab carla-server --adapter 2          # or the Web UI's "Bật server" button (finds the adapter itself)
fflab carla --ply <scene.ply> --clip outputs/ncore/<clip>/<clip>.json --npc 3 --duration 10 --gpu 1
```

Per frame (`tools/carla_nurec_bridge.py`):

1. gsplat 3DGUT renders the Gaussians (RGB, alpha, expected depth); Instant
   NuRec's sky cubemap and per-camera colour affine fill the rest (3DGUT scenes
   carry their own sky dome).
2. CARLA, in synchronous mode with its map unloaded and the scene placed 300 m
   up, renders the same pinhole camera: RGB, semantic segmentation, depth.
   Vehicles follow the recorded road: same-direction traffic in the ego lane,
   oncoming traffic one lane to the left (KITTI-360 roads here have one lane
   each way).
3. Vehicle pixels replace the reconstruction unless it is clearly nearer
   (0.8 m margin; unknown depth never hides a vehicle), plus a soft contact
   shadow from each vehicle's CARLA bounding box.

Details that matter:

- Feed-forward reconstruction puts a sheet of floaters 2-6 m in front of the
  camera, ~1 m above the real road, and has no other near road. They stay in
  the picture but are removed from the occlusion depth inside the space the
  recording car drove through (empty by definition), so they cannot hide
  wheels. Vehicles stand on the real road, 1.65 m under KITTI-360 camera 0.
- CARLA needs `libvulkan.so.1`; `tools/carla_server.sh` unpacks the Ubuntu
  package into `$CARLA_ROOT/vk` with `apt-get download` if it is missing.
  Vulkan numbers GPUs differently from nvidia-smi; the Web UI tries adapters
  until the server lands on the requested GPU (by memory growth: a Vulkan
  process is not listed by nvidia-smi here) and remembers the mapping in
  `~/.fflab/carla.json`.
- After teleporting, CARLA needs a few ticks before vehicles render fully;
  the bridge settles 5 ticks before the first frame.
- 3DGUT scenes (`fflab train`) live in the KITTI-360 world shifted to camera 0
  of their first frame; the bridge maps them onto the clip's NCore world from
  `data_poses.zip` and the run's `config.json` (camera poses agree to 1e-12 m
  and 0.05 degrees on sequence 0003).
- Video is H.264 when `imageio-ffmpeg` is installed in the gsplat venv
  (plays in browsers and Google Drive), otherwise mp4v.

Measured (RTX 6000 Ada, sequence 0003, 1280x640): 0.08 s per frame with
CARLA; inserted vehicle centres within 0.6 px (median) of their CARLA masks.

## Local execution

Use only a trusted upstream checkout, Python interpreter, and checkpoint.
The backend runs code from the configured repository on your computer.

1. Prepare the upstream environment separately; do not install its dependencies
   into the Fusion Lab Python environment.
2. Put an existing NCore clip, matching checkpoint, and prepared checkout under
   `FUSION_WORKSPACE`, or run the app with that variable pointing at their common
   parent. This workflow does not download model weights or datasets.
3. Open the Gaussian 3D panel and enter repository, interpreter, checkpoint, and
   NCore JSON paths. Select the model matching the checkpoint.
4. Run preflight. Resolve missing assets/CUDA/dependencies before starting.
5. Start reconstruction. A single job runs at a time. Cancel if needed; partial
   outputs remain available on disk rather than being deleted.
6. Download artifacts after the run. A successful process must produce a nonempty
   PLY; a successful exit without outputs is not called a reconstruction.

The preview checkbox requires the upstream render extra, a CUDA toolchain, and
Ninja. Review the upstream third-party terms before installing the render extra.
Native Windows success is not established. The web tool itself runs on Windows;
the external backend may need Linux, WSL, Colab, or Kaggle.

## Colab / Kaggle

Download the Instant NuRec notebook from the Gaussian 3D panel, not the separate
CPU-preview export buttons. The notebook pins upstream revision:

`f2ecafcdf28f2dc8f2cc17447a772571cde558be`

Setup is explicitly opt-in because Python/CUDA dependencies use substantial disk
space. Supply an already accessible local NCore JSON and matching checkpoint.
No credentials are stored in the notebook. Do not paste HF or Google tokens into
the UI. Downloading this notebook does not upload any dataset.

Default execution limits are one time chunk and a merge target of 200,000
Gaussians. The target is **not a hard VRAM limit**: the model can produce more
Gaussians before merging. A chunk may still contain many frames; this does not
guarantee a 50 MiB input. T4 or a 6 GiB local GPU may run out of memory. Input size
restrictions on the CPU sampler do not automatically apply to an NCore clip.

Local inference requests offline Hugging Face/Transformers operation and a local
checkpoint override. These environment settings are not an OS network sandbox.
Use trusted, fully local NCore references; the external backend owns its parsing.

## Verification status

Unit/API tests simulate subprocesses. They verify command construction, limits,
failure/cancellation handling, containment, and notebook generation. They do not
prove reconstruction quality or GPU compatibility.

Real runs (RTX 6000 Ada, 2026-11-16): the NVIDIA demo clip with pa-front, and
KITTI-360 sequence 0003 clips of 40 and 150 frames with pa-front and
pa-multiview, through the Web UI API, `fflab nurec` and the runner directly; see
the table above.

## Upstream references

- https://github.com/NVIDIA/instant-nurec/tree/f2ecafcdf28f2dc8f2cc17447a772571cde558be
- https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/README.md
- https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/THIRD_PARTY_LICENSE.txt
