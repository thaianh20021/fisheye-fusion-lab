# Project Context

Updated: 2026-11-13 (Asia/Saigon)

## Goal

Build a local-first Fisheye Fusion Lab supporting:

- Pinhole cameras.
- Single fisheye.
- Two/four-camera fisheye fusion.
- LiDAR-assisted Gaussian reconstruction.
- Native Windows/Linux execution.
- Optional Colab/Kaggle T4 training.

NeRF methods are currently out of scope. Main research direction is 3D Gaussian Splatting, especially 3DGUT, Fast3DGS-style optimization, SplatAD, Instant NuRec, and fisheye camera models.

## Current Local Project

Workspace:

`C:\Users\ADMIN\Documents\Codex\2026-11-02\reconstructing-real-world-driving-scenes-using`

Important files:

- `README.md`: project overview and local CLI.
- `research_summary.md`: earlier Vietnamese research summary.
- `bao_cao_chi_tiet.md`: detailed research report.
- `tools/colab_cli_kitti360.py`: Colab trainer.
- `tools/colab_download_kitti360_full.py`: resumable KITTI-360 downloader.
- `tools/download_kitti_poses.py`: official pose archive downloader.
- `tools/kitti360_3dgut_fast3dgs_colab.ipynb`: repeatable Colab notebook.
- `tools/prepare_small_scene.py`: small local static scene preparation.
- `tools/train_small_gs.py`: local small Gaussian experiment.

## Colab Dataset

Full archive target:

`/content/drive/MyDrive/datasets/kitti360_full`

The archive contains the KITTI-360 camera/LiDAR tar files, calibration, metadata, and:

`data_poses.zip`

The pose archive contains official files such as:

`2013_05_28_drive_0003_sync/cam0_to_world.txt`

The notebook checks every required file. Existing files are skipped. Missing files are downloaded. Partial files use `.part` and resume when possible.

## Colab Pipeline

Current trainer:

- Sequence: `2013_05_28_drive_0003_sync`.
- Cameras: `image_00`, `image_01`, `image_02`, `image_03`.
- `image_00`/`image_01`: pinhole.
- `image_02`/`image_03`: fisheye.
- LiDAR: real Velodyne scans.
- Pose: official KITTI-360 `cam0_to_world.txt`.
- Renderer: `gsplat` with `with_ut=True`, `with_eval3d=True`, `packed=False`.
- Optimizer: `SelectiveAdam`.
- Default test: 30 frames, 15,000 Gaussians, 10,000 steps.
- Output: `/content/kitti360_cli_out`.

Environment overrides:

```python
os.environ["KITTI360_N_FRAMES"] = "10"
os.environ["KITTI360_STEPS"] = "1000"
```

## Verified Runs

Earlier verified runs:

- 6 frames, 3,000 steps, ICP poses: held-out fisheye PSNR about `14.88 dB`; geometry had large blobs because scans were merged in local coordinates and scales exploded.
- Fixed 6-frame run: transformed LiDAR into world coordinates and clamped scales to `0.005..0.30 m`.
- 30 frames, official poses, 10,000 steps: held-out fisheye PSNR `8.62 dB`.
- 10 frames, official poses, 10,000 steps: held-out fisheye PSNR `6.67 dB`.
- Latest 30-frame run after pose convention correction: held-out fisheye PSNR `7.35 dB`; render was no longer black but remained blob-like.

Downloaded local artifacts:

- `outputs/compare_official_30f.png`
- `outputs/gaussians_official_30f.ply`
- `outputs/complete_10f_compare.png`
- `outputs/complete_10f_gaussians.ply`
- `outputs/road_segment_30f_compare.png`
- `outputs/road_segment_30f.ply`

## Known Problems

1. The minimal trainer does not initialize Gaussian colors from projected images.
2. It has no real 3DGS densification/pruning schedule.
3. It optimizes a low-level prototype, not a production 3DGUT/Fast3DGS implementation.
4. Fisheye Mei calibration is approximated by a polynomial radial model.
5. Pose/camera convention is not yet validated with a projection overlay.
6. Six to thirty frames and 15,000 Gaussians are insufficient for a clean road segment.
7. Colab CLI sessions occasionally disappear with `404/401` or return `Service Unavailable`.
8. The latest desired front-road rerun was blocked because the T4 session was lost while restarting.

## Projection Check Result (2026-11-13, seq 0003, frame 12 -> 17)

Run on a Colab CPU VM with `tools/kitti360_projection_check.py` (it downloads calibration/pkl/poses and streams the needed frames from Hugging Face, so Drive is not required).

- Pinhole `image_00`/`image_01`: LiDAR overlay matches building, poles and car. Trainer chain B equals official chain A (median 0.0005 px).
- Multi-frame world chain C (frame 12 LiDAR re-projected into frame 17) is identical for the trainer pose chain and the documented `cam0_to_world @ rect0_from_velo` chain, and the overlay stays aligned. The pose convention is correct.
- Fisheye `image_02`/`image_03`: the LiDAR band lines up with the horizon and ground. The equidistant fit used by the trainer differs from exact Mei by under 0.002 px. The official chain draws extra points beyond 90 degrees; the trainer drops them (theta < pi/2) because the fisheye FOV exceeds 180 degrees.
- Edge-ratio is not reliable for fisheye (black mask, near-uniform depth rows); judge fisheye by the overlays.
- Drive for the account used in the check has only `datasets/kitti360_archive/calibration.zip`, not `kitti360_full`.

Conclusion: geometry is not the cause of low PSNR. Known problems 1-3 and 6 (colors, densification, Gaussian count) are the next work.

## Next Technical Gate

Do not increase training steps yet. First run a projection sanity check:

1. Load one official pose.
2. Transform LiDAR into the front camera.
3. Project points with the front camera intrinsic.
4. Overlay projected points on the real image.
5. Verify road/building alignment.

After alignment is correct:

1. Initialize Gaussian colors from visible image pixels.
2. Use 30-100 frames.
3. Use 50k-100k initial Gaussians.
4. Add densification/pruning.
5. Train 10k-30k steps.
6. Export `compare_road.png` and PLY.

## Commands

Local smoke tests:

```powershell
python -m py_compile tools\colab_cli_kitti360.py
python -m json.tool tools\kitti360_3dgut_fast3dgs_colab.ipynb > $null
```

Colab CLI:

```powershell
colab new --session kitti360 --gpu T4
colab exec -s kitti360 --timeout 1800 --env KITTI360_N_FRAMES=30 --env KITTI360_STEPS=10000 -f tools/colab_cli_kitti360.py
```

Projection sanity check (run this before the next train):

```powershell
colab exec -s kitti360 --timeout 900 -f tools/kitti360_projection_check.py
colab download -s kitti360 /content/kitti360_projection_check/projection_report.json outputs/projection_report.json
```

It writes `overlay_image_0X.png` and `projection_report.json` to `/content/kitti360_projection_check`.
Chains: A = official devkit calibration (reference), B = trainer's pkl `lidar2cam`/intrinsics,
C = frame F LiDAR lifted to world with the trainer pose chain and re-projected into frame F+5.
Edge ratio near 1.0 means misaligned; above about 1.3 means aligned. B/C must match A before training.

Do not use `!python` in PowerShell. That syntax belongs inside a notebook cell.
