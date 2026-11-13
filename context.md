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

Workspace (user's machine):

`C:\Users\ADMIN\Documents\Codex\2026-11-02\reconstructing-real-world-driving-scenes-using`

GitHub: `thaianh20021/fisheye-fusion-lab`, branch `main`.

Important files:

- `README.md`: project overview, quick start, layout.
- `docs/kitti360-3dgut.md`: KITTI-360 3DGUT pipeline, results, limitations, next steps.
- `fflab/`: Node.js CLI (setup, train, render, check, Colab runner); `fflab/README.md` (Vietnamese).
- `tools/colab_cli_kitti360.py`: trainer (env-driven; MCMC, mask, SSIM, sky, sync, resume).
- `tools/render_flythrough.py`: fly-through video from a PLY.
- `tools/kitti360_projection_check.py`: LiDAR-to-camera projection check.
- `tools/kitti360_3dgut_fast3dgs_colab.ipynb`: notebook version of the same pipeline.
- `tools/colab_download_kitti360_full.py`, `tools/download_kitti_poses.py`: dataset downloaders.
- `tools/prepare_small_scene.py`, `tools/train_small_gs.py`: earlier small local experiment.
- `research_summary.md`, `bao_cao_chi_tiet.md`: research reports (Vietnamese).

## Colab Dataset

Only sequence `2013_05_28_drive_0003_sync` is needed (about 1.7 GB: 4 camera tars + 1 LiDAR tar),
plus calibration, `ImageSets-rosa-6class/kitti360_rosa_infos_{val,train}.pkl` and `data_poses.zip`.
The trainer downloads missing small files and streams missing tars from the Hugging Face mirror,
reading only up to the frames it needs, so a Drive copy is optional.

Drive copy (second Colab account): `MyDrive/datasets/kitti360_full`; run outputs in
`MyDrive/datasets/kitti360_fast3dgut_runs/<run>`. The first account's Drive held only `calibration.zip`.

## Colab Pipeline

Current trainer (`tools/colab_cli_kitti360.py`, defaults as used by `fflab`):

- Sequence 0003, 100 consecutive frames (12-520, every 5th raw frame), val + train metadata merged.
- Cameras `image_00`/`image_01` pinhole, `image_02`/`image_03` fisheye (Mei, fitted polynomial).
- Gaussians seeded from real LiDAR with colours from the observing camera; optional sky dome.
- gsplat 1.5.3 `with_ut=True`, `with_eval3d=True`, `packed=False`; `SelectiveAdam`; `MCMCStrategy`
  (cap 1M); `scene_scale` capped at 20 m.
- Loss: masked L1 + optional 0.2 (1 - SSIM) (`KITTI360_SSIM_IMPL` = torch or fused) + opacity/scale reg.
- Every 10th frame held out; PSNR reported full-frame and on the supervised region.
- Checkpoints every 2,500 steps, `KITTI360_SYNC_DIR` copy, `KITTI360_RESUME=1`.

## Earlier Prototype Runs

Prototype trainer (before image colours, densification and the projection check):

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

Resolved:

1. Gaussian colours now come from the observing camera.
2. Densification: gsplat `MCMCStrategy` (DefaultStrategy is incompatible with 3DGUT).
3. Pose/camera convention validated with the projection check.
4. 100 frames and up to 1M Gaussians instead of 6-30 frames and 15k.
5. Colab session loss: `fflab` runs jobs detached, polls logs, checkpoints to Drive and resumes.

Still open:

1. Renders are blurry; a car on the left smears (moving or parked not yet checked).
2. Sky is a fake 200 m dome; wrong for large viewpoint changes.
3. Held-out frames are interleaved with training frames: PSNR measures interpolation only.
4. Fisheye beyond 90 degrees is not rendered (gsplat limit).
5. Each configuration ran once; run-to-run noise unknown.
6. Colab T4 allocation can fail with `Service Unavailable`; switching account worked.

## Projection Check Result (2026-11-13, seq 0003, frame 12 -> 17)

Run on a Colab CPU VM with `tools/kitti360_projection_check.py` (it downloads calibration/pkl/poses and streams the needed frames from Hugging Face, so Drive is not required).

- Pinhole `image_00`/`image_01`: LiDAR overlay matches building, poles and car. Trainer chain B equals official chain A (median 0.0005 px).
- Multi-frame world chain C (frame 12 LiDAR re-projected into frame 17) is identical for the trainer pose chain and the documented `cam0_to_world @ rect0_from_velo` chain, and the overlay stays aligned. The pose convention is correct.
- Fisheye `image_02`/`image_03`: the LiDAR band lines up with the horizon and ground. The equidistant fit used by the trainer differs from exact Mei by under 0.002 px. The official chain draws extra points beyond 90 degrees; the trainer drops them (theta < pi/2) because the fisheye FOV exceeds 180 degrees.
- Edge-ratio is not reliable for fisheye (black mask, near-uniform depth rows); judge fisheye by the overlays.
- Drive for the account used in the check has only `datasets/kitti360_archive/calibration.zip`, not `kitti360_full`.

Conclusion: geometry is not the cause of low PSNR. Known problems 1-3 and 6 (colors, densification, Gaussian count) are the next work.

## Ablation Results (2026-11-13, seq 0003, 100 frames, 10k steps, 1M Gaussian cap, T4)

Held-out PSNR (every 10th frame held out; fisheye measured outside the car hood and circle mask).
Each configuration was run once; differences under about 1 dB are not established.

| Camera | Base | + mask | + SSIM | + sky |
|---|---|---|---|---|
| image_00 | 13.4 | 13.2 | 14.1 | 17.6 |
| image_01 | 13.6 | 13.6 | 14.1 | 17.6 |
| image_02 | 11.5 | 11.5 | 11.3 | 16.2 |
| image_03 | 11.0 | 11.1 | 11.2 | 16.0 |

- The sky dome (large Gaussians seeded on a 200 m dome with colours from the images) is the one change that clearly matters.
- The hood mask did not move PSNR. SSIM helps pinhole a little; the conv2d SSIM makes training about 2.4x slower (482 s vs 197 s), fused-ssim removes most of that.
- Densification: 3DGUT (`with_eval3d`) gives no means2d gradients, so gsplat `DefaultStrategy` crashes; `MCMCStrategy` is used.
- Position lr and split thresholds scale with `scene_scale`; it is capped at 20 m (a 500 m trajectory would give lr 0.06).
- Remaining defects in the fly-through: blur on the sides, a parked or moving car smeared on the left.
- Outputs per run on Drive: `MyDrive/datasets/kitti360_fast3dgut_runs/<run>/` (gaussians.ply, ckpt.pt, compare_*.png, metrics.json, config.json, fly.mp4).
- Trainer env: `KITTI360_MASK`, `KITTI360_SSIM`, `KITTI360_SKY`, `KITTI360_CAP_MAX`, `KITTI360_SYNC_DIR`, `KITTI360_RUN_NAME`, `KITTI360_RESUME=1`.
- Colab CLI on a headless host: install `google-colab-cli --with keyrings.alt` and set `PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring`; `colab exec --env` persists in the kernel, so reset flags such as `KITTI360_MASK_ONLY` explicitly.

## fflab CLI and fused-ssim (2026-11-13)

`fflab/` is a Node.js CLI (no npm dependencies, Node 18.3+) that wraps the Python tools: `list`, `doctor`, `setup`, `train`, `render`, `check`, and `colab train|render|check|logs|stop`. Backends and accelerators are JSON manifests in `fflab/backends` and `fflab/accelerators`. See `fflab/README.md`.

First real T4 run through `fflab colab train --accel fused-ssim` (100 frames, 10k steps, 1M cap, mask+SSIM+sky):

| Run | Train time | Held-out PSNR masked (image_00/01/02/03) |
|---|---|---|
| abl3, conv2d SSIM | 487 s | 17.6 / 17.6 / 16.2 / 16.0 |
| fused-ssim | 215 s (1008 s minus 793 s gsplat compile) | 18.2 / 18.2 / 16.0 / 16.0 |

One run each, on different T4 VMs.

## Next Steps

1. Repeat the best configuration (sky + mask + fused-ssim) with 3 seeds to measure noise.
2. Hold out a contiguous segment for real novel-view evaluation.
3. Better sky: segmentation + learned background.
4. Identify and mask or model moving vehicles.
5. Full-resolution training on T4 if memory allows.
6. Kaggle fallback; connect `fflab` to the local Web UI.

## Commands

```bash
cd fflab && npm test                       # fflab unit + integration tests
python -m pytest -q                        # Python tests
node fflab/bin/fflab.js doctor
node fflab/bin/fflab.js colab check --session fflab-test          # CPU VM is enough
node fflab/bin/fflab.js colab train --gpu T4 --accel fused-ssim --run-name road1
node fflab/bin/fflab.js colab render --run-name road1
node fflab/bin/fflab.js colab logs --run road1                    # re-attach
node fflab/bin/fflab.js colab stop --session fflab
```

Headless hosts: `uv tool install google-colab-cli --with keyrings.alt` and
`export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring`. `google-colab-cli` does not support
Windows; use WSL there. Do not use `!python` in PowerShell; that syntax belongs in a notebook cell.
