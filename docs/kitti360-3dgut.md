# KITTI-360 3DGUT reconstruction

LiDAR-seeded 3D Gaussian reconstruction of a 500 m KITTI-360 road segment from two pinhole and two
fisheye cameras. Best result so far: held-out PSNR 18.2 dB (pinhole) and 16.0 dB (fisheye), trained
in about 4 minutes on a Colab T4. Geometry and poses are verified; image quality is still blurry.

## Pipeline

1. **Data.** Sequence `2013_05_28_drive_0003_sync` (any KITTI-360 sequence via `--sequence`). The results
   below use the first 100 frames with poses, frames 12 to 520 (about 500 m). This sequence has gaps in its
   metadata (12-137, 157, then 167 onward), so those frames are not strictly consecutive; the trainer now
   defaults to a contiguous run (`--max-gap 5`, which would pick 167-654 here) and `--max-gap 1000000`
   reproduces the 12-520 choice. `val.pkl` holds frames 2-282 of this sequence and `train.pkl` the rest; both
   are merged. The exact frames are saved in `config.json` and the fly-through renderer reads them. Images are
   trained at half resolution (pinhole 704x188, fisheye 700x700). Every 10th frame is held out (40 views), 360
   views train.
2. **Initialisation.** Each LiDAR point becomes a Gaussian in world coordinates (origin at the first
   frame, z up). Its colour is sampled from the camera that saw it in the same frame; points no camera
   saw are dropped. Up to 120k points; initial scale = mean distance to 3 neighbours.
3. **Sky (optional, on by default).** About 26k large Gaussians on a 200 m dome, coloured from the
   images. LiDAR never reaches the sky, so without this the sky has no Gaussians at all.
4. **Rendering.** gsplat 1.5.3 with `with_ut=True`, `with_eval3d=True` (3DGUT): Gaussians are evaluated
   in 3D through the real camera model, so fisheye is rendered correctly. The Mei fisheye model is fitted
   by gsplat's polynomial radial model (error under 0.002 px).
5. **Optimisation.** `SelectiveAdam` updates only Gaussians visible in the current view (Fast3DGS
   style). Position learning rate = 1.6e-4 x `scene_scale` (capped at 20 m), decayed 100x.
6. **Densification.** gsplat `MCMCStrategy`: from step 500, every 100 steps, relocate Gaussians with
   opacity below 0.005 and grow by 5% up to the cap (1M). `DefaultStrategy` cannot be used: 3DGUT
   produces no 2D gradients.
7. **Loss.** L1 over the supervised region (fisheye car hood masked out), optional 0.2 x (1 - SSIM),
   plus 0.01 x mean opacity and 0.01 x mean scale.
8. **Outputs.** `gaussians.ply`, `compare_image_0X.png`, `metrics.json` (PSNR full frame and masked),
   `config.json`, checkpoints every 2,500 steps, optional sync to a Drive folder and resume.

Code: `tools/colab_cli_kitti360.py` (trainer), `tools/render_flythrough.py` (video),
`tools/kitti360_projection_check.py` (geometry check). Run them through [`fflab`](../fflab/README.md)
or the notebook `tools/kitti360_3dgut_fast3dgs_colab.ipynb`.

## How to run

```bash
# Colab T4 from any Linux/macOS/WSL shell (needs google-colab-cli)
node fflab/bin/fflab.js colab train --gpu T4 --accel fused-ssim --run-name road1
node fflab/bin/fflab.js colab render --run-name road1
node fflab/bin/fflab.js colab stop

# Local NVIDIA GPU
node fflab/bin/fflab.js setup --accel fused-ssim
node fflab/bin/fflab.js train --accel fused-ssim --run-name road1
```

Only sequence 0003 is needed (about 1.7 GB: four camera tars and one LiDAR tar). Missing calibration,
metadata and `data_poses.zip` are downloaded; missing tars are streamed from the
[Hugging Face mirror](https://huggingface.co/datasets/danny042/KITTI-360-fisheye) and read only up to
the frames needed. The first run on a fresh VM compiles gsplat's CUDA kernels once (about 13-15 min).

## Results

### Geometry check

`fflab check` projects one LiDAR scan into all four cameras three ways: official KITTI-360 calibration,
the trainer's metadata, and frame 12 lifted to world and re-projected into frame 17. All agree
(pinhole difference 0.0005 px; overlays align with buildings, poles, cars, horizon). The trainer's
pose convention `cam0_to_world @ lidar2cam` is correct. Fisheye loses points beyond 90 degrees because
gsplat limits the angle; KITTI-360 fisheye covers more than 180 degrees.

### Ablation (100 frames, 10k steps, 1M cap, T4, held-out PSNR on the supervised region)

| Configuration | image_00 | image_01 | image_02 (fisheye) | image_03 (fisheye) | Train time |
|---|---|---|---|---|---|
| Base | 13.4 | 13.6 | 11.5 | 11.0 | 199 s |
| + hood mask | 13.2 | 13.6 | 11.5 | 11.1 | 197 s |
| + SSIM (conv2d) | 14.1 | 14.1 | 11.3 | 11.2 | 482 s |
| + sky dome | 17.6 | 17.6 | 16.2 | 16.0 | 487 s |
| sky + **fused-ssim** | **18.2** | **18.2** | **16.0** | **16.0** | **215 s** |

- The sky dome is the one change that clearly matters (+3.5 dB pinhole, +4.9 dB fisheye).
- The hood mask does not move PSNR; the car hood is not why fisheye scores low.
- SSIM adds 0.5-0.9 dB on pinhole. The conv2d version made training about 2.4x slower; the
  `fused-ssim` CUDA kernel removes almost all of that cost.
- Each configuration ran once (fused-ssim on a different T4 VM), so differences under about 1 dB are
  not established. Train time excludes the one-off gsplat compile.

Earlier prototype runs (no image colours, no densification, 15k Gaussians, 6-30 frames) scored
6.7-14.9 dB with blob-like renders; see `context.md`.

## Limitations

- Renders are blurry, especially road sides and foliage; a car on the left smears into a dark streak
  (not yet checked whether it is moving).
- The sky dome is fake geometry at 200 m: right from the driven path, wrong for large viewpoint changes.
- Held-out frames sit between training frames (0.5 s apart), so PSNR measures interpolation, not
  extrapolation to new viewpoints.
- Fisheye pixels beyond 90 degrees are not rendered.
- The trainer reads only sequence 0003.

## Next steps

1. Repeat the best configuration with 3 seeds to measure run-to-run noise.
2. Hold out a contiguous segment to measure real novel-view quality.
3. Better sky: segmentation plus a learned background instead of the dome.
4. Dynamic objects: identify moving vehicles and mask or model them.
5. Full-resolution training (memory permitting on a T4).
6. Kaggle as a fallback runtime; hook `fflab` into the local Web UI.

## Pitfalls met (and handled in fflab)

- `colab exec --env` persists variables in the kernel; a leaked `KITTI360_MASK_ONLY=1` once made four
  runs exit early. fflab writes every option into the job launcher explicitly.
- Long `colab exec` calls drop their connection; fflab runs jobs detached and polls the log.
- Colab's upload API returns HTTP 500 if the target directory does not exist; fflab creates it first.
- Headless hosts have no OS keyring: install `google-colab-cli --with keyrings.alt` and set
  `PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring` (the token is then stored in plain text).
- With the full trajectory as `scene_scale` (368 m), the position learning rate was 0.06 and MCMC/split
  thresholds were metres; it is capped at 20 m.
