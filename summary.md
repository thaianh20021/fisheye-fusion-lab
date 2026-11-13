# Summary

## Objective

Develop a tool combining pinhole and multi-camera fisheye fusion, with LiDAR-assisted 3D Gaussian reconstruction. The tool should run natively on Windows/Linux and optionally train on Colab/Kaggle T4.

## Chosen Direction

- Primary reconstruction family: 3D Gaussian Splatting.
- Fisheye renderer baseline: 3DGUT-style unscented-transform rasterization.
- Optimizer experiment: `gsplat` `SelectiveAdam`, inspired by Fast3DGS.
- Main dataset: KITTI-360, because it provides pinhole cameras, fisheye cameras, LiDAR, calibration, and official poses.
- Local UI/tool direction: Fisheye Fusion Lab.

## Current Deliverables

- `fflab/`: Node.js CLI (no npm dependencies) to set up, train, render and check locally or on a
  Colab VM. Backends and accelerators are JSON manifests: `3dgut-gsplat`; `selective-adam` and
  `mcmc` built in; `fused-ssim` optional.
- `tools/colab_cli_kitti360.py`: KITTI-360 trainer (4 cameras + LiDAR, 3DGUT, SelectiveAdam, MCMC,
  sky dome, hood mask, SSIM, checkpoints, Drive sync, resume).
- `tools/render_flythrough.py`: fly-through video; `tools/kitti360_projection_check.py`: geometry check.
- `tools/kitti360_3dgut_fast3dgs_colab.ipynb`: notebook version (updated, not re-run end to end).
- Local Web UI and Python CLI (preview fusion, sampling, KITTI-360 import, small experiments).
- Documentation: `docs/kitti360-3dgut.md`, `fflab/README.md`, `context.md`.

## How to Run

```bash
node fflab/bin/fflab.js colab train --gpu T4 --accel fused-ssim --run-name road1
node fflab/bin/fflab.js colab render --run-name road1
node fflab/bin/fflab.js colab stop
```

Only sequence 0003 is needed (about 1.7 GB); missing files are downloaded or streamed.

## What Works

- Projection check: LiDAR overlays align on all four cameras; trainer poses match the official chain.
- 100-frame, 1M-Gaussian training on a Colab T4 in about 4 minutes (after a one-off 13 min compile).
- Best held-out PSNR: 18.2 dB pinhole, 16.0 dB fisheye (sky + mask + fused-ssim).
- Sky dome: +3.5 dB pinhole, +4.9 dB fisheye. fused-ssim: 215 s instead of 487 s, same quality.
- `fflab colab` verified end to end on a real T4 (train) and CPU VM (check).

## What Does Not Work Well Yet

- Renders are blurry; a car on the left smears.
- The sky is a fake dome at 200 m.
- PSNR measures interpolation between training frames, not new viewpoints.
- Fisheye beyond 90 degrees is not rendered.
- Single run per configuration; run-to-run noise unknown.
- `fflab train` on a local GPU and on Windows not yet tested.

## Next Steps

1. Three seeds of the best configuration to measure noise.
2. Contiguous held-out segment for novel-view evaluation.
3. Better sky (segmentation + learned background); handle moving vehicles.
4. Full-resolution training; Kaggle fallback; Web UI integration.

## Current Status

As of 2026-11-13: the KITTI-360 3DGUT pipeline and `fflab` are merged on `main`. The project has
moved from geometry validation to quality improvement.
