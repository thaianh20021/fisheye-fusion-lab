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

- Local Web UI and CLI in the project.
- Small local Gaussian experiment.
- Full KITTI-360 archive workflow on Google Drive.
- Colab downloader that skips existing files and resumes partial files.
- Colab notebook:

`tools/kitti360_3dgut_fast3dgs_colab.ipynb`

- Colab trainer:

`tools/colab_cli_kitti360.py`

## Notebook Workflow

1. Mount Google Drive.
2. Check and download missing KITTI-360 files.
3. Check `data_poses.zip`.
4. Configure frame count and training steps.
5. Train four cameras: two pinhole and two fisheye.
6. Export PLY and comparison image.

Recommended first run:

```python
os.environ["KITTI360_N_FRAMES"] = "10"
os.environ["KITTI360_STEPS"] = "1000"
```

Research run:

```python
os.environ["KITTI360_N_FRAMES"] = "30"
os.environ["KITTI360_STEPS"] = "10000"
```

## What Worked

- Full archive stored on Drive.
- Real LiDAR initialization works.
- Official KITTI-360 pose archive downloads and parses.
- `gsplat==1.5.3` works with `packed=False`, `with_ut=True`.
- Training and PLY export complete on T4 for small subsets.
- Original giant-blur issue was fixed by transforming every LiDAR scan into world coordinates and clamping Gaussian scales.

## What Does Not Work Well Yet

The current minimal trainer can complete training but does not yet produce a clean, realistic road segment. Renders are sparse/blob-like, and one pose convention version produced an almost black result.

Root causes likely include:

- Camera/LiDAR/world transform convention still needs projection-overlay validation.
- Gaussian colors start from zero rather than image-derived values.
- No densification or pruning.
- Too few Gaussians for a road scene.
- Approximate fisheye radial conversion.

## Required Next Step

Implement a projection diagnostic before another long train:

- Project official-pose LiDAR into `CAM_FRONT`.
- Save an overlay against the real front image.
- Confirm the projected road and buildings align.

Only after that should the trainer be upgraded to:

- image-based color initialization,
- 50k-100k Gaussians,
- densification/pruning,
- 30-100 frames,
- 10k-30k steps.

## Current Status

As of 2026-11-13:

- Notebook exists and includes check/download logic.
- Full dataset remains on Google Drive.
- Latest T4 restart was blocked by Colab `Service Unavailable`; no dataset loss.
- The project is at the geometry-validation stage, not yet at final-quality reconstruction.
