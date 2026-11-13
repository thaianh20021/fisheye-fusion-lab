# Small-Data Workflow

Use a small sample first. Do not download a complete dataset just to validate the
pipeline, inspect calibration, or render a short preview.

## Rules

- Limit the sample by frame count and total size.
- Start with four frames and at most 50 MiB.
- Use `--dry-run` before writing files.
- Keep the sample outside the repository when possible.
- Do not download gated or full dataset archives for this workflow.
- No dataset is included or downloaded by this project.

## Recommended Datasets

### KITTI-360

Use KITTI-360 first for driving experiments. It provides a practical mix of
perspective cameras, fisheye cameras, poses, and LiDAR-related scene data.
Download only the sequence and sensor files needed for a short calibration or
rendering test. Preserve the original camera model and calibration metadata;
do not silently convert KITTI-360 fisheye parameters to OpenCV fisheye
parameters.

Suggested progression:

1. Four consecutive frames from one short segment.
2. The same frames from the selected camera pair or four-camera group.
3. A small LiDAR slice for overlay checks.
4. A longer segment only after validation succeeds.

### Newer College

Use Newer College to study exact four-fisheye plus LiDAR fusion. It is useful
for sensor synchronization, calibration, and projection experiments, but it is
robotic/handheld data rather than an automotive driving dataset. Start with
four synchronized frames and the corresponding LiDAR scans.

## Sampling Commands

The current sampling command supports a frame limit, a MiB limit, and a
non-writing dry run:

```powershell
python -m fisheye_fusion.cli sample `
  C:\data\kitti-360\manifest.json `
  --output C:\data\samples\kitti360-small `
  --max-frames 4 `
  --max-mib 50 `
  --dry-run
```

After reviewing the reported files and size, run the same command without
`--dry-run`:

```powershell
python -m fisheye_fusion.cli sample `
  C:\data\kitti-360\manifest.json `
  --output C:\data\samples\kitti360-small `
  --max-frames 4 `
  --max-mib 50
```

Newer College example:

```powershell
python -m fisheye_fusion.cli sample `
  C:\data\newer-college\manifest.json `
  --output C:\data\samples\newer-college-small `
  --max-frames 4 `
  --max-mib 50 `
  --dry-run
```

Do not remove `--dry-run` until the selected files and estimated size are
acceptable.

## Validate and Render

Once a sampled directory contains a manifest, validate it before rendering:

```powershell
python -m fisheye_fusion.cli validate C:\data\samples\kitti360-small\manifest.json
python -m fisheye_fusion.cli render `
  C:\data\samples\kitti360-small\manifest.json `
  --frame 0 `
  --output C:\data\samples\kitti360-small\preview.png
```

The preview is a CPU projection/fusion check. It is not a trained 3D
reconstruction and does not replace a backend-specific training run.

## Colab and Kaggle

Upload only the sampled directory or a compressed archive of it. Keep the
sample at four frames and 50 MiB for the first run, then increase one limit at
a time. Do not upload full KITTI-360 or Newer College archives to a T4
session before the local preview and manifest validation pass.
