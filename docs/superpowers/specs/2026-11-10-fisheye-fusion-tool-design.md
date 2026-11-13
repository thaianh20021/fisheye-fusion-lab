# Fisheye Fusion Tool - Design

Date: 2026-11-10
Status: draft; implementation pending review

## 1. Scope

Build a local web tool for calibrated pinhole and fisheye data. Support one camera,
two-camera fisheye fusion, four-camera fisheye fusion, and mixed rigs. Native
Windows and Linux are primary targets. Colab and Kaggle are optional GPU runtimes.

NeRF methods are out of scope. Research adapters target Gaussian methods only:
Street Gaussians, OmniRe, 3DGUT, SplatAD, IDSplat, STORM, DGGT, Instant NuRec,
UniGaussian, GOFish, DirectFisheye-GS, Fisheye-GS.

Do not merge all upstream repositories into one Python environment. Each backend
has its own environment, version constraints, license record, and launch command.

## 2. Architecture

```text
Browser UI
  -> FastAPI local server
  -> manifest validation
  -> dataset adapter / frame selection
  -> camera geometry and CPU fusion
  -> backend adapter
       -> local CPU preview
       -> local CUDA reconstruction
       -> Colab/Kaggle package export
  -> artifact inspection / evaluation / export
```

Core stack: Python, NumPy, OpenCV, FastAPI. UI is a small responsive browser app.
CUDA is optional in the core and mandatory only for backends that require it.
No custom renderer or neural model is implemented in the MVP.

## 3. UI

Views:

1. Project: dataset root, manifest, adapter, selected sequence.
2. Rig: camera list, model, calibration, extrinsics, selected fusion group.
3. Preview: image grid, LiDAR overlay, panorama, valid coverage mask.
4. Run: backend, runtime, resolution, frame range, resource budget.
5. Outputs: preview image/video, Gaussian bundle metadata, logs, report.

Runtime choices: local CPU, local CUDA, Colab T4, Kaggle T4.
Backend choices are disabled until their adapter is implemented and validated.
Unsupported camera models fail visibly rather than being treated as pinhole.

## 4. Canonical Manifest

JSON is the MVP interchange format. Camera count is a list, not separate codepaths.

```json
{
  "schema_version": 1,
  "project": "example",
  "coordinates": {
    "camera": "opencv",
    "rig": "right-handed-x-right-y-down-z-forward",
    "units": "meters",
    "extrinsics": "T_rig_from_camera"
  },
  "cameras": [
    {
      "id": "front",
      "model": "opencv_fisheye",
      "resolution": [1280, 720],
      "K": [[640, 0, 640], [0, 640, 360], [0, 0, 1]],
      "D": [0, 0, 0, 0],
      "T_rig_from_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ],
  "frames": [
    {
      "timestamp_ns": 0,
      "images": {"front": "images/front/000000.png"},
      "T_world_from_rig": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
  ],
  "fusion": {
    "camera_ids": ["front"],
    "output": "equirectangular",
    "resolution": [1024, 512]
  }
}
```

Optional LiDAR records include path, timestamp, format, and T_rig_from_lidar.
Validate finite numbers, focal lengths, matrix shapes, rigid transforms, unique
IDs, referenced cameras, monotonically increasing timestamps, input sizes, and
paths contained within the project root. No arbitrary command fields in manifests.

## 5. Geometry and Fusion

Initial camera models: OpenCV pinhole and OpenCV fisheye/equidistant polynomial.
Dataset-specific models require explicit conversion or a tested model adapter.
KITTI-360 MEI and WoodScape polynomial are not silently mapped to OpenCV fisheye.

Two operations must remain distinct:

- CPU image fusion: rotational panorama/ground-plane preview. This is not 3D
  reconstruction and cannot remove parallax from translated cameras.
- Gaussian reconstruction: joint optimization in one world frame through a
  compatible backend. Never promise a correct scene by concatenating four PLYs.

LiDAR projection uses measured depth and the calibrated camera transform.
Panorama output includes valid coverage. Two/four cameras do not guarantee 360
degrees without sufficient actual FOV. BEV requires an explicit ground plane and
is labelled as a plane approximation.

## 6. Datasets

- Newer College 2021 multi-camera: four grayscale fisheyes with LiDAR; handheld,
  not a front/rear/left/right automotive layout.
- KITTI-360: two fisheyes plus two perspective cameras and LiDAR.
- WoodScape: four RGB fisheyes; public raw LiDAR availability not verified.
- A2D2: later adapter; audit camera model and downloadable temporal sequences.

Start with synthetic fixtures for validation, then a user-downloaded small sample.
Never download full datasets or accept licenses automatically.

## 7. Local and Remote Runtimes

Local CPU supports validation, geometry, overlays, fusion previews, and export.
Local CUDA performs a backend-specific preflight: GPU, driver, toolkit, Python,
extension compiler, disk, RAM and VRAM. Native Windows support is verified per
backend, not inferred from support in the main UI.

Remote export produces:

```text
job/
  manifest.json
  images/
  calibration/
  lidar/                 optional
  run_colab.ipynb
  run_kaggle.ipynb
  job.json
  README.md
```

The MVP exports notebooks and bundles; users authenticate and execute remotely.
No Google/Kaggle credentials stored by the app. No fabricated live remote status.
Checkpoint/output import is explicit. Export size and included files shown first.
Notebook pins repo revision and environment; saves logs and result metadata.
Only supported backends get runnable notebooks; others remain research entries.
No guarantee that every model fits T4 or RTX 3060 6 GB.

## 8. Backend Rollout

1. CPU geometry and fixture-based fusion.
2. Instant NuRec artifact inspection and export workflow already exercised by user.
3. 3DGUT native fisheye training adapter after Windows build verification.
4. SplatAD dynamic camera/LiDAR adapter in isolated environment.
5. Remaining methods after license, weights, dataset and hardware audits.

Source checkout is separate from installation. Record URL, commit, license,
download status and verified capabilities. UniGaussian remains paper-only unless
official runnable source is found.

## 9. Validation

- Unit checks for pinhole/fisheye projection, inverse rays, extrinsics, 90-degree
  rotations, malformed inputs, path traversal and camera references.
- Synthetic 1/2/4-camera fixture tests with known angular color targets.
- LiDAR overlay with occlusion and positive-depth tests.
- UI checks on desktop/mobile, keyboard operation, errors and empty states.
- Windows native end-to-end smoke test; Linux verified separately.
- Remote notebook tested separately on actual T4 before claiming cloud support.
- Performance logs: wall time, RAM, VRAM where available, image size, frame count.

## 10. Delivery Gates

M0: validated manifest, camera math, synthetic fixtures and tests.
M1: local UI, 1/2/4 preview, LiDAR overlay, report export.
M2: one real dataset subset and confirmed coordinate/calibration correctness.
M3: one working Gaussian backend plus reproducible cloud bundle.
M4: four-camera reconstruction and held-out-view evaluation.
M5: additional backends, simulator feasibility, broader OS/runtime matrix.

Deferred: live capture, automatic calibration, depth-aware stitching, optical-flow
seams, automatic cloud job control, training all 12 methods, and simulator physics.
