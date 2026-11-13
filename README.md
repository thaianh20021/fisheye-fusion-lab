# Fisheye Fusion Lab

Local-first tool for pinhole, single-fisheye, two-camera fisheye, and four-camera
fisheye preview fusion. CPU is the baseline. A separate Instant NuRec inference
runner and Colab/Kaggle notebooks connect the upstream Gaussian reconstruction
backend; a prepared environment, NCore input, and checkpoint are required.
Instant NuRec has not been validated locally. A separate static gsplat experiment
has trained and rendered a small real Stump scene on native Windows CUDA.
NeRF methods are outside the current scope.

See `docs/small-scene.md` for reproduction. The completed 1,200-step experiment
uses 12 images, 4,334 Gaussians, and about 36 MiB of downloaded data.
Mean held-out PSNR improved from 16.77 to 19.50 dB. Results are intentionally
low-detail: no densification, dynamic actors, or fisheye reconstruction.
The Web UI opens images, metrics, novel viewpoints, and the exported PLY.
The experiment selector also opens a verified GEER run: 19.62 dB on the same
held-out views. The Web UI also includes an interactive GEER/gsplat training
panel ("Huan luyen 3DGEER / gsplat") supporting both NPZ scenes and 1/2/4-camera
fisheye/pinhole rig manifests (`examples/synthetic_training_rig/manifest.json`).
See `docs/geer-experiment.md` for source fixes, reproduction, UI training APIs,
and the distinction between synthetic fisheye rig checks and real pinhole training.

See `docs/reconstruction.md` for the 3D workflow, setup boundaries, and limitations.

See `docs/backend_matrix.md` for the Gaussian/feed-forward source registry.

## Large Dataset Policy

The full KITTI-360 archive is intentionally not committed to GitHub. It is
about 100+ GiB and is stored on Google Drive for Colab reuse. The notebook
downloads missing files from the public dataset mirror and skips files already
present:

- Dataset mirror: `https://huggingface.co/datasets/danny042/KITTI-360-fisheye`
- Official poses: `https://www.cvlibs.net/datasets/kitti-360/`
- Colab notebook: `tools/kitti360_3dgut_fast3dgs_colab.ipynb`

Run the notebook with a mounted Drive. It checks the archive before downloading
anything and writes only temporary extracted frames to the Colab runtime.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn fisheye_fusion.server:app --reload
```

Open `http://127.0.0.1:8000`.

CLI:

```powershell
python -m fisheye_fusion.cli validate examples\synthetic_rig\manifest.json
python -m fisheye_fusion.cli render examples\synthetic_rig\manifest.json --output preview.png

# Sample only a few frames; dry-run first to avoid filling the disk.
python -m fisheye_fusion.cli sample examples\synthetic_rig\manifest.json --output outputs\sample --max-frames 4 --max-mib 50 --dry-run
python -m fisheye_fusion.cli sample examples\synthetic_rig\manifest.json --output outputs\sample --max-frames 4 --max-mib 50
```

The sampler copies only selected camera files, never downloads a full dataset,
and refuses existing destinations. For real datasets, prepare a manifest first;
KITTI-360 local import is available; Newer College import is not implemented yet.

Dataset adapters are local-only by design. They convert an already-downloaded
small sample into the canonical manifest; they never fetch gated or full
datasets and never rewrite fisheye calibration silently.

## KITTI-360 Local Import

Only existing extracted files are read. No download or archive extraction.
The dataset must contain `calibration/`, `data_poses/<sequence>/poses.txt`,
and selected camera images and `timestamps.txt` under `data_2d_raw/<sequence>/`.
Camera 00/01 use `data_rect/`; camera 02/03 use `data_rgb/`.
Fisheye uses Mei (`xi`), not the OpenCV equidistant model.

```powershell
# Default: preflight only, 50 MiB. IDs must have exact poses in poses.txt.
python -m fisheye_fusion.cli import-kitti360 C:\data\kitti360 --sequence 2013_05_28_drive_0000_sync --frames 0 --output outputs\kitti360-sample
# Add --write to copy the sample. Add --cameras image_00 image_01 image_02 image_03 for all four cameras.
```

The Local Web UI includes the same import under "Nhap KITTI-360 local".
Web imports stay within `FUSION_WORKSPACE`; the CLI can read an external disk.
Limits: 32 frames and 200 MiB maximum, including the generated manifest.
Output must be new and outside the source dataset. Missing exact poses fail
explicitly; no interpolation. Per-camera timestamps are preserved, but the preview
does not compensate for timing offsets or camera translation. LiDAR import and
KITTI-360 neural training are not implemented. Tests use synthetic KITTI-shaped fixtures;
real dataset validation remains outstanding.

Calibration reference:
https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py
