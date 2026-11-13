# Fisheye Fusion Lab

Tools for reconstructing driving scenes from pinhole and fisheye cameras plus LiDAR with 3D Gaussian
Splatting, and a local Web UI for multi-camera fisheye preview fusion. The main pipeline trains a
3DGUT model of a 500 m KITTI-360 road segment (2 pinhole + 2 fisheye cameras, real LiDAR) in about
4 minutes on a Colab T4. Held-out PSNR is 18.2 dB on pinhole and 16.0 dB on fisheye. NeRF methods are
out of scope.

## What is here

| Component | What it does | Status | Docs |
|---|---|---|---|
| KITTI-360 3DGUT trainer (`tools/colab_cli_kitti360.py`) | LiDAR-seeded Gaussians, 3DGUT rendering, SelectiveAdam, MCMC densification, sky dome, hood mask, SSIM | Verified on Colab T4 (100 frames, 1M Gaussians) | [docs/kitti360-3dgut.md](docs/kitti360-3dgut.md) |
| `fflab` CLI (`fflab/`) | Node.js CLI: setup, train, render, projection check, locally or on a Colab VM; backends and accelerators as JSON manifests | Colab path verified on T4 and CPU; local GPU and Windows untested | [fflab/README.md](fflab/README.md) |
| Projection check (`tools/kitti360_projection_check.py`) | LiDAR-to-camera overlays against official calibration and poses | Verified on real data | [docs/kitti360-3dgut.md](docs/kitti360-3dgut.md#geometry-check) |
| Fly-through renderer (`tools/render_flythrough.py`) | H.264 video along the front-camera path | Verified on T4 | [docs/kitti360-3dgut.md](docs/kitti360-3dgut.md) |
| Colab notebook (`tools/kitti360_3dgut_fast3dgs_colab.ipynb`) | Same trainer, cell by cell, with Drive | Updated to the current trainer, not re-run end to end | — |
| Local Web UI and Python CLI (`fisheye_fusion/`) | Preview fusion, sampling, KITTI-360 import, small gsplat and GEER experiments | CPU baseline; small static scene 19.50 dB, GEER 19.62 dB held-out | below, [docs/small-scene.md](docs/small-scene.md), [docs/geer-experiment.md](docs/geer-experiment.md) |
| Instant NuRec runner | Inference with pretrained weights in a separate environment | Needs NCore input and a checkpoint | [docs/reconstruction.md](docs/reconstruction.md) |

Method registry: [docs/backend_matrix.md](docs/backend_matrix.md). Project log and decisions:
[context.md](context.md).

## Quick start: KITTI-360 on a Colab T4

Needs Node.js 18.3+ and, for Colab, `google-colab-cli` (Linux, macOS or WSL).

```bash
uv tool install google-colab-cli --with keyrings.alt
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
node fflab/bin/fflab.js colab train --gpu T4 --accel fused-ssim --run-name road1
node fflab/bin/fflab.js colab render --run-name road1
node fflab/bin/fflab.js colab stop
```

Results land in `./fflab-runs/road1/` (`gaussians.ply`, comparison images, `metrics.json`, `fly.mp4`).
On a machine with an NVIDIA GPU use `fflab setup` and `fflab train` instead. Every command has
`--dry-run`; `fflab help train` lists the options.

| Configuration (100 frames, 10k steps, 1M cap, T4) | PSNR pinhole | PSNR fisheye | Train time |
|---|---|---|---|
| Base | 13.4-13.6 dB | 11.0-11.5 dB | 199 s |
| + sky dome, mask, conv2d SSIM | 17.6 dB | 16.0-16.2 dB | 487 s |
| + sky dome, mask, fused-ssim (default) | 18.2 dB | 16.0 dB | 215 s |

Single runs; differences under about 1 dB are not established. Renders are still blurry; see the
limitations in [docs/kitti360-3dgut.md](docs/kitti360-3dgut.md#limitations).

## Repository layout

| Path | Content |
|---|---|
| `fflab/` | Node.js CLI, backend and accelerator manifests, tests |
| `tools/` | Python trainer, renderer, projection check, Colab notebook, earlier experiment scripts |
| `fisheye_fusion/` | Local Web UI (FastAPI) and Python CLI |
| `docs/` | Method and experiment documentation |
| `tests/` | Python tests (`python -m pytest -q`) |
| `examples/` | Synthetic rigs and manifests |
| `context.md`, `summary.md` | Project log, results and next steps |

## Large Dataset Policy

KITTI-360 is not committed to GitHub. The 3DGUT pipeline needs only sequence 0003: calibration,
metadata pickles, `data_poses.zip`, four camera tars and one LiDAR tar, about 1.7 GB in total.
Missing small files are downloaded automatically and missing tars are streamed and read only up to
the frames needed, so a local or Drive copy is optional:

- Dataset mirror: `https://huggingface.co/datasets/danny042/KITTI-360-fisheye`
- Official poses: `https://www.cvlibs.net/datasets/kitti-360/`
- Drive copy used on Colab: `MyDrive/datasets/kitti360_full` (pass `--data` and `--drive`)

## Local Web UI and Python CLI

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
(cd fflab && npm test)
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
does not compensate for timing offsets or camera translation. This importer does
not read LiDAR or train; KITTI-360 Gaussian training lives in `tools/` and runs through `fflab`
(see `docs/kitti360-3dgut.md`). Importer tests use synthetic KITTI-shaped fixtures.

Calibration reference:
https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/project.py
