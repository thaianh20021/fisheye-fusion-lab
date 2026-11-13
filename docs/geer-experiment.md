# GEER Windows experiment

Verified locally on 2026-11-11, RTX 3060 Laptop 6 GB, native Windows.
Separate `.venv-geer` and `outputs/cuda-cache-geer` preserve the gsplat baseline.
No new dataset or Torch download. Both experiments use the same 12 Stump images,
10 training views, 2 held-out views, 4,334 fixed Gaussians, seed 42, 1,200 steps.

## Source

Repository: https://github.com/boschresearch/3dgeer

Branch: `gsplat-geer`

Commit: `f15cee225206944e38acdd0ab4a7ab4e4aae78a1`

Local checkout: `work/3dgeer-src`. Upstream LICENSE.md is retained.
Review upstream licensing before redistributing this integration.

## Setup used

```powershell
git clone --depth 1 --filter=blob:none --sparse --branch gsplat-geer https://github.com/boschresearch/3dgeer.git work/3dgeer-src
git -C work/3dgeer-src sparse-checkout set gsplat
uv venv --system-site-packages .venv-geer --python 3.12
uv pip install --python .venv-geer/Scripts/python.exe --no-deps ninja==1.13.2 jaxtyping==0.3.11 rich==15.0.0
```

Two local source fixes were required:

1. In `gsplat/cuda/_backend.py`, use
   `extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]`.
2. In `gsplat/cuda/csrc/IntersectGEERTile.cu`, replace `std::isfinite`
   with `::isfinite` in device code. MSVC otherwise resolves CPU-only overloads.

This commit did not contain the expected GLM submodule directory. Reuse
the headers supplied by the installed baseline gsplat 1.5.3 wheel:

```powershell
Copy-Item -LiteralPath .venv-gs/Lib/site-packages/gsplat/cuda/csrc/third_party -Destination work/3dgeer-src/gsplat/cuda/csrc/third_party -Recurse
$env:BUILD_NO_CUDA = "1"
uv pip install --python .venv-geer/Scripts/python.exe --no-deps --no-build-isolation -e work/3dgeer-src
.\tools\run_gs.ps1 -UseGeer tools/check_gs_cuda.py --geer
.\tools\run_gs.ps1 -UseGeer tools/check_gs_cuda.py --geer --fisheye
.\tools\run_gs.ps1 -UseGeer tools/train_small_gs.py examples/real_small_scene/training.npz --output outputs/geer-run --steps 1200 --geer
```

Run from the workspace. The launcher loads the VS2022 developer environment;
see `small-scene.md` for shared prerequisites and hardware-specific flags.
Compilation took 357 seconds. Existing output folders are never overwritten.

## Observed results

| Metric | Baseline gsplat | GEER |
| --- | ---: | ---: |
| Mean held-out PSNR before | 16.77 dB | 16.45 dB |
| Mean held-out PSNR after | 19.50 dB | 19.62 dB |
| Optimization loop | 3.34 s | 4.50 s |
| Training and final exports | 3.58 s | 4.67 s |
| Peak Torch allocated tensors | 27.68 MiB | 30.87 MiB |

Times exclude setup, CUDA compilation, and initial rendering.
Memory is not total process VRAM. Single tiny runs are not a performance
benchmark; the 0.11 dB difference does not establish general superiority.
Different projection modes have different initial renders.

Verified forward/backward CUDA for pinhole and fisheye. Fisheye check uses
both one synthetic Gaussian (`tools/check_gs_cuda.py --geer --fisheye`) and a
4-camera synthetic fisheye rig (`examples/synthetic_training_rig/manifest.json`,
`opencv_fisheye` model, 512 shared world-space Gaussians, full 6-DoF camera extrinsics).
Real-photo scene training (`examples/real_small_scene/training.npz`) remains
pinhole, static, with no densification or moving actors.
Upstream COLMAP calibration used all views, including held-out photographs.

## Interactive UI Training & Multi-Camera Rig Flow

The Web UI ("Huan luyen 3DGEER / gsplat") and REST API run single-GPU training
jobs directly from the browser:

- `POST /api/training/inspect` — validate NPZ or rig `manifest.json`, return camera list and view counts.
- `POST /api/training` — start a `geer` or `gsplat` training job with selectable 1/2/4 cameras and steps.
- `GET /api/training/{job_id}` — poll live step/loss/PSNR progress (`status.json`).
- `POST /api/training/{job_id}/cancel` — terminate active training process tree.
- `GET /api/training/{job_id}/artifacts/{name}` — serve rendered `reference.png`, `before.png`, `after.png`, `metrics.json`, `gaussians.ply`.

Generate the synthetic 4-camera fisheye training rig fixture (if not present):

```powershell
python tools/make_training_rig.py
```

The UI's experiment selector opens both runs, including before/after images,
nine novel views, metrics and PLY downloads. PLY stores Gaussian parameters,
not a portable guarantee of GEER-identical rendering in other viewers.

```powershell
$env:EXPERIMENT_RUN = "geer"
python tests/check_real_results.py
python tests/browser_training_check.py
python -m pytest -q
```

Artifact validation, end-to-end UI training (real Stump NPZ, 2/4-camera synthetic
fisheye rig, reload recovery, cancellation), and desktop/tablet/mobile checks passed.
Full suite: 269 passed.
