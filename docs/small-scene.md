# Small real-scene CUDA experiment

This is static pinhole 3D Gaussian Splatting, not NeRF, SplatAD, Instant
NuRec, or dynamic fisheye fusion. The scene is Stump from Mip-NeRF360.
Only 12 photographs and sparse calibration are downloaded: about 36 MiB.
Training uses 10 images; 2 are held out from gradient optimization.
Upstream COLMAP used all views, so this is not independent calibration.

## Windows environment used

- Python 3.12, existing PyTorch 2.6.0+cu124
- CUDA Toolkit 12.4, VS2022 C++ tools, Windows SDK
- RTX 3060 Laptop 6 GB

```powershell
uv venv --system-site-packages .venv-gs --python 3.12
uv pip install --python .venv-gs/Scripts/python.exe --no-deps gsplat==1.5.3 ninja==1.13.2 jaxtyping==0.3.11 rich==15.0.0
```

This environment reuses existing Torch, NumPy, SciPy, and OpenCV.
It is not a standalone dependency lock for other machines.

In `.venv-gs/Lib/site-packages/gsplat/cuda/_backend.py`, replace:

```python
extra_cflags = [opt_level, "-Wno-attributes"]
```

with:

```python
extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]
```

MSVC rejects `-Wno-attributes`. Reinstalling gsplat removes this local patch.
The launcher targets compute capability 8.6 and permits the locally installed
MSVC 14.44 with CUDA 12.4; adapt those settings for other hardware/toolchains.

```powershell
.\tools\run_gs.ps1 tools/check_gs_cuda.py
.\tools\run_gs.ps1 tools/train_small_gs.py examples/real_small_scene/training.npz --output outputs/real-gs-run --steps 1200
```

The training command refuses existing output directories. Results include a
standard binary Gaussian PLY, held-out reference/before/after images, nine
translated novel views, and measured metrics. The Local Web UI reads this
fixed output folder using the "Mo ket qua da chay" button.

Fixed sparse Gaussians and degree-zero color keep this experiment small.
No densification, moving actors, or production-quality claims.
Dataset provenance and download accounting are stored alongside the NPZ.
Check upstream image rights before redistributing the sample.

## Verified run: 2026-11-11

- CUDA forward and backward check passed.
- 1,200 steps, 4,334 Gaussians, 400 x 266 pixels.
- Held-out PSNR: 17.286 to 20.748 dB; 16.252 to 18.262 dB.
- Measured optimization loop: 3.34 s; training plus final exports: 3.58 s.
  Excludes environment setup, CUDA compilation (338 s), and initial rendering.
- Peak Torch allocated tensor memory: 27.68 MiB, not total GPU process memory.
- Finite PLY checked; nine translated renders differ; image loading, PLY
  download, and desktop/tablet/mobile overflow checks passed.

```powershell
python tests/check_real_results.py
```
