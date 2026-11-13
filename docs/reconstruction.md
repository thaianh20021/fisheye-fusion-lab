# Gaussian 3D reconstruction

## What is connected

The Local Web UI launches the official Instant NuRec inference CLI in a separate
Python environment. It tracks status/logs and exposes generated Gaussian PLY,
sky/dynamic sidecars, and optional calibrated `.render.png` files. No panorama
is converted to fake depth or labeled as a reconstructed scene.

This is **inference with pretrained weights**, not per-scene training. CPU
panorama fusion, KITTI-360 import, and Instant NuRec are separate workflows.
The canonical Fusion Lab manifest is not NCore and is rejected by this backend.
No automatic KITTI-360-to-NCore or arbitrary 2/4-fisheye rig conversion exists.

The upstream released profiles are front-camera or multiview (1/3/5 context
cameras). The renderer preview is on the source camera trajectory, not an
arbitrary free-camera novel view. A Gaussian-capable viewer is needed to
interactively inspect PLY; sky/dynamic sidecars are not displayed automatically
by ordinary PLY viewers.

## Local execution

Use only a trusted upstream checkout, Python interpreter, and checkpoint.
The backend runs code from the configured repository on your computer.

1. Prepare the upstream environment separately; do not install its dependencies
   into the Fusion Lab Python environment.
2. Put an existing NCore clip, matching checkpoint, and prepared checkout under
   `FUSION_WORKSPACE`, or run the app with that variable pointing at their common
   parent. This workflow does not download model weights or datasets.
3. Open the Gaussian 3D panel and enter repository, interpreter, checkpoint, and
   NCore JSON paths. Select the model matching the checkpoint.
4. Run preflight. Resolve missing assets/CUDA/dependencies before starting.
5. Start reconstruction. A single job runs at a time. Cancel if needed; partial
   outputs remain available on disk rather than being deleted.
6. Download artifacts after the run. A successful process must produce a nonempty
   PLY; a successful exit without outputs is not called a reconstruction.

The preview checkbox requires the upstream render extra, a CUDA toolchain, and
Ninja. Review the upstream third-party terms before installing the render extra.
Native Windows success is not established. The web tool itself runs on Windows;
the external backend may need Linux, WSL, Colab, or Kaggle.

## Colab / Kaggle

Download the Instant NuRec notebook from the Gaussian 3D panel, not the separate
CPU-preview export buttons. The notebook pins upstream revision:

`f2ecafcdf28f2dc8f2cc17447a772571cde558be`

Setup is explicitly opt-in because Python/CUDA dependencies use substantial disk
space. Supply an already accessible local NCore JSON and matching checkpoint.
No credentials are stored in the notebook. Do not paste HF or Google tokens into
the UI. Downloading this notebook does not upload any dataset.

Default execution limits are one time chunk and a merge target of 200,000
Gaussians. The target is **not a hard VRAM limit**: the model can produce more
Gaussians before merging. A chunk may still contain many frames; this does not
guarantee a 50 MiB input. T4 or a 6 GiB local GPU may run out of memory. Input size
restrictions on the CPU sampler do not automatically apply to an NCore clip.

Local inference requests offline Hugging Face/Transformers operation and a local
checkpoint override. These environment settings are not an OS network sandbox.
Use trusted, fully local NCore references; the external backend owns its parsing.

## Verification status

Unit/API tests simulate subprocesses. They verify command construction, limits,
failure/cancellation handling, containment, and notebook generation. They do not
prove reconstruction quality or GPU compatibility. No real inference has been
run by this integration yet because local checkpoint/NCore assets are absent.

## Upstream references

- https://github.com/NVIDIA/instant-nurec/tree/f2ecafcdf28f2dc8f2cc17447a772571cde558be
- https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/README.md
- https://github.com/NVIDIA/instant-nurec/blob/f2ecafcdf28f2dc8f2cc17447a772571cde558be/THIRD_PARTY_LICENSE.txt
