# Backend Source Matrix

Research registry for the local tool. Metadata only; upstream repositories keep
their own environments, checkpoints, and license obligations.

| Method | Official source | Camera | LiDAR | Scene | Local status |
|---|---|---|---|---|---|
| Street Gaussians | https://github.com/zju3dv/street_gaussians | Driving cameras | Depth/LiDAR init | Dynamic vehicles | Research |
| OmniRe / DriveStudio | https://github.com/ziyc/drivestudio | Multi-camera driving | Init/depth | Dynamic actors | Research |
| 3DGUT | https://github.com/nv-tlabs/3DGRUT | Pinhole/fisheye/rolling shutter | No dedicated path | Mainly static | Runnable via `fflab` (`3dgut-gsplat`): gsplat 1.5.3's 3DGUT, KITTI-360, verified on Colab T4 |
| SplatAD | https://github.com/georghess/neurad-studio | Pinhole/rolling shutter | Camera + LiDAR | Dynamic driving | Colab research |
| IDSplat | https://github.com/zenseact/idsplat | Calibrated cameras | Camera/LiDAR synthesis | Instance dynamic | Research |
| GaussianSTORM | https://github.com/NVlabs/GaussianSTORM | Sparse RGB | Not documented | Dynamic | Research |
| DGGT | https://github.com/xiaomi-research/dggt | Unposed RGB | Not documented | Dynamic 4D | Research |
| Instant NuRec | https://github.com/NVIDIA/instant-nurec | NCore non-pinhole | Separate NuRec path | Static + dynamic | Inference verified |
| UniGaussian | https://openreview.net/pdf?id=ZaRdmvvkvF | Pinhole/fisheye | Optional | Dynamic driving | Paper-only |
| GOFish | https://github.com/tuallen/GOFish | Wide fisheye | LiDAR-free | Static | Research |
| DirectFisheye-GS | https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS | Fisheye/pinhole | Not documented | Static | Research |
| Fisheye-GS | https://github.com/zmliao/Fisheye-GS | Fisheye/pinhole | Not documented | Static | Research |

`3dgut-gsplat` uses the 3DGUT implementation inside gsplat (`with_ut`, `with_eval3d`), not the 3DGRUT
repository; LiDAR enters only as initialisation. Results: [kitti360-3dgut.md](kitti360-3dgut.md).
A runnable method gets a manifest in `fflab/backends/`.

NeRF methods are excluded from the development scope. Runtime and license claims
must be rechecked against the selected upstream revision before an adapter is
marked runnable.
