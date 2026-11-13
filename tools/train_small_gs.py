"""Small static 3DGS experiment using the upstream gsplat CUDA rasterizer."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


def save_ply(path, means, quats, log_scales, logits, rgb):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(len(means), dtype=[(name, "<f4") for name in names])
    values = np.column_stack([means, np.zeros_like(means), (rgb - .5) / .28209479177387814,
                              logits, log_scales, quats])
    for index, name in enumerate(names):
        data[name] = values[:, index]
    with Path(path).open("xb") as stream:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {len(data)}\n"
        header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
        stream.write(header.encode("ascii"))
        stream.write(data.tobytes())


def main():
    from gsplat import rasterization
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--points", type=int, default=15000)
    parser.add_argument("--geer", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 100 <= args.points <= 30000:
        parser.error("steps must be 1..10000 and points 100..30000")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training backend")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    with np.load(args.dataset, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    images = arrays["images"].astype("float32")
    if images.ndim != 4 or images.shape[-1] != 3 or not np.isfinite(images).all():
        raise ValueError("Expected finite RGB images [M,H,W,3]")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be normalized to 0..1")
    points = arrays["points"].astype("float32")
    rgb = arrays["colors"].astype("float32")
    if points.shape != rgb.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sparse points/colors must have shape [N,3]")
    keep = np.isfinite(points).all(1) & np.isfinite(rgb).all(1)
    points, rgb = points[keep], rgb[keep]
    if len(points) < 100:
        raise ValueError("Need at least 100 valid SfM points")
    if len(points) > args.points:
        selection = rng.choice(len(points), args.points, replace=False)
        points, rgb = points[selection], rgb[selection]
    train = arrays["train_indices"].astype(int).tolist()
    test = arrays["test_indices"].astype(int).tolist()
    if not train or not test or set(train) & set(test):
        raise ValueError("Need disjoint nonempty train and held-out views")
    if min(train + test) < 0 or max(train + test) >= len(images):
        raise ValueError("Invalid camera indices")
    viewmats = arrays["viewmats"].astype("float32")
    intrinsics = arrays["Ks"].astype("float32")
    if viewmats.shape != (len(images), 4, 4) or intrinsics.shape != (len(images), 3, 3):
        raise ValueError("Camera matrix dimensions do not match images")
    if not np.isfinite(viewmats).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Camera matrices must be finite")
    args.output.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    to_gpu = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    targets, views, Ks = map(to_gpu, (images, viewmats, intrinsics))
    models = arrays.get("camera_models", np.full(len(images), "pinhole")).tolist()
    distortion = to_gpu(arrays.get("distortions", np.zeros((len(images), 4))))
    if len(models) != len(images) or any(model not in ("pinhole", "fisheye") for model in models):
        raise ValueError("Unsupported camera model")
    if distortion.shape != (len(images), 4) or not torch.isfinite(distortion).all():
        raise ValueError("Expected finite distortion [M,4]")
    if not args.geer and (any(model != "pinhole" for model in models) or distortion.any()):
        raise ValueError("Calibrated fisheye requires GEER")
    height, width = images.shape[1:3]
    distances = cKDTree(points).query(points, k=4)[0][:, 1:].mean(1).clip(1e-4)
    extent = float(np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
    extent = max(extent, 1e-3)
    means = torch.nn.Parameter(to_gpu(points))
    quats = torch.nn.Parameter(to_gpu(np.tile([1, 0, 0, 0], (len(points), 1))))
    scales = torch.nn.Parameter(to_gpu(np.log(distances)[:, None].repeat(3, axis=1)))
    opacities = torch.nn.Parameter(torch.full((len(points),), -1.5, device=device))
    colors = torch.nn.Parameter(torch.logit(to_gpu(rgb).clamp(.001, .999)))
    # ponytail: fixed sparse initialization; add densification for production quality.
    optimizer = torch.optim.Adam([
        {"params": [means], "lr": extent * .00016},
        {"params": [quats], "lr": .001},
        {"params": [scales], "lr": .005},
        {"params": [opacities], "lr": .05},
        {"params": [colors], "lr": .025},
    ], eps=1e-15)

    def render(index, view=None):
        return rasterization(
            means, quats, scales.exp(), opacities.sigmoid(), colors.sigmoid(),
            views[index:index+1] if view is None else view[None], Ks[index:index+1],
            width, height, packed=False, near_plane=.01, far_plane=10000.,
            camera_model=models[index],
            radial_coeffs=distortion[index:index+1] if models[index] == "fisheye" else None,
            **({"with_geer": True, "with_eval3d": True} if args.geer else {}),
        )[0][0]

    def write_image(name, value):
        pixels = (value.detach().clamp(0, 1).cpu().numpy() * 255).round().astype("uint8")
        if not cv2.imwrite(str(args.output / name), pixels[:, :, ::-1]):
            raise OSError("Image write failed")

    def psnr(index):
        error = (render(index) - targets[index]).square().mean().clamp_min(1e-10)
        return float(-10 * torch.log10(error))

    with torch.no_grad():
        before = {str(i): psnr(i) for i in test}
        write_image("before.png", render(test[0]))
        write_image("reference.png", targets[test[0]])
    started = time.monotonic()
    trace = []
    for step in range(args.steps):
        index = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        prediction = render(index)
        loss = (prediction - targets[index]).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; stopping without declaring success")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scales.clamp_(min=np.log(extent * 1e-5), max=np.log(extent * .5))
            opacities.clamp_(-12, 12)
        if step % 100 == 0 or step == args.steps - 1:
            item = {"step": step + 1, "l1": float(loss.detach()), "seconds": round(time.monotonic() - started, 2)}
            trace.append(item)
            print(json.dumps(item), flush=True)
    with torch.no_grad():
        after = {str(i): psnr(i) for i in test}
        write_image("after.png", render(test[0]))
        for index in test:
            write_image(f"heldout-{index}.png", render(index))
        # A small translation sweep is genuinely new viewpoints, not a panorama.
        novel = []
        camera = torch.linalg.inv(views[test[0]])
        for index, offset in enumerate(np.linspace(-.05, .05, 9)):
            shifted = camera.clone()
            shifted[:3, 3] += camera[:3, 0] * float(offset * extent)
            name = f"novel-{index:02d}.png"
            write_image(name, render(test[0], torch.linalg.inv(shifted)))
            novel.append(name)
        normalized = torch.nn.functional.normalize(quats, dim=-1)
        values = [tensor.detach().cpu().numpy() for tensor in (means, normalized, scales, opacities, colors.sigmoid())]
        save_ply(args.output / "scene.ply", *values)
    metrics = {
        "backend": "3DGEER / static 3DGS" if args.geer else "gsplat 1.5.3 / static 3DGS",
        "dataset": str(args.dataset.resolve()),
        "gpu": torch.cuda.get_device_name(0), "steps": args.steps, "gaussians": len(points),
        "resolution": [width, height], "train_indices": train, "test_indices": test,
        "camera_models": models,
        "heldout_psnr_before": before, "heldout_psnr_after": after, "trace": trace,
        "seconds": round(time.monotonic() - started, 2),
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "before": "before.png", "after": "after.png", "reference": "reference.png",
        "novel_views": novel, "ply": "scene.ply",
        "limitations": "Static calibrated scene, fixed world-space seed Gaussians, no densification or dynamic actors; not SplatAD/Instant NuRec. Held-out images excluded from gradient training; seed/calibration provenance determines evaluation independence.",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
