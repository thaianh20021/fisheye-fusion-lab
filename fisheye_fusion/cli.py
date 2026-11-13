import argparse
from pathlib import Path

from .fusion import fuse_images
from .manifest import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="fisheye-fusion")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("manifest")
    render = sub.add_parser("render")
    render.add_argument("manifest")
    render.add_argument("--frame", type=int, default=0)
    render.add_argument("--output", default="preview.png")
    sample = sub.add_parser("sample")
    sample.add_argument("manifest")
    sample.add_argument("--output", required=True)
    sample.add_argument("--start", type=int, default=0)
    sample.add_argument("--stride", type=int, default=1)
    sample.add_argument("--max-frames", type=int, default=4)
    sample.add_argument("--max-mib", type=float, default=50)
    sample.add_argument("--include-lidar", action="store_true")
    sample.add_argument("--dry-run", action="store_true")
    kitti = sub.add_parser("import-kitti360")
    kitti.add_argument("dataset_root")
    kitti.add_argument("--sequence", required=True)
    kitti.add_argument("--frames", nargs="+", type=int, required=True)
    kitti.add_argument("--cameras", nargs="+", default=["image_02", "image_03"])
    kitti.add_argument("--output", required=True)
    kitti.add_argument("--max-mib", type=int, default=50)
    kitti.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.command == "import-kitti360":
        from .kitti360 import build_kitti360_manifest
        from .sample import sample_project
        manifest = build_kitti360_manifest(
            args.dataset_root, args.sequence, args.frames,
            camera_ids=args.cameras, max_bytes=args.max_mib * 1024**2,
        )
        result = sample_project(
            manifest, args.dataset_root, args.output, max_frames=len(args.frames),
            max_bytes=args.max_mib * 1024**2, write=args.write,
        )
        print(f"{'wrote' if result['written'] else 'planned'} {result['frames']} frames, "
              f"{result['bytes'] / 1048576:.1f} MiB -> {result['output']}")
        return 0
    manifest_path = Path(getattr(args, "manifest"))
    if args.command == "validate":
        manifest = load_manifest(manifest_path)
        print(f"valid: {manifest.project} ({len(manifest.cameras)} cameras)")
    elif args.command == "render":
        manifest = load_manifest(manifest_path)
        image, coverage = fuse_images(manifest, manifest_path.parent, args.frame)
        import cv2
        cv2.imwrite(args.output, image[:, :, ::-1])
        print(f"wrote {args.output}; coverage={coverage.mean():.3f}")
    else:
        from .sample import sample_dataset
        result = sample_dataset(
            manifest_path,
            args.output,
            start=args.start,
            stride=args.stride,
            max_frames=args.max_frames,
            max_bytes=round(args.max_mib * 1024 * 1024),
            include_lidar=args.include_lidar,
            write=not args.dry_run,
        )
        print(f"{'planned' if args.dry_run else 'wrote'} {result['frames']} frames, "
              f"{result['bytes'] / 1048576:.1f} MiB -> {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
