import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/danny042/KITTI-360-fisheye/resolve/main"
ROOT = Path("/content/drive/MyDrive/datasets/kitti360_full")
ROOT.mkdir(parents=True, exist_ok=True)

FILES = [
    "calibration/calib_cam_to_pose.txt",
    "calibration/calib_cam_to_velo.txt",
    "calibration/calib_sick_to_velo.txt",
    "calibration/image_02.yaml",
    "calibration/image_03.yaml",
    "calibration/perspective.txt",
    "ImageSets-dapetr-v49-port/kitti360_dapetr_infos_train.pkl",
    "ImageSets-dapetr-v49-port/kitti360_dapetr_infos_val.pkl",
    "ImageSets-rosa-6class/kitti360_rosa_infos_train.pkl",
    "ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl",
]
for seq in ("0000", "0002", "0003", "0004", "0005", "0006", "0007", "0009", "0010"):
    name = f"2013_05_28_drive_{seq}_sync"
    FILES.extend(
        [f"data_2d_raw/{name}_image_{cam}.tar" for cam in ("00", "01", "02", "03")]
    )
    FILES.append(f"data_3d_raw/{name}.tar")


def download(rel):
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    part = dst.with_suffix(dst.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "kitti360-full-loader"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    try:
        with urllib.request.urlopen(req, timeout=120) as src:
            mode = "ab" if start else "wb"
            total = int(src.headers.get("Content-Length", "0")) + start
            done = start
            with part.open(mode) as out:
                while True:
                    block = src.read(8 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total:
                        print(f"[{done * 100 // total:3d}%] {rel}", flush=True)
        part.replace(dst)
        print(f"[DONE] {rel}", flush=True)
    except Exception as exc:
        print(f"[FAIL] {rel}: {exc}", flush=True)
        return False
    return True


print(f"Target: {ROOT}", flush=True)
print(f"Files: {len(FILES)}", flush=True)
failed = [rel for rel in FILES if not download(rel)]
if failed:
    print("FAILED FILES:", *failed, sep="\n", flush=True)
    sys.exit(1)
print("FULL KITTI360 ARCHIVE READY", flush=True)
