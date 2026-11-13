from pathlib import Path

p = Path("/content/drive/MyDrive/datasets/kitti360_full")
for x in sorted(p.rglob("*")):
    if x.is_file() and any(k in x.name.lower() for k in ("pose", "calib", "cam", "oxts")):
        print(x.relative_to(p))
