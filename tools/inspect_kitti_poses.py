from pathlib import Path
from zipfile import ZipFile

p = Path("/content/drive/MyDrive/datasets/kitti360_full/data_poses.zip")
with ZipFile(p) as z:
    for n in ("2013_05_28_drive_0003_sync/poses.txt", "2013_05_28_drive_0003_sync/cam0_to_world.txt"):
        print("FILE", n)
        with z.open(n) as f:
            for _ in range(2):
                print(f.readline().decode().strip())
