from pathlib import Path
from urllib.request import Request, urlopen

url = "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip"
out = Path("/content/drive/MyDrive/datasets/kitti360_full/data_poses.zip")
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists() or out.stat().st_size < 1_000_000:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as src, out.open("wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
print(out, out.stat().st_size)
