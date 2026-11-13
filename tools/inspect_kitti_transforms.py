import pickle
from pathlib import Path
import numpy as np

p = Path("/content/drive/MyDrive/datasets/kitti360_full/ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl")
d = pickle.load(p.open("rb"))
x = next(x for x in d["data_list"] if "2013_05_28_drive_0003_sync" in x["lidar_path"])
for k in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"):
    info = x["images"][k]
    print(k)
    print(np.array(info["lidar2cam"], dtype=np.float32))
