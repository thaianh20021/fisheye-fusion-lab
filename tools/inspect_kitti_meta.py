import pickle
from pathlib import Path

p = Path("/content/drive/MyDrive/datasets/kitti360_full/ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl")
d = pickle.load(p.open("rb"))
x = next(x for x in d["data_list"] if "2013_05_28_drive_0003_sync" in x["lidar_path"])
print("ITEM_KEYS", sorted(x.keys()))
print("POSE_KEYS", [k for k in x if "pose" in k.lower() or "lidar" in k.lower()])
print("FRAME", x.get("frame_id"))
print("IMAGE_KEYS", list(x["images"].keys()))
print("IMAGE_ITEM_KEYS", sorted(next(iter(x["images"].values())).keys()))
