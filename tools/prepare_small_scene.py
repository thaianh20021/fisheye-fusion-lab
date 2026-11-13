"""Download a pinned, calibrated Mip-NeRF 360 subset; no archives or weights.

Binary layout reference (independent stdlib parser, no copied implementation):
https://colmap.github.io/format.html
https://github.com/colmap/colmap/blob/3.9.1/scripts/python/read_write_model.py
"""

import hashlib
import io
import json
from pathlib import Path
import re
import struct
import urllib.request

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "examples" / "real_small_scene"
REPO = "YuanhaoXD/Mip-NeRF360"
REVISION = "df31214ab1a3d72e0e4bfa1bfb3c03132b5cbe22"
SCENE = "stump"
LIMIT = 50 * 1024**2
# Conservatively includes the source-discovery API requests made before this script.
DISCOVERY_ALLOWANCE = 1024**2
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
          2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def write_json(path, data):
    part = path.with_suffix(path.suffix + ".part")
    try:
        part.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


class Downloads:
    """Persistent conservative budget: failed attempts remain fully charged."""

    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "download_ledger.json"
        self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                      {"limit": LIMIT, "charged_bytes": DISCOVERY_ALLOWANCE,
                       "discovery_allowance_bytes": DISCOVERY_ALLOWANCE, "files": {}})
        if not DISCOVERY_ALLOWANCE <= self.state["charged_bytes"] <= LIMIT:
            raise ValueError("Invalid download ledger")

    def fetch(self, url, relative, expected=None, ceiling=256 * 1024):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe download path")
        destination = self.root / relative
        record = self.state["files"].get(relative.as_posix())
        if destination.exists():
            data = destination.read_bytes()
            if (record and record["url"] == url and len(data) == record["bytes"]
                    and hashlib.sha256(data).hexdigest() == record["sha256"]):
                return data
            raise ValueError(f"Unverified existing file: {destination}")
        reserved = expected if expected is not None else ceiling
        if reserved <= 0 or self.state["charged_bytes"] + reserved > LIMIT:
            raise ValueError("50 MiB cumulative download budget would be exceeded")
        self.state["charged_bytes"] += reserved
        write_json(self.path, self.state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        size = 0
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
            with urllib.request.urlopen(request, timeout=90) as response:
                length = response.headers.get("Content-Length")
                if length and (int(length) > reserved or
                               (expected is not None and int(length) != expected)):
                    raise ValueError("Unexpected Content-Length")
                with part.open("wb") as output:
                    while size < reserved:
                        block = response.read(min(65536, reserved - size))
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        size += len(block)
            if expected is not None and size != expected:
                raise ValueError(f"Truncated download: {relative}")
            if expected is None and size == reserved:
                raise ValueError("Metadata response reached its size ceiling")
            self.state["charged_bytes"] -= reserved - size
            self.state["files"][relative.as_posix()] = {
                "url": url, "bytes": size, "sha256": digest.hexdigest()}
            write_json(self.path, self.state)
            part.replace(destination)
        finally:
            part.unlink(missing_ok=True)
        print(f"{relative}: {size:,} bytes", flush=True)
        return destination.read_bytes()

    def tree(self, folder):
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{SCENE}/{folder}?limit=1000"
        data = self.fetch(url, "metadata/" + REPO.replace("/", "_") + "_" +
                          folder.replace("/", "_") + ".json")
        entries = json.loads(data)
        if not isinstance(entries, list) or len(entries) >= 1000:
            raise ValueError("Unexpected or potentially incomplete API tree")
        return {item["path"]: item for item in entries if item["type"] == "file"}

    def file(self, entry):
        path, size = entry["path"], entry["size"]
        if not path.startswith(SCENE + "/") or not isinstance(size, int):
            raise ValueError("Unexpected source entry")
        relative = path[len(SCENE) + 1:]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        data = self.fetch(url, relative, expected=size)
        source_hash = entry.get("lfs", {}).get("oid", "")
        if re.fullmatch(r"[0-9a-f]{64}", source_hash):
            if hashlib.sha256(data).hexdigest() != source_hash:
                raise ValueError(f"Source SHA256 mismatch: {path}")
        elif "lfs" not in entry:
            blob = b"blob " + str(len(data)).encode() + b"\0" + data
            if hashlib.sha1(blob).hexdigest() != entry["oid"]:
                raise ValueError(f"Git blob mismatch: {path}")
        return data


def unpack(stream, fmt):
    fmt = "<" + fmt
    data = stream.read(struct.calcsize(fmt))
    if len(data) != struct.calcsize(fmt):
        raise ValueError("Truncated COLMAP record")
    return struct.unpack(fmt, data)


def skip(stream, count):
    if count < 0 or stream.tell() + count > len(stream.getbuffer()):
        raise ValueError("Truncated COLMAP array")
    stream.seek(count, 1)


def read_cameras(data):
    stream, cameras = io.BytesIO(data), {}
    for _ in range(unpack(stream, "Q")[0]):
        cid, model, width, height = unpack(stream, "iiQQ")
        if model not in MODELS:
            # ponytail: common perspective models only; add exact fisheye remapping if needed.
            raise ValueError(f"Unsupported camera model {model}")
        params = unpack(stream, "d" * MODELS[model][1])
        if cid in cameras or width <= 0 or height <= 0 or not np.isfinite(params).all():
            raise ValueError("Invalid camera")
        cameras[cid] = (model, width, height, params)
    if stream.read(1):
        raise ValueError("Trailing camera data")
    return cameras


def read_images(data):
    stream, images = io.BytesIO(data), {}
    for _ in range(unpack(stream, "Q")[0]):
        values = unpack(stream, "i7di")
        name = bytearray()
        while True:
            char = stream.read(1)
            if not char or len(name) > 4096:
                raise ValueError("Invalid COLMAP image name")
            if char == b"\0":
                break
            name.extend(char)
        name = name.decode("utf-8")
        count = unpack(stream, "Q")[0]
        offset = stream.tell()
        skip(stream, count * 24)
        observations = np.frombuffer(data, dtype=[("xy", "<f8", (2,)), ("id", "<i8")],
                                     count=count, offset=offset)
        if name in images:
            raise ValueError("Duplicate COLMAP image name")
        images[name] = {"id": values[0], "q": values[1:5], "t": values[5:8],
                        "camera_id": values[8], "observations": observations}
    if stream.read(1):
        raise ValueError("Trailing image data")
    return images


def read_points(data, training_ids):
    stream, points = io.BytesIO(data), {}
    for _ in range(unpack(stream, "Q")[0]):
        record = unpack(stream, "Q3d3Bd")
        count = unpack(stream, "Q")[0]
        offset = stream.tell()
        skip(stream, count * 8)
        track = np.frombuffer(data, dtype="<i4", count=count * 2, offset=offset).reshape(-1, 2)
        if any(int(image_id) in training_ids for image_id in track[:, 0]):
            if not np.isfinite(record[1:4]).all():
                raise ValueError("Nonfinite sparse point")
            points[record[0]] = (record[1:4], record[4:7])
    if stream.read(1) or not points:
        raise ValueError("Invalid or empty sparse points")
    return points


def viewmat(image):
    q = np.asarray(image["q"], dtype=np.float64)
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError("Invalid camera quaternion")
    q /= np.linalg.norm(q)
    w, v = q[0], q[1:]
    x, y, z = v
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    result = np.eye(4)
    result[:3, :3] = (w*w - v @ v) * np.eye(3) + 2*np.outer(v, v) + 2*w*cross
    result[:3, 3] = image["t"]
    if not np.isfinite(result).all():
        raise ValueError("Invalid camera translation")
    return result.astype(np.float32)


def intrinsics(camera, width, height):
    model, original_w, original_h, p = camera
    distortion = np.zeros(5)
    if model in (0, 2, 3):
        fx, cx, cy = p[:3]
        fy = fx
        if model in (2, 3):
            distortion[0] = p[3]
        if model == 3:
            distortion[1] = p[4]
    else:
        fx, fy, cx, cy = p[:4]
        if model == 4:
            distortion[:4] = p[4:8]
    if min(fx, fy, width, height) <= 0:
        raise ValueError("Invalid focal length or image dimensions")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K[0] *= width / original_w
    K[1] *= height / original_h
    return K, distortion


def prepare(root=ROOT):
    downloads = Downloads(root)
    calibration = downloads.tree("sparse/0")
    photos = downloads.tree("images_8")
    entries = [calibration[f"{SCENE}/sparse/0/{name}.bin"]
               for name in ("cameras", "images", "points3D")]
    if sum(e["size"] for e in entries) >= 40 * 1024**2:
        raise ValueError("Sparse model exceeds 40 MiB")
    # Preflight before payload downloads, using the largest possible 12-image subset.
    photo_entries = [e for p, e in photos.items() if p.lower().endswith((".jpg", ".png"))]
    if len(photo_entries) < 12:
        raise ValueError("Not enough source photos")
    needed = entries + sorted(photo_entries, key=lambda e: e["size"], reverse=True)[:12]
    missing = sum(e["size"] for e in needed
                  if not (root / e["path"][len(SCENE) + 1:]).exists())
    if downloads.state["charged_bytes"] + missing > LIMIT:
        raise ValueError("Dataset cannot fit within remaining 50 MiB budget")
    cameras = read_cameras(downloads.file(entries[0]))
    poses = read_images(downloads.file(entries[1]))
    available = {Path(e["path"]).name: e for e in photo_entries}
    names = sorted(set(poses) & set(available), key=lambda n: poses[n]["id"])[:12]
    if len(names) != 12:
        raise ValueError("Not enough calibrated photos")
    test_indices = np.array([3, 8], dtype=np.int64)
    train_indices = np.array([i for i in range(12) if i not in test_indices], dtype=np.int64)
    train_ids = {poses[names[i]]["id"] for i in train_indices}
    sparse = read_points(downloads.file(entries[2]), train_ids)
    point_ids = sorted(sparse)
    if len(point_ids) > 20000:
        point_ids = [point_ids[i] for i in np.linspace(0, len(point_ids)-1, 20000, dtype=int)]
    images, Ks, views, details = [], [], [], []
    for name in names:
        raw = downloads.file(available[name])
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Unreadable photo: {name}")
        h, w = bgr.shape[:2]
        K, distortion = intrinsics(cameras[poses[name]["camera_id"]], w, h)
        if np.any(distortion):
            bgr = cv2.undistort(bgr, K, distortion, newCameraMatrix=K)
        out_w = min(w, 400)
        out_h = max(1, round(h * out_w / w))
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        K[0] *= out_w / w
        K[1] *= out_h / h
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        Ks.append(K.astype(np.float32))
        views.append(viewmat(poses[name]))
        details.append({"name": name, "image_id": poses[name]["id"],
                        "camera_id": poses[name]["camera_id"], "source_size": [w, h],
                        "output_size": [out_w, out_h], "undistorted": bool(np.any(distortion))})
    arrays = {
        "images": np.stack(images), "Ks": np.stack(Ks), "viewmats": np.stack(views),
        "points": np.array([sparse[i][0] for i in point_ids], dtype=np.float32),
        "colors": np.array([sparse[i][1] for i in point_ids], dtype=np.float32) / 255,
        "train_indices": train_indices, "test_indices": test_indices}
    part = root / "training.npz.part"
    try:
        with part.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        part.replace(root / "training.npz")
    finally:
        part.unlink(missing_ok=True)
    metadata = {
        "dataset": "Mip-NeRF 360", "scene": "stump", "repository": REPO,
        "revision": REVISION, "upstream": "https://jonbarron.info/mipnerf360/",
        "citation": "Barron et al., Mip-NeRF 360, CVPR 2022",
        "license_note": "Mirror declares apache-2.0; consult upstream dataset terms for image rights.",
        "coordinate_system": "OpenCV world-to-camera: +X right, +Y down, +Z forward",
        "units": "Original COLMAP reconstruction units; no rescaling or recentering",
        "selection": "First 12 available registered image IDs, ascending; positions 3 and 8 held out",
        "point_selection": "Points observed by at least one training image, deterministic cap 20000",
        "holdout_note": "Photos withheld from training; upstream COLMAP reconstruction used all views.",
        "images": details, "cameras": {str(k): {"model": MODELS[v[0]][0],
            "width": v[1], "height": v[2], "parameters": v[3]} for k, v in cameras.items()},
        "point_count": len(point_ids), "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(), "download_limit_bytes": LIMIT,
        "download_budget_charged_bytes": downloads.state["charged_bytes"],
        "discovery_allowance_bytes": DISCOVERY_ALLOWANCE,
        "files": downloads.state["files"],
        "training_npz_sha256": hashlib.sha256((root / "training.npz").read_bytes()).hexdigest()}
    write_json(root / "provenance.json", metadata)
    print(json.dumps({"shape": arrays["images"].shape, "points": len(point_ids),
                      "charged_MiB": downloads.state["charged_bytes"] / 1024**2}))


if __name__ == "__main__":
    prepare()
