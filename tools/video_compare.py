"""Stack two videos of the same drive (top / bottom) with a label on each, for side-by-side comparison.

    <python with opencv> tools/video_compare.py a.mp4 "label A" b.mp4 "label B" --out compare.mp4

Frames are paired one to one (the shorter video sets the length); B is resized to A's width.
H.264 when imageio-ffmpeg is installed, else mp4v.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def label(frame: np.ndarray, text: str) -> np.ndarray:
    h = max(28, frame.shape[0] // 18)
    scale = h / 34
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (0, 0), (tw + 24, h + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("label_a")
    p.add_argument("b")
    p.add_argument("label_b")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ca, cb = cv2.VideoCapture(a.a), cv2.VideoCapture(a.b)
    fps = ca.get(cv2.CAP_PROP_FPS) or 20
    out = Path(a.out)
    writer, n = None, 0
    while True:
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not (ok_a and ok_b):
            break
        if fb.shape[1] != fa.shape[1]:
            fb = cv2.resize(fb, (fa.shape[1], round(fb.shape[0] * fa.shape[1] / fb.shape[1])), interpolation=cv2.INTER_AREA)
        frame = np.vstack([label(fa, a.label_a), label(fb, a.label_b)])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
    if writer is None:
        raise SystemExit("could not read both videos")
    writer.release()
    try:
        import imageio_ffmpeg

        tmp = out.with_suffix(".h264.mp4")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(out),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-movflags", "+faststart", str(tmp)])
        if r.returncode == 0:
            tmp.replace(out)
    except ImportError:
        pass
    print(f"wrote {out}: {n} frames, {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    main()
