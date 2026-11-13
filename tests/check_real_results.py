"""Run after training: python tests/check_real_results.py."""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
run = os.environ.get("EXPERIMENT_RUN", "baseline")
out = root / "outputs" / {"baseline": "real-gs-run", "geer": "geer-run"}[run]
metrics = json.loads((out / "metrics.json").read_text())
before = np.mean(list(metrics["heldout_psnr_before"].values()))
after = np.mean(list(metrics["heldout_psnr_after"].values()))
assert after > before, (before, after)
frames = [cv2.imread(str(out / name)) for name in metrics["novel_views"]]
assert all(frame is not None and frame.std() > 5 for frame in frames)
assert np.abs(frames[0].astype(float) - frames[-1]).mean() > 1
with (out / "scene.ply").open("rb") as stream:
    header = []
    while True:
        line = stream.readline()
        assert line, "Truncated PLY"
        header.append(line)
        if line == b"end_header\n":
            break
    assert f"element vertex {metrics['gaussians']}\n".encode() in header
    values = np.frombuffer(stream.read(), dtype="<f4")
    assert values.size == metrics["gaussians"] * 17
    assert np.isfinite(values).all()

qa = root / "outputs" / "qa"
qa.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(os.environ.get("FUSION_URL", "http://127.0.0.1:8767"))
    page.locator("#experimentRun").select_option(run)
    page.locator("#loadExperiment").click()
    page.locator("#experimentNovelView").wait_for()
    page.wait_for_function("""() => [...document.querySelectorAll('#experimentContent img')]
        .every(img => img.complete && img.naturalWidth > 0)""")
    page.locator("#experimentNovelView").fill("8")
    page.wait_for_function("""() => document.querySelector('.experiment-novel img')
        .src.includes('novel-08.png')""")
    with page.expect_download() as download:
        page.locator('a[download="scene.ply"]').click()
    assert download.value.failure() is None
    for width in (1440, 768, 390):
        page.set_viewport_size({"width": width, "height": 1000})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#realExperiment").screenshot(path=str(qa / f"{run}-{width}.png"))
    assert not errors, errors
    browser.close()
print(f"PASS: PSNR {before:.2f} -> {after:.2f}; PLY finite; novel views distinct; UI desktop/mobile")
