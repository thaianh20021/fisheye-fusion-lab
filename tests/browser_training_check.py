"""Actual CUDA/UI test, opt-in: python tests/browser_training_check.py."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

url = os.environ.get("FUSION_URL", "http://127.0.0.1:8767")
out = Path("outputs/qa")
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    results = []
    for dataset, selected, steps in [
        ("examples/real_small_scene/training.npz", [], 1200),
        ("examples/synthetic_training_rig/manifest.json", ["front", "rear"], 100),
        ("examples/synthetic_training_rig/manifest.json", ["front", "right", "rear", "left"], 100),
    ]:
        page.locator("#trainingDataset").fill(dataset)
        page.locator("#trainingInspect").click()
        page.wait_for_function("!document.querySelector('#trainingSettings').disabled")
        assert not page.locator("#trainingError").inner_text()
        if selected:
            page.locator("#trainingCameraCount").select_option(str(len(selected)))
            for label in page.locator("#trainingCameras label").all():
                label.locator("input").set_checked(label.inner_text().split(" · ")[0] in selected)
        page.locator("#trainingSteps").fill(str(steps))
        with page.expect_response(lambda response: response.url.endswith("/api/training")
                                  and response.request.method == "POST") as response:
            page.locator("#trainingStart").click()
        job = response.value.json()
        assert "job_id" in job, job
        page.reload()
        page.wait_for_function(
            "() => /Hoàn tất|Thất bại|Đã dừng/.test(document.querySelector('#trainingStatus').textContent)",
            timeout=120000)
        assert "Hoàn tất" in page.locator("#trainingStatus").inner_text(), page.locator("#trainingLog").inner_text()
        state = page.request.get(f"{url}/api/training/{job['job_id']}").json()
        assert state["state"] == "succeeded" and state["step"] == steps
        metrics = page.request.get(f"{url}/api/training/{job['job_id']}/artifacts/metrics.json").json()
        assert len(metrics["test_indices"]) == (len(selected) or 2)
        page.wait_for_function("""() => [...document.querySelectorAll('#trainingPreviews img')]
            .length === 2 && [...document.querySelectorAll('#trainingPreviews img')]
            .every(img => img.complete && img.naturalWidth > 0)""")
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#trainingForm").screenshot(path=str(out / f"train-{len(selected)}-{width}.png"))
        results.append({"job_id": job["job_id"], "cameras": len(selected), "steps": steps})
        print("Train PASS", results[-1], flush=True)
    # Start a longer real job and cancel through the UI, including the process tree.
    page.locator("#trainingDataset").fill("examples/real_small_scene/training.npz")
    page.locator("#trainingInspect").click()
    page.wait_for_function("!document.querySelector('#trainingSettings').disabled")
    page.locator("#trainingSteps").fill("10000")
    page.locator("#trainingStart").click()
    page.wait_for_function("!document.querySelector('#trainingCancel').disabled")
    page.locator("#trainingCancel").click()
    page.wait_for_function("document.querySelector('#trainingStatus').textContent.includes('Đã dừng')",
                           timeout=30000)
    assert not errors, errors
    browser.close()
(out / "training-ui-results.json").write_text(json.dumps(results, indent=2))
print("PASS real CUDA train, 2/4-camera synthetic rig, reload, cancel, desktop/mobile")
