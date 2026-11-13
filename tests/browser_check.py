"""Run separately: python tests/browser_check.py (local server required)."""
from pathlib import Path
import os
from playwright.sync_api import sync_playwright

out = Path("outputs/qa")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(os.environ.get("FUSION_URL", "http://127.0.0.1:8765"))
    page.locator("summary").click()
    page.locator("#kittiRoot").fill("missing-kitti360-test-fixture")
    page.locator("#kittiSequence").fill("2013_05_28_drive_0000_sync")
    page.locator("#kittiPlan").click()
    page.wait_for_function("document.querySelector('#message').textContent.includes('dataset root')")
    page.locator("summary").click()
    page.locator("#demo").click()
    page.locator("#render").wait_for(state="visible")
    page.wait_for_function("!document.querySelector('#render').disabled")
    assert page.locator("#images img").count() == 4
    page.locator("#samplePlan").click()
    page.wait_for_function("document.querySelector('#sampleResult').textContent.includes('MiB')")
    assert "Dự kiến" in page.locator("#sampleResult").inner_text()
    page.locator("#render").click()
    page.wait_for_function("document.querySelector('#coverage').textContent.includes('%')")
    coverage4 = float(page.locator("#coverage").inner_text().rstrip("%"))
    page.locator("#cameras input[value=left]").uncheck()
    page.locator("#cameras input[value=right]").uncheck()
    page.locator("#render").click()
    page.wait_for_function("document.querySelector('#coverage').textContent.includes('%')")
    coverage2 = float(page.locator("#coverage").inner_text().rstrip("%"))
    assert coverage2 < coverage4
    page.locator("#cameras input[value=rear]").uncheck()
    page.locator("#render").click()
    page.wait_for_function("document.querySelector('#coverage').textContent.includes('%')")
    coverage1 = float(page.locator("#coverage").inner_text().rstrip("%"))
    assert coverage1 < coverage2
    page.locator("#demo").click()
    page.wait_for_function("document.querySelectorAll('#cameras input:checked').length === 4")
    page.locator("#render").click()
    page.wait_for_function("document.querySelector('#coverage').textContent.includes('%')")
    with page.expect_download() as download:
        page.locator("#colab").click()
    download.value.save_as(out / "colab.zip")
    page.wait_for_function("document.querySelector('#message').textContent.includes('bundle')")
    for width, name in [(1440, "desktop"), (768, "tablet"), (390, "mobile")]:
        page.set_viewport_size({"width": width, "height": 1000})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    page.locator("#mask").check()
    assert page.locator("#panorama").is_visible()
    page.locator("#pinhole").click()
    page.wait_for_function("document.querySelector('#cameras').textContent.includes('pinhole')")
    page.locator("#render").click()
    page.wait_for_function("document.querySelector('#coverage').textContent.includes('%')")
    assert not errors, errors
    print(f"Browser PASS; fisheye coverage 1/2/4: {coverage1}/{coverage2}/{coverage4}; no overflow")
    browser.close()
