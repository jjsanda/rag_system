"""Capture Streamlit UI screenshots (and a short GIF) by driving the app with Playwright.

Assumes the API (with the sample corpus ingested) and the Streamlit UI are already running
(see scripts/capture_media.sh). Best-effort: it always produces the home screenshot; the
answer screenshot and GIF are captured if the interaction succeeds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

UI_URL = "http://127.0.0.1:8501"
QUESTION = "How many castes are there in a honeybee colony?"
ASSETS = Path("docs/assets")
FRAMES = ASSETS / "_frames"


def _frame(page, index: int) -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(FRAMES / f"{index:02d}.png"))


def _question_box(page):
    try:
        return page.get_by_label("Your question")
    except Exception:
        return page.get_by_placeholder("How many", exact=False)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 920}, device_scale_factor=1)
        page.goto(UI_URL, wait_until="networkidle", timeout=60_000)
        page.get_by_text("Ask a grounded question").wait_for(timeout=60_000)
        time.sleep(2.0)  # let the sidebar /info call and layout settle
        page.screenshot(path=str(ASSETS / "ui-home.png"))
        _frame(page, 0)

        try:
            box = _question_box(page)
            box.click()
            box.fill(QUESTION)
            time.sleep(0.5)
            _frame(page, 1)
            page.get_by_role("button", name="Ask").click()
            page.get_by_role("heading", name="Sources").wait_for(timeout=60_000)
            time.sleep(1.5)
            page.screenshot(path=str(ASSETS / "ui-answer.png"))
            _frame(page, 2)
            page.locator("summary").first.click(timeout=5_000)
            time.sleep(0.8)
            _frame(page, 3)
        except Exception as exc:  # best-effort; keep whatever we captured
            print(f"interaction step failed ({exc}); keeping captured frames", file=sys.stderr)

        browser.close()

    frame_paths = sorted(FRAMES.glob("*.png"))
    if len(frame_paths) >= 2:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        images[0].save(
            ASSETS / "demo.gif",
            save_all=True,
            append_images=images[1:],
            duration=1400,
            loop=0,
            optimize=True,
        )
    produced = sorted(p.name for p in ASSETS.glob("*.png")) + (
        ["demo.gif"] if (ASSETS / "demo.gif").exists() else []
    )
    print("captured:", produced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
