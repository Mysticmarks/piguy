#!/usr/bin/env python3
"""Playwright GUI sanity test for Pi-Guy dashboard + face snapshot flow.

Usage:
  python scripts/gui_sanity_playwright.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright


VIEWPORT = {"width": 1920, "height": 1080}
WAIT_MS = 2200


def capture_face_routes(page) -> None:
    page.goto("http://127.0.0.1:5000/face", wait_until="domcontentloaded")
    page.wait_for_load_state("load")
    page.wait_for_selector("body")
    page.wait_for_timeout(WAIT_MS)
    page.screenshot(path="artifacts/gui/face-fixed.png", full_page=True)

    page.goto("http://127.0.0.1:5000/face?embed=1", wait_until="domcontentloaded")
    page.wait_for_load_state("load")
    page.wait_for_selector("body")
    page.wait_for_timeout(WAIT_MS)
    page.screenshot(path="artifacts/gui/face-embed-fixed.png", full_page=True)


def main() -> None:
    artifacts_dir = Path("artifacts/gui")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto("http://127.0.0.1:5000", wait_until="networkidle")

        # Field/control sanity coverage
        page.check("#chat-mute")
        page.check("#chat-vision")
        page.uncheck("#chat-vision")

        # Conversation sanity: math + English language prompt
        page.fill(
            "#chat-input",
            "Let's do a sanity test: explain why 0.999... equals 1, then connect "
            "that to ambiguity in English sentence parsing with one concise example.",
        )
        page.click("#chat-send")
        page.wait_for_timeout(2500)

        # Continue conversation
        page.fill(
            "#chat-input",
            "Now give me a short 3-line dialogue where one line uses a pun and "
            "one line contains a valid algebraic identity.",
        )
        page.click("#chat-send")
        page.wait_for_timeout(2500)

        # Clear + final ping
        page.click("#chat-clear")
        page.fill(
            "#chat-input",
            "Final sanity ping: summarize what this dashboard is trying to do in one sentence.",
        )
        page.click("#chat-send")
        page.wait_for_timeout(2000)

        page.screenshot(path="artifacts/gui/dashboard-fixed.png", full_page=True)
        capture_face_routes(page)
        browser.close()


if __name__ == "__main__":
    main()
