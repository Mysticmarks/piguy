#!/usr/bin/env python3
"""Playwright GUI sanity test for Pi-Guy dashboard chat controls.

Usage:
  python scripts/gui_sanity_playwright.py
"""

from playwright.sync_api import sync_playwright


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
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

        page.screenshot(path="artifacts/gui-sanity-full.png", full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
