#!/usr/bin/env python3
"""Capture deterministic snapshots of the Pi-Guy UI for build artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


VIEWPORT = {"width": 1600, "height": 900}


def capture(url: str, output: Path, wait_ms: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        # Some Pi-Guy pages keep background requests open, so waiting for
        # "networkidle" can hang and prevent snapshots from being produced.
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_load_state("load")
        page.wait_for_selector("body")
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=str(output), full_page=True)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a Pi-Guy page snapshot")
    parser.add_argument("--url", default="http://127.0.0.1:5000/face", help="Page URL to capture")
    parser.add_argument("--output", default="artifacts/face-snapshot.png", help="Output PNG path")
    parser.add_argument("--wait-ms", type=int, default=2200, help="Extra wait before capture")
    args = parser.parse_args()

    capture(args.url, Path(args.output), args.wait_ms)
