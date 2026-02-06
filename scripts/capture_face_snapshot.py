#!/usr/bin/env python3
"""Capture deterministic snapshots of Pi-Guy face routes for build artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


VIEWPORT = {"width": 1920, "height": 1080}
DEFAULT_BASE_URL = "http://127.0.0.1:5000"


def capture_page(page, url: str, output: Path, wait_ms: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Some Pi-Guy pages keep background requests open, so waiting for
    # "networkidle" can hang and prevent snapshots from being produced.
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_load_state("load")
    page.wait_for_selector("body")
    page.wait_for_timeout(wait_ms)
    page.screenshot(path=str(output), full_page=True)


def capture(url: str, output: Path, wait_ms: int) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        capture_page(page, url, output, wait_ms)
        browser.close()


def capture_standard_face_set(base_url: str, wait_ms: int) -> None:
    targets = [
        (f"{base_url}/face", Path("artifacts/gui/face-fixed.png")),
        (f"{base_url}/face?embed=1", Path("artifacts/gui/face-embed-fixed.png")),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        for url, output in targets:
            capture_page(page, url, output, wait_ms)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Pi-Guy face snapshots")
    parser.add_argument("--url", default=f"{DEFAULT_BASE_URL}/face", help="Page URL to capture")
    parser.add_argument("--output", default="artifacts/face-snapshot.png", help="Output PNG path")
    parser.add_argument(
        "--standard-set",
        action="store_true",
        help=(
            "Capture /face and /face?embed=1 into "
            "artifacts/gui/face-fixed.png and artifacts/gui/face-embed-fixed.png"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL used by --standard-set",
    )
    parser.add_argument("--wait-ms", type=int, default=2200, help="Extra wait before capture")
    args = parser.parse_args()

    if args.standard_set:
        capture_standard_face_set(args.base_url, args.wait_ms)
    else:
        capture(args.url, Path(args.output), args.wait_ms)
