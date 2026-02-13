#!/usr/bin/env python3
"""Capture deterministic snapshots of Pi-Guy face routes for build artifacts."""

from __future__ import annotations

import argparse
import base64
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
        (f"{base_url}/", Path("artifacts/gui/dashboard-fixed.png")),
        (f"{base_url}/face", Path("artifacts/gui/face-fixed.png")),
        (f"{base_url}/face?embed=1", Path("artifacts/gui/face-embed-fixed.png")),
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        for url, output in targets:
            capture_page(page, url, output, wait_ms)
        browser.close()


def _as_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing snapshot required for overview: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_frontends_overview(output: Path) -> None:
    tiles = [
        ("/", Path("artifacts/gui/dashboard-fixed.png")),
        ("/face", Path("artifacts/gui/face-fixed.png")),
        ("/face?embed=1", Path("artifacts/gui/face-embed-fixed.png")),
    ]

    cards = "\n".join(
        (
            "<article class='tile'>"
            f"<h2>{label}</h2>"
            f"<img alt='{label} snapshot' src='{_as_data_url(image_path)}' />"
            "</article>"
        )
        for label, image_path in tiles
    )

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <style>
          body {{
            margin: 0;
            padding: 24px;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
          }}
          h1 {{
            margin: 0 0 16px;
            font-size: 28px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
          }}
          .tile {{
            background: #111827;
            border: 1px solid #334155;
            border-radius: 10px;
            overflow: hidden;
          }}
          .tile h2 {{
            margin: 0;
            padding: 10px 12px;
            font-size: 20px;
            font-weight: 600;
            border-bottom: 1px solid #334155;
          }}
          .tile img {{
            display: block;
            width: 100%;
            height: auto;
            background: #020617;
          }}
        </style>
      </head>
      <body>
        <h1>Pi-Guy frontend snapshots overview</h1>
        <section class=\"grid\">{cards}</section>
      </body>
    </html>
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1240})
        page.set_content(html, wait_until="load")
        page.screenshot(path=str(output), full_page=True)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture Pi-Guy face snapshots")
    parser.add_argument("--url", default=f"{DEFAULT_BASE_URL}/face", help="Page URL to capture")
    parser.add_argument("--output", default="artifacts/face-snapshot.png", help="Output PNG path")
    parser.add_argument(
        "--standard-set",
        action="store_true",
        help=(
            "Capture /, /face, and /face?embed=1, then generate artifacts/gui/frontends-overview.png"
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
        create_frontends_overview(Path("artifacts/gui/frontends-overview.png"))
    else:
        capture(args.url, Path(args.output), args.wait_ms)
