#!/usr/bin/env python3
"""Run conversational/multimodal sanity checks and capture avatar mood snapshots.

Usage:
  python scripts/emotion_modalities_playwright.py
"""

from __future__ import annotations

import pathlib
import time

from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:5000"
ARTIFACT_DIR = pathlib.Path("artifacts/emotion-modalities")
VIEWPORT = {"width": 1680, "height": 1200}


def capture(page, name: str) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(ARTIFACT_DIR / name), full_page=True)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_selector("#chat-input")
        page.wait_for_timeout(1300)

        # Ensure realtime bridge starts (companion pipeline state changes).
        page.click("#duplex-auto")
        page.wait_for_timeout(900)

        # Mood-provoking prompts to force different avatar states.
        prompts = [
            (
                "happy-chat.png",
                "You just helped me recover lost project files and I am genuinely thrilled. Celebrate with me in two energetic lines.",
            ),
            (
                "sad-chat.png",
                "I accidentally deleted a month of photos and feel pretty sad right now. Reply with empathy and a simple next step.",
            ),
            (
                "angry-chat.png",
                "The build failed again after five retries and I am frustrated. Give me a sharp but constructive triage plan.",
            ),
            (
                "thinking-chat.png",
                "Think carefully: compare vector search vs keyword search in exactly three concise bullets.",
            ),
            (
                "surprised-chat.png",
                "Wow, the tiny edge device just outperformed the cloud baseline. React with surprise and explain one plausible reason.",
            ),
        ]

        for file_name, text in prompts:
            page.fill("#chat-input", text)
            page.click("#chat-send")
            page.wait_for_timeout(2200)
            capture(page, file_name)

        # Exercise manual avatar controls/modalities quickly.
        page.click("#duplex-blink")
        page.wait_for_timeout(450)
        page.evaluate(
            """async () => {
                await fetch('/api/look?x=0.72&y=0.34');
            }"""
        )
        page.wait_for_timeout(650)

        # Toggle vision mode and run one multimodal-ish turn without image.
        page.evaluate(
            """() => {
                const vision = document.querySelector('#chat-vision');
                if (vision) {
                    vision.checked = true;
                    vision.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }"""
        )
        page.fill(
            "#chat-input",
            "Vision pipeline check: if no image is attached, report fallback behavior in one sentence.",
        )
        page.click("#chat-send")
        page.wait_for_timeout(1700)
        page.evaluate(
            """() => {
                const vision = document.querySelector('#chat-vision');
                if (vision) {
                    vision.checked = false;
                    vision.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }"""
        )

        # Verify realtime API responds to turn + state queries.
        result = page.evaluate(
            """async () => {
                const started = await fetch('/api/realtime/session/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ profile: 'full-duplex' }),
                }).then((r) => r.json());

                const turn = await fetch('/api/realtime/turn', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: started.session_id,
                        text: 'Cross-modality health check: summarize link status, mood, and action policy.',
                        modality: {
                            vision: 'camera feed simulated',
                            audio: 'microphone idle',
                            sensor: 'cpu nominal'
                        }
                    }),
                }).then((r) => r.json());

                const state = await fetch(`/api/realtime/state?session_id=${started.session_id}`).then((r) => r.json());
                return { started, turn, state };
            }"""
        )

        capture(page, "final-modalities.png")

        report_path = ARTIFACT_DIR / "summary.txt"
        report_path.write_text(
            "\n".join(
                [
                    f"timestamp={int(time.time())}",
                    f"start_status={result['started'].get('status')}",
                    f"turn_status={result['turn'].get('status')}",
                    f"state_status={result['state'].get('status')}",
                    f"turn_mood={result['turn'].get('mood')}",
                    f"thought_events={len(result['turn'].get('thought_events', []))}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        browser.close()


if __name__ == "__main__":
    main()
