# GUI sanity test plan (static QA + automated browser run)

## Static QA findings

1. **Chat calls can be blocked in production API-key mode unless the frontend injects `X-API-Key`.**
   - `before_request` enforces API auth for all `/api/*` routes when `PIGUY_API_KEY` is set, but the browser `fetch('/api/chat')` and `fetch('/api/vision')` requests send only `Content-Type`. In production with API key enabled, chat/vision will return `401` unless a proxy or frontend change adds this header.
2. **Voice input depends on browser SpeechRecognition and degrades gracefully only with a chat log message.**
   - The UI reports unsupported browsers via a system message, which is acceptable but not discoverable until clicking the button.
3. **Vision toggle requires both checkbox + attached file.**
   - If the checkbox is enabled but no file is attached, logic falls back to text chat; this is safe behavior but may surprise users expecting a validation warning.
4. **Chat state retention is intentionally short.**
   - In-memory browser context keeps the last 12 chat messages for `/api/chat`; this controls payload growth but can reduce continuity for long exchanges.

## Automated sanity workflow

Use the Playwright script in `scripts/gui_sanity_playwright.py` to exercise the major chat controls/fields and run a short math+English conversation prompt sequence.

### What it exercises

- `#chat-mute` checkbox
- `#chat-vision` checkbox
- `#chat-input` textarea
- `#chat-send` button
- `#chat-clear` button
- Conversation prompts covering math + English-language sanity content

### Run

```bash
python scripts/gui_sanity_playwright.py
```

The script expects the app to be available at `http://127.0.0.1:5000` and writes a screenshot to `artifacts/gui-sanity-full.png`.
